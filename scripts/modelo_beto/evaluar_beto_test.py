from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MAX_LENGTH = 128
BATCH_SIZE = 16

LABELS = [
    "Contratos",
    "Facturas",
    "Colaboraciones",
    "Clientes",
    "Publicidad",
    "Varios",
]

LABEL_TO_ID = {
    label: index for index, label in enumerate(LABELS)
}

ID_TO_LABEL = {
    index: label for label, index in LABEL_TO_ID.items()
}


def bytes_to_mib(value: int) -> float:
    return round(value / (1024**2), 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    return value


def validate_test_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "id",
        "text",
        "label",
        "label_id",
        "source",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en test.csv: {sorted(missing_columns)}"
        )

    result = df[
        [
            "id",
            "text",
            "label",
            "label_id",
            "source",
        ]
    ].copy()

    result["text"] = result["text"].fillna("").astype(str).str.strip()
    result["label"] = result["label"].fillna("").astype(str).str.strip()
    result["source"] = result["source"].fillna("").astype(str).str.strip()

    result["label_id"] = pd.to_numeric(
        result["label_id"],
        errors="raise",
    ).astype(int)

    if result["text"].eq("").any():
        raise ValueError("Existen textos vacíos en test.csv.")

    invalid_labels = sorted(set(result["label"]) - set(LABELS))

    if invalid_labels:
        raise ValueError(
            f"Se encontraron etiquetas no oficiales: {invalid_labels}"
        )

    expected_ids = result["label"].map(LABEL_TO_ID)

    if not expected_ids.equals(result["label_id"]):
        mismatches = int((expected_ids != result["label_id"]).sum())
        raise ValueError(
            f"Hay {mismatches} filas con label y label_id incoherentes."
        )

    if result["id"].duplicated().any():
        raise ValueError("Existen IDs duplicados en test.csv.")

    return result


def run_batch(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device, non_blocking=True)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            outputs = model(**encoded)

    return outputs.logits.detach().float().cpu()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    test_path = project_root / "data" / "processed" / "test.csv"
    model_dir = (
        project_root
        / "outputs"
        / "models"
        / "mailpyme_beto_model_v2"
    )
    metrics_dir = project_root / "outputs" / "metrics"
    plots_dir = project_root / "outputs" / "plots"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        raise FileNotFoundError(
            f"No se encontró el conjunto de prueba: {test_path}"
        )

    if not model_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado: {model_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. Se cancela la evaluación."
        )

    print("=" * 76)
    print("MAILPYME AI - EVALUACIÓN FINAL DE BETO")
    print("=" * 76)
    print(f"Modelo: {model_dir}")
    print(f"Test: {test_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"Max length: {MAX_LENGTH}")

    test_df = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
    )
    test_df = validate_test_dataframe(test_df)

    print(f"\nCorreos de prueba: {len(test_df)}")

    print("\n1. Cargando tokenizador y modelo final...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        use_fast=True,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    model.eval()

    device = torch.device("cuda:0")
    model.to(device)

    if model.config.num_labels != len(LABELS):
        raise ValueError(
            f"El modelo tiene {model.config.num_labels} salidas; "
            f"se esperaban {len(LABELS)}."
        )

    # Calentamiento de GPU para no incluir inicialización CUDA en el tiempo.
    warmup_texts = test_df["text"].iloc[: min(8, len(test_df))].tolist()
    _ = run_batch(
        warmup_texts,
        tokenizer,
        model,
        device,
    )
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    all_logits: list[torch.Tensor] = []

    print("2. Ejecutando inferencia sobre test.csv...")

    start = time.perf_counter()

    for batch_start in range(0, len(test_df), BATCH_SIZE):
        batch_texts = test_df["text"].iloc[
            batch_start : batch_start + BATCH_SIZE
        ].tolist()

        batch_logits = run_batch(
            batch_texts,
            tokenizer,
            model,
            device,
        )

        all_logits.append(batch_logits)

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start

    logits = torch.cat(all_logits, dim=0)

    if logits.shape != (len(test_df), len(LABELS)):
        raise RuntimeError(
            f"Forma de logits inesperada: {tuple(logits.shape)}"
        )

    if not torch.isfinite(logits).all().item():
        raise RuntimeError(
            "Los logits contienen valores NaN o infinitos."
        )

    probabilities = torch.softmax(logits, dim=-1).numpy()
    predicted_ids = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    true_ids = test_df["label_id"].to_numpy()

    accuracy = accuracy_score(true_ids, predicted_ids)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            true_ids,
            predicted_ids,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            true_ids,
            predicted_ids,
            labels=list(range(len(LABELS))),
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        true_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
    )

    predicted_labels = [
        ID_TO_LABEL[int(label_id)]
        for label_id in predicted_ids
    ]

    predictions_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "source": test_df["source"],
            "label_real": test_df["label"],
            "label_predicha": predicted_labels,
            "confianza": np.round(confidences, 6),
            "correcto": predicted_ids == true_ids,
        }
    )

    errors_df = predictions_df.loc[
        ~predictions_df["correcto"]
    ].copy()

    correct_mask = predictions_df["correcto"].to_numpy()

    mean_confidence = float(confidences.mean())
    mean_confidence_correct = (
        float(confidences[correct_mask].mean())
        if correct_mask.any()
        else None
    )
    mean_confidence_errors = (
        float(confidences[~correct_mask].mean())
        if (~correct_mask).any()
        else None
    )

    samples_per_second = len(test_df) / elapsed_seconds
    milliseconds_per_email_batch_equivalent = (
        elapsed_seconds / len(test_df) * 1000
    )

    peak_memory_mib = bytes_to_mib(
        torch.cuda.max_memory_allocated()
    )

    summary = {
        "status": "OK",
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_dir": str(model_dir),
        "test_file": str(test_path),
        "test_sha256": sha256_file(test_path),
        "num_test_samples": int(len(test_df)),
        "metrics": {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
        },
        "confidence": {
            "mean_all": mean_confidence,
            "mean_correct": mean_confidence_correct,
            "mean_errors": mean_confidence_errors,
        },
        "errors": {
            "count": int(len(errors_df)),
            "rate": float(len(errors_df) / len(test_df)),
        },
        "performance": {
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "elapsed_seconds": float(elapsed_seconds),
            "samples_per_second": float(samples_per_second),
            "batch_equivalent_ms_per_email": float(
                milliseconds_per_email_batch_equivalent
            ),
            "gpu_peak_memory_mib": peak_memory_mib,
            "note": (
                "Esta medición corresponde a evaluación por lotes. "
                "La latencia individual se medirá en el siguiente paso."
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_compiled": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "labels": LABELS,
    }

    metrics_csv = pd.DataFrame(
        [
            {"metric": "accuracy", "value": accuracy},
            {"metric": "precision_macro", "value": precision_macro},
            {"metric": "recall_macro", "value": recall_macro},
            {"metric": "f1_macro", "value": f1_macro},
            {
                "metric": "precision_weighted",
                "value": precision_weighted,
            },
            {
                "metric": "recall_weighted",
                "value": recall_weighted,
            },
            {
                "metric": "f1_weighted",
                "value": f1_weighted,
            },
            {
                "metric": "num_test_samples",
                "value": len(test_df),
            },
            {
                "metric": "num_errors",
                "value": len(errors_df),
            },
            {
                "metric": "mean_confidence",
                "value": mean_confidence,
            },
            {
                "metric": "samples_per_second",
                "value": samples_per_second,
            },
            {
                "metric": "gpu_peak_memory_mib",
                "value": peak_memory_mib,
            },
        ]
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.index.name = "label"
    report_df = report_df.reset_index()

    matrix_df = pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS,
    )
    matrix_df.index.name = "real/predicha"

    metrics_json_path = (
        metrics_dir / "metricas_test_beto_v2.json"
    )
    metrics_csv_path = (
        metrics_dir / "metricas_test_beto_v2.csv"
    )
    report_path = (
        metrics_dir
        / "reporte_clasificacion_test_beto_v2.csv"
    )
    matrix_csv_path = (
        metrics_dir
        / "matriz_confusion_test_beto_v2.csv"
    )
    predictions_path = (
        metrics_dir
        / "predicciones_test_beto_v2.csv"
    )
    errors_path = (
        metrics_dir
        / "errores_test_beto_v2.csv"
    )
    matrix_plot_path = (
        plots_dir
        / "matriz_confusion_test_beto_v2.png"
    )

    with metrics_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_serializable(summary),
            file,
            ensure_ascii=False,
            indent=2,
        )

    metrics_csv.to_csv(
        metrics_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    report_df.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    matrix_df.to_csv(
        matrix_csv_path,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    errors_df.to_csv(
        errors_path,
        index=False,
        encoding="utf-8-sig",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS,
    )

    figure, axis = plt.subplots(figsize=(10, 8))
    display.plot(
        ax=axis,
        values_format="d",
        colorbar=False,
        xticks_rotation=35,
    )
    axis.set_title(
        "Matriz de confusión - BETO MailPyme AI v2"
    )
    figure.tight_layout()
    figure.savefig(
        matrix_plot_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nResultados finales de TEST:")
    print(f"  Accuracy:             {accuracy:.6f}")
    print(f"  Precision macro:      {precision_macro:.6f}")
    print(f"  Recall macro:         {recall_macro:.6f}")
    print(f"  F1 macro:             {f1_macro:.6f}")
    print(f"  Precision ponderada:  {precision_weighted:.6f}")
    print(f"  Recall ponderado:     {recall_weighted:.6f}")
    print(f"  F1 ponderado:         {f1_weighted:.6f}")
    print(f"  Errores:              {len(errors_df)} de {len(test_df)}")
    print(f"  Confianza promedio:   {mean_confidence:.6f}")
    print(f"  Tiempo por lotes:     {elapsed_seconds:.4f} s")
    print(f"  Throughput:           {samples_per_second:.2f} correos/s")
    print(f"  Pico de VRAM:         {peak_memory_mib:.2f} MiB")

    print("\nReporte por categoría:")
    printable_report = report_df.loc[
        report_df["label"].isin(LABELS)
    ]
    print(
        printable_report[
            [
                "label",
                "precision",
                "recall",
                "f1-score",
                "support",
            ]
        ].to_string(index=False)
    )

    print("\nMatriz de confusión:")
    print(matrix_df.to_string())

    print("\nArchivos generados:")
    for path in [
        metrics_json_path,
        metrics_csv_path,
        report_path,
        matrix_csv_path,
        predictions_path,
        errors_path,
        matrix_plot_path,
    ]:
        print(f"  {path}")

    print(
        "\nEVALUACIÓN FINAL DE TEST COMPLETADA CORRECTAMENTE."
    )


if __name__ == "__main__":
    main()

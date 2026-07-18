from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 128
SMOKE_TEST_BATCH_SIZE = 8
SEED = 42

OFFICIAL_LABELS = [
    "Contratos",
    "Facturas",
    "Colaboraciones",
    "Clientes",
    "Publicidad",
    "Varios",
]

LABEL_TO_ID = {
    label: index for index, label in enumerate(OFFICIAL_LABELS)
}

ID_TO_LABEL = {
    index: label for label, index in LABEL_TO_ID.items()
}


def bytes_to_mib(value: int) -> float:
    return round(value / (1024**2), 2)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    train_path = project_root / "data" / "processed" / "train.csv"
    cache_dir = project_root / "cache" / "huggingface"
    metrics_dir = project_root / "outputs" / "metrics"

    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(
            f"No se encontró el conjunto de entrenamiento en: {train_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch no detecta CUDA. No se continuará con CPU."
        )

    device = torch.device("cuda:0")

    print("=" * 72)
    print("VERIFICACIÓN DE BETO EN GPU")
    print("=" * 72)
    print(f"Modelo base: {MODEL_NAME}")
    print(f"Dispositivo: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {bytes_to_mib(torch.cuda.get_device_properties(0).total_memory)} MiB")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Batch de prueba: {SMOKE_TEST_BATCH_SIZE}")

    print("\n1. Cargando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        cache_dir=str(cache_dir),
    )

    print("2. Descargando o cargando BETO con cabeza de 6 categorías...")
    print(
        "   Nota: es normal que Transformers avise que la capa clasificadora "
        "fue inicializada aleatoriamente."
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(OFFICIAL_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        cache_dir=str(cache_dir),
    )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print("3. Moviendo el modelo a la RTX 4060...")
    model.to(device)
    model.eval()
    torch.cuda.synchronize(device)

    memory_after_model = torch.cuda.memory_allocated(device)

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(train_df.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas en train.csv: {sorted(missing_columns)}"
        )

    sample_size = min(SMOKE_TEST_BATCH_SIZE, len(train_df))
    sample_df = train_df.sample(
        n=sample_size,
        random_state=SEED,
    )

    texts = sample_df["text"].fillna("").astype(str).tolist()

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    sequence_length_in_batch = int(encoded["input_ids"].shape[1])
    encoded = {
        key: value.to(device, non_blocking=True)
        for key, value in encoded.items()
    }

    print("4. Ejecutando inferencia de prueba en precisión mixta FP16...")

    torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            outputs = model(**encoded)

    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logits = outputs.logits

    expected_shape = (sample_size, len(OFFICIAL_LABELS))
    actual_shape = tuple(logits.shape)

    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Forma de logits inesperada: {actual_shape}. "
            f"Se esperaba: {expected_shape}."
        )

    if not torch.isfinite(logits).all().item():
        raise RuntimeError(
            "Los logits contienen valores NaN o infinitos."
        )

    predicted_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    predicted_labels = [ID_TO_LABEL[index] for index in predicted_ids]

    peak_memory = torch.cuda.max_memory_allocated(device)
    current_memory = torch.cuda.memory_allocated(device)

    result = {
        "status": "OK",
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "cuda_compiled": torch.version.cuda,
            "total_vram_mib": bytes_to_mib(
                torch.cuda.get_device_properties(0).total_memory
            ),
            "memory_after_model_mib": bytes_to_mib(memory_after_model),
            "current_memory_mib": bytes_to_mib(current_memory),
            "peak_memory_mib": bytes_to_mib(peak_memory),
        },
        "model": {
            "name": MODEL_NAME,
            "num_labels": len(OFFICIAL_LABELS),
            "labels": OFFICIAL_LABELS,
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
        },
        "smoke_test": {
            "batch_size": sample_size,
            "max_length": MAX_LENGTH,
            "dynamic_padded_length": sequence_length_in_batch,
            "logits_shape": list(actual_shape),
            "finite_logits": True,
            "elapsed_ms": round(elapsed_ms, 2),
            "predicted_label_counts_untrained_head": {
                label: predicted_labels.count(label)
                for label in OFFICIAL_LABELS
            },
            "note": (
                "Las predicciones no tienen valor evaluativo porque la cabeza "
                "de clasificación todavía no ha sido entrenada."
            ),
        },
    }

    output_path = metrics_dir / "verificacion_beto_gpu.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\nResultados:")
    print(f"  Parámetros totales:      {total_parameters:,}")
    print(f"  Parámetros entrenables:  {trainable_parameters:,}")
    print(f"  Longitud real del lote:  {sequence_length_in_batch} tokens")
    print(f"  Forma de logits:         {actual_shape}")
    print(f"  Logits finitos:          Sí")
    print(f"  Tiempo del lote:         {elapsed_ms:.2f} ms")
    print(f"  VRAM tras cargar modelo: {bytes_to_mib(memory_after_model)} MiB")
    print(f"  Pico de VRAM:            {bytes_to_mib(peak_memory)} MiB")
    print(f"\nEvidencia guardada en:\n  {output_path}")
    print("\nVerificación completada correctamente.")


if __name__ == "__main__":
    main()

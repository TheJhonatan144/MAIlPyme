from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 128
NUM_LABELS = 6
SEED = 42

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

TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_train_batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "max_length": MAX_LENGTH,
    "fp16": True,
    "tf32": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "metric_for_best_model": "f1_macro",
    "greater_is_better": True,
    "early_stopping_patience": 2,
    "save_total_limit": 2,
    "seed": SEED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prevalidación y entrenamiento de BETO para MailPyme AI."
    )
    parser.add_argument(
        "--mode",
        choices=["preflight", "train"],
        required=True,
        help=(
            "preflight valida el pipeline sin entrenar. "
            "train ejecuta el fine-tuning completo."
        ),
    )
    return parser.parse_args()


def configure_reproducibility() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

    # Mejora la reproducibilidad sin forzar algoritmos que podrían fallar.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"text", "label", "label_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {path.name}: {sorted(missing)}"
        )

    df = df[["text", "label", "label_id"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df["label_id"] = pd.to_numeric(
        df["label_id"],
        errors="raise",
    ).astype(int)

    if df["text"].eq("").any():
        raise ValueError(f"Hay textos vacíos en {path.name}.")

    invalid_labels = sorted(set(df["label"]) - set(LABELS))
    if invalid_labels:
        raise ValueError(
            f"Etiquetas no oficiales en {path.name}: {invalid_labels}"
        )

    invalid_ids = sorted(
        set(df["label_id"]) - set(range(NUM_LABELS))
    )
    if invalid_ids:
        raise ValueError(
            f"label_id inválidos en {path.name}: {invalid_ids}"
        )

    expected_ids = df["label"].map(LABEL_TO_ID)
    mismatches = int((expected_ids != df["label_id"]).sum())
    if mismatches:
        raise ValueError(
            f"Hay {mismatches} filas con label y label_id incoherentes "
            f"en {path.name}."
        )

    return df


def dataframe_to_dataset(
    df: pd.DataFrame,
    tokenizer,
) -> Dataset:
    dataset = Dataset.from_pandas(
        df[["text", "label_id"]],
        preserve_index=False,
    )
    dataset = dataset.rename_column("label_id", "labels")

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizando",
    )

    return tokenized


def calculate_class_weights(
    train_df: pd.DataFrame,
) -> tuple[torch.Tensor, dict[str, float]]:
    classes = np.arange(NUM_LABELS)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label_id"].to_numpy(),
    )

    tensor = torch.tensor(weights, dtype=torch.float32)
    mapping = {
        ID_TO_LABEL[index]: round(float(weight), 6)
        for index, weight in enumerate(weights)
    }

    return tensor, mapping


def make_weighted_loss(class_weights: torch.Tensor):
    def weighted_loss(
        outputs,
        labels: torch.Tensor,
        num_items_in_batch=None,
    ) -> torch.Tensor:
        logits = outputs.logits
        weights = class_weights.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        # Todos los microbatches de entrenamiento tienen el mismo tamaño
        # en este dataset; la media mantiene una escala estable.
        return F.cross_entropy(
            logits,
            labels,
            weight=weights,
            reduction="mean",
        )

    return weighted_loss


def compute_metrics(eval_prediction) -> dict[str, float]:
    predictions, labels = eval_prediction

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_ids = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, predicted_ids)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            labels,
            predicted_ids,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels,
            predicted_ids,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def bytes_to_mib(value: int) -> float:
    return round(value / (1024**2), 2)


def build_training_arguments(
    checkpoints_dir: Path,
    logs_dir: Path,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(checkpoints_dir),
        learning_rate=TRAINING_CONFIG["learning_rate"],
        per_device_train_batch_size=TRAINING_CONFIG[
            "per_device_train_batch_size"
        ],
        per_device_eval_batch_size=TRAINING_CONFIG[
            "per_device_eval_batch_size"
        ],
        gradient_accumulation_steps=TRAINING_CONFIG[
            "gradient_accumulation_steps"
        ],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_steps=TRAINING_CONFIG["warmup_ratio"],
        eval_strategy=TRAINING_CONFIG["eval_strategy"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        logging_strategy="steps",
        logging_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model=TRAINING_CONFIG[
            "metric_for_best_model"
        ],
        greater_is_better=TRAINING_CONFIG["greater_is_better"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        fp16=TRAINING_CONFIG["fp16"],
        tf32=TRAINING_CONFIG["tf32"],
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        eval_accumulation_steps=4,
        optim="adamw_torch",
        disable_tqdm=False,
        push_to_hub=False,
    )


def main() -> None:
    args = parse_args()
    configure_reproducibility()

    project_root = Path(__file__).resolve().parents[1]

    processed_dir = project_root / "data" / "processed"
    cache_dir = project_root / "cache" / "huggingface"
    model_root = project_root / "outputs" / "models"
    checkpoints_dir = model_root / "checkpoints_beto_v2"
    final_model_dir = model_root / "mailpyme_beto_model_v2"
    metrics_dir = project_root / "outputs" / "metrics"
    logs_dir = project_root / "logs" / "tensorboard"

    for directory in [
        cache_dir,
        checkpoints_dir,
        final_model_dir,
        metrics_dir,
        logs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. Se cancela para evitar entrenar en CPU."
        )

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("=" * 76)
    print("MAILPYME AI - PIPELINE DE ENTRENAMIENTO BETO")
    print("=" * 76)
    print(f"Modo: {args.mode}")
    print(f"Modelo: {MODEL_NAME}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Max length: {MAX_LENGTH}")

    print("\n1. Cargando divisiones...")
    train_df = load_split(processed_dir / "train.csv")
    validation_df = load_split(
        processed_dir / "validation.csv"
    )
    test_df = load_split(processed_dir / "test.csv")

    print(
        f"   Train={len(train_df)}, "
        f"Validation={len(validation_df)}, "
        f"Test={len(test_df)}"
    )

    print("\n2. Calculando pesos de clase con el conjunto train...")
    class_weights_tensor, class_weights_mapping = (
        calculate_class_weights(train_df)
    )

    for label, weight in class_weights_mapping.items():
        print(f"   {label:<16} {weight:.6f}")

    print("\n3. Cargando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        cache_dir=str(cache_dir),
    )

    print("\n4. Tokenizando divisiones...")
    train_dataset = dataframe_to_dataset(
        train_df,
        tokenizer,
    )
    validation_dataset = dataframe_to_dataset(
        validation_df,
        tokenizer,
    )
    test_dataset = dataframe_to_dataset(
        test_df,
        tokenizer,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    print("\n5. Cargando BETO con 6 categorías...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        cache_dir=str(cache_dir),
    )

    model.config.problem_type = "single_label_classification"

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    training_args = build_training_arguments(
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
    )

    weighted_loss = make_weighted_loss(
        class_weights_tensor
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        compute_loss_func=weighted_loss,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=TRAINING_CONFIG[
                    "early_stopping_patience"
                ]
            )
        ],
    )

    config_evidence = {
        "model_name": MODEL_NAME,
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
        "training_config": TRAINING_CONFIG,
        "class_weights": class_weights_mapping,
        "dataset_sizes": {
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_compiled": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_mib": bytes_to_mib(
                torch.cuda.get_device_properties(0).total_memory
            ),
        },
        "parameters": {
            "total": int(total_parameters),
            "trainable": int(trainable_parameters),
        },
    }

    config_path = metrics_dir / "config_entrenamiento_beto_v2.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            config_evidence,
            file,
            ensure_ascii=False,
            indent=2,
        )

    if args.mode == "preflight":
        print("\n6. Ejecutando evaluación de prevalidación...")
        print(
            "   Las métricas serán bajas porque la cabeza aún no está entrenada."
        )

        start = time.perf_counter()
        metrics = trainer.evaluate(
            eval_dataset=validation_dataset,
            metric_key_prefix="preflight",
        )
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start

        preflight_result = {
            "status": "OK",
            "elapsed_seconds": round(elapsed_seconds, 3),
            "metrics_untrained_head": {
                key: float(value)
                if isinstance(value, (int, float))
                else value
                for key, value in metrics.items()
            },
            "gpu_peak_memory_mib": bytes_to_mib(
                torch.cuda.max_memory_allocated()
            ),
            "note": (
                "Las métricas no tienen valor evaluativo. "
                "Solo validan el pipeline antes del entrenamiento."
            ),
        }

        preflight_path = (
            metrics_dir
            / "preflight_entrenamiento_beto_v2.json"
        )
        with preflight_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                preflight_result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print("\nPrevalidación completada:")
        print(f"  Tiempo: {elapsed_seconds:.2f} s")
        print(
            "  Pico de VRAM: "
            f"{preflight_result['gpu_peak_memory_mib']} MiB"
        )
        print(
            "  Forma funcional: dataset → tokenizer → "
            "collator → BETO → loss → métricas"
        )
        print(f"\nConfiguración guardada en:\n  {config_path}")
        print(f"\nEvidencia guardada en:\n  {preflight_path}")
        print("\nPREVALIDACIÓN APROBADA.")
        return

    print("\n6. Iniciando entrenamiento completo...")
    print(
        "   Este modo se ejecutará únicamente cuando el paso "
        "correspondiente sea autorizado."
    )

    train_start = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    train_elapsed_seconds = time.perf_counter() - train_start

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    training_summary = {
        "status": "TRAINING_COMPLETED",
        "elapsed_seconds": round(train_elapsed_seconds, 3),
        "train_metrics": {
            key: float(value)
            if isinstance(value, (int, float))
            else value
            for key, value in train_result.metrics.items()
        },
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "epochs_completed": trainer.state.epoch,
        "gpu_peak_memory_mib": bytes_to_mib(
            torch.cuda.max_memory_allocated()
        ),
        "final_model_dir": str(final_model_dir),
    }

    training_summary_path = (
        metrics_dir
        / "resumen_entrenamiento_beto_v2.json"
    )
    with training_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nEntrenamiento completado.")
    print(f"Modelo final: {final_model_dir}")
    print(f"Resumen: {training_summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
CANDIDATE_LENGTHS = [128, 192, 256, 320, 384, 512]
BATCH_SIZE = 64


def tokenize_lengths(
    texts: list[str],
    tokenizer,
    batch_size: int = BATCH_SIZE,
) -> list[int]:
    lengths: list[int] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])

    return lengths


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "dataset_preparado.csv"
    )
    cache_dir = project_root / "cache" / "huggingface"
    metrics_dir = project_root / "outputs" / "metrics"
    plots_dir = project_root / "outputs" / "plots"

    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset preparado en: {dataset_path}"
        )

    print("=" * 72)
    print("ANÁLISIS DE LONGITUD DE TOKENS CON BETO")
    print("=" * 72)
    print(f"Dataset: {dataset_path}")
    print(f"Tokenizador: {MODEL_NAME}")
    print("\nDescargando o cargando el tokenizador...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        cache_dir=str(cache_dir),
    )

    df = pd.read_csv(dataset_path, encoding="utf-8-sig")

    if "text" not in df.columns:
        raise ValueError(
            "El dataset no contiene la columna 'text'. "
            "Ejecuta primero preparar_dataset.py."
        )

    texts = df["text"].fillna("").astype(str).tolist()

    if not texts:
        raise ValueError("No hay textos para analizar.")

    print(f"Correos a analizar: {len(texts)}")
    print("Tokenizando sin truncar...")

    token_lengths = tokenize_lengths(texts, tokenizer)
    df["token_length"] = token_lengths

    lengths = np.asarray(token_lengths, dtype=np.int32)

    percentiles_requested = [50, 75, 90, 95, 97, 99, 100]
    percentile_values = np.percentile(
        lengths,
        percentiles_requested,
        method="nearest",
    )

    stats = {
        "model_name": MODEL_NAME,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_model_max_length": int(tokenizer.model_max_length),
        "num_emails": int(len(lengths)),
        "min_tokens": int(lengths.min()),
        "mean_tokens": round(float(lengths.mean()), 2),
        "median_tokens": int(np.median(lengths)),
        "max_tokens": int(lengths.max()),
        "percentiles": {
            f"p{percentile}": int(value)
            for percentile, value in zip(
                percentiles_requested,
                percentile_values,
            )
        },
        "candidate_coverage": {},
    }

    coverage_rows = []

    for max_length in CANDIDATE_LENGTHS:
        fits = int((lengths <= max_length).sum())
        truncated = int((lengths > max_length).sum())
        coverage_pct = round(fits / len(lengths) * 100, 2)
        truncation_pct = round(truncated / len(lengths) * 100, 2)

        stats["candidate_coverage"][str(max_length)] = {
            "fits_without_truncation": fits,
            "would_be_truncated": truncated,
            "coverage_pct": coverage_pct,
            "truncation_pct": truncation_pct,
        }

        coverage_rows.append(
            {
                "max_length": max_length,
                "fits_without_truncation": fits,
                "would_be_truncated": truncated,
                "coverage_pct": coverage_pct,
                "truncation_pct": truncation_pct,
            }
        )

    # Recomendación automática conservadora:
    # el menor candidato que conserva al menos 95 % de los correos.
    recommended = next(
        (
            row["max_length"]
            for row in coverage_rows
            if row["coverage_pct"] >= 95.0
        ),
        512,
    )
    stats["recommended_max_length_95pct"] = int(recommended)

    lengths_output = metrics_dir / "longitudes_tokens_beto.csv"
    coverage_output = metrics_dir / "cobertura_max_length.csv"
    summary_output = metrics_dir / "resumen_longitudes_tokens.json"
    plot_output = plots_dir / "distribucion_longitud_tokens.png"

    df[
        [
            "id",
            "label",
            "source",
            "token_length",
        ]
    ].to_csv(
        lengths_output,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(coverage_rows).to_csv(
        coverage_output,
        index=False,
        encoding="utf-8-sig",
    )

    with summary_output.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, edgecolor="black", alpha=0.8)
    plt.axvline(
        recommended,
        linestyle="--",
        linewidth=2,
        label=f"Recomendado: {recommended} tokens",
    )
    plt.axvline(
        256,
        linestyle=":",
        linewidth=2,
        label="Referencia: 256 tokens",
    )
    plt.title("Distribución de longitud de correos tokenizados con BETO")
    plt.xlabel("Cantidad de tokens")
    plt.ylabel("Número de correos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_output, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nEstadísticas:")
    print(f"  Mínimo:  {stats['min_tokens']}")
    print(f"  Promedio: {stats['mean_tokens']}")
    print(f"  Mediana: {stats['median_tokens']}")
    print(f"  Máximo:  {stats['max_tokens']}")

    print("\nPercentiles:")
    for name, value in stats["percentiles"].items():
        print(f"  {name}: {value}")

    print("\nCobertura por max_length:")
    coverage_df = pd.DataFrame(coverage_rows)
    print(coverage_df.to_string(index=False))

    print(
        "\nRecomendación provisional basada en cobertura >= 95 %: "
        f"{recommended} tokens"
    )

    print("\nArchivos generados:")
    for path in [
        lengths_output,
        coverage_output,
        summary_output,
        plot_output,
    ]:
        print(f"  {path}")

    print("\nAnálisis completado correctamente.")


if __name__ == "__main__":
    main()

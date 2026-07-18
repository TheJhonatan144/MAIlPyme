from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

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

LABEL_ALIASES = {
    "Variados": "Varios",
    "Varios/Extras": "Varios",
}

REQUIRED_COLUMNS = [
    "id",
    "subject",
    "sender",
    "date",
    "body",
    "label",
    "source",
]


def normalize_text(value: object) -> str:
    """Normaliza solo para detectar duplicados, no para entrenar."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_content_hash(row: pd.Series) -> str:
    normalized = "|||".join(
        [
            normalize_text(row["subject"]),
            normalize_text(row["sender"]),
            normalize_text(row["body"]),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_dataset(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing_columns}"
        )

    if df.empty:
        raise ValueError("El dataset está vacío.")

    empty_fields = {}
    for column in ["subject", "sender", "body", "label"]:
        empty_count = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )
        if empty_count:
            empty_fields[column] = int(empty_count)

    if empty_fields:
        raise ValueError(
            f"Existen campos obligatorios vacíos: {empty_fields}"
        )

    if df["id"].duplicated().any():
        duplicate_ids = (
            df.loc[df["id"].duplicated(keep=False), "id"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Existen IDs repetidos. Ejemplos: {duplicate_ids[:10]}"
        )

    invalid_labels = sorted(set(df["label"]) - set(OFFICIAL_LABELS))
    if invalid_labels:
        raise ValueError(
            f"Se encontraron etiquetas no oficiales: {invalid_labels}"
        )

    duplicated_content = df["content_hash"].duplicated(keep=False)
    if duplicated_content.any():
        examples = df.loc[
            duplicated_content,
            ["id", "label", "subject", "content_hash"],
        ].head(20)

        conflicting = (
            df.loc[duplicated_content]
            .groupby("content_hash")["label"]
            .nunique()
            .gt(1)
            .any()
        )

        if conflicting:
            raise ValueError(
                "Hay correos duplicados con etiquetas diferentes.\n"
                f"{examples.to_string(index=False)}"
            )

        raise ValueError(
            "Hay correos duplicados por contenido normalizado.\n"
            f"{examples.to_string(index=False)}"
        )


def build_distribution(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for label in OFFICIAL_LABELS:
        total = int((full_df["label"] == label).sum())
        train = int((train_df["label"] == label).sum())
        validation = int((validation_df["label"] == label).sum())
        test = int((test_df["label"] == label).sum())

        rows.append(
            {
                "label": label,
                "train": train,
                "validation": validation,
                "test": test,
                "total": total,
                "train_pct": round(train / total * 100, 2),
                "validation_pct": round(validation / total * 100, 2),
                "test_pct": round(test / total * 100, 2),
            }
        )

    rows.append(
        {
            "label": "TOTAL",
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
            "total": len(full_df),
            "train_pct": round(len(train_df) / len(full_df) * 100, 2),
            "validation_pct": round(
                len(validation_df) / len(full_df) * 100, 2
            ),
            "test_pct": round(len(test_df) / len(full_df) * 100, 2),
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    input_path = (
        project_root
        / "data"
        / "raw"
        / "dataset_mailpyme_integrado.csv"
    )
    processed_dir = project_root / "data" / "processed"
    metrics_dir = project_root / "outputs" / "metrics"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {input_path}"
        )

    print("=" * 70)
    print("PREPARACIÓN DEL DATASET MAILPYME AI")
    print("=" * 70)
    print(f"Archivo de entrada: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df.columns = [str(column).strip() for column in df.columns]

    for column in REQUIRED_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()

    df["label"] = df["label"].replace(LABEL_ALIASES)
    df["content_hash"] = df.apply(build_content_hash, axis=1)

    validate_dataset(df)

    # La entrada mantiene la redacción original, incluidos errores naturales.
    df["text"] = (
        "Asunto: "
        + df["subject"]
        + " Remitente: "
        + df["sender"]
        + " Cuerpo: "
        + df["body"]
    )

    df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)

    # Primer corte: 80 % entrenamiento y 20 % temporal.
    train_df, temporary_df = train_test_split(
        df,
        test_size=VALIDATION_RATIO + TEST_RATIO,
        random_state=SEED,
        stratify=df["label"],
        shuffle=True,
    )

    # Segundo corte: divide el 20 % temporal en 10 % validación y 10 % prueba.
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temporary_df["label"],
        shuffle=True,
    )

    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    validation_df = validation_df.sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df = df.reset_index(drop=True)

    # Verificación de fuga exacta entre divisiones.
    train_hashes = set(train_df["content_hash"])
    validation_hashes = set(validation_df["content_hash"])
    test_hashes = set(test_df["content_hash"])

    overlaps = {
        "train_validation": len(train_hashes & validation_hashes),
        "train_test": len(train_hashes & test_hashes),
        "validation_test": len(validation_hashes & test_hashes),
    }

    if any(overlaps.values()):
        raise RuntimeError(
            f"Se detectó fuga de contenido entre divisiones: {overlaps}"
        )

    output_columns = [
        "id",
        "subject",
        "sender",
        "date",
        "body",
        "text",
        "label",
        "label_id",
        "source",
        "content_hash",
    ]

    full_output = processed_dir / "dataset_preparado.csv"
    train_output = processed_dir / "train.csv"
    validation_output = processed_dir / "validation.csv"
    test_output = processed_dir / "test.csv"

    df[output_columns].to_csv(
        full_output,
        index=False,
        encoding="utf-8-sig",
    )
    train_df[output_columns].to_csv(
        train_output,
        index=False,
        encoding="utf-8-sig",
    )
    validation_df[output_columns].to_csv(
        validation_output,
        index=False,
        encoding="utf-8-sig",
    )
    test_df[output_columns].to_csv(
        test_output,
        index=False,
        encoding="utf-8-sig",
    )

    distribution = build_distribution(
        full_df=df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )
    distribution_path = metrics_dir / "distribucion_splits.csv"
    distribution.to_csv(
        distribution_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = pd.concat(
        [
            train_df.assign(split="train"),
            validation_df.assign(split="validation"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )[
        ["id", "split", "label", "label_id", "source", "content_hash"]
    ]

    manifest_path = metrics_dir / "manifiesto_splits.csv"
    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "dataset_name": "MailPyme AI",
        "input_file": str(input_path),
        "input_sha256": sha256_file(input_path),
        "seed": SEED,
        "official_labels": OFFICIAL_LABELS,
        "label_to_id": LABEL_TO_ID,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "sizes": {
            "full": len(df),
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
        "class_distribution_full": {
            label: int((df["label"] == label).sum())
            for label in OFFICIAL_LABELS
        },
        "content_overlap": overlaps,
        "normalized_duplicate_count": 0,
        "input_format": "Asunto: {subject} Remitente: {sender} Cuerpo: {body}",
        "privacy_note": (
            "Los archivos procesados permanecen localmente. "
            "No publicar correos reales ni datos sensibles sin anonimización."
        ),
    }

    summary_path = metrics_dir / "resumen_dataset.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nValidaciones:")
    print("  Columnas obligatorias: OK")
    print("  Campos obligatorios vacíos: 0")
    print("  IDs duplicados: 0")
    print("  Contenido duplicado normalizado: 0")
    print("  Etiquetas no oficiales: 0")
    print(f"  Fuga entre divisiones: {overlaps}")

    print("\nTamaños:")
    print(f"  Dataset completo: {len(df)}")
    print(f"  Train:            {len(train_df)}")
    print(f"  Validation:       {len(validation_df)}")
    print(f"  Test:             {len(test_df)}")

    print("\nDistribución:")
    print(distribution.to_string(index=False))

    print("\nArchivos generados:")
    for path in [
        full_output,
        train_output,
        validation_output,
        test_output,
        distribution_path,
        manifest_path,
        summary_path,
    ]:
        print(f"  {path}")

    print("\nPreparación completada correctamente.")


if __name__ == "__main__":
    main()

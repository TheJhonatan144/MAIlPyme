# MailPyme AI - BETO v2

Modelo de clasificación de correos empresariales en español para el MVP MailPyme AI.

## Modelo base

`dccuchile/bert-base-spanish-wwm-cased`

## Categorías

1. Contratos
2. Facturas
3. Colaboraciones
4. Clientes
5. Publicidad
6. Varios

## Entrada esperada

```python
texto = (
    f"Asunto: {subject} "
    f"Remitente: {sender} "
    f"Cuerpo: {body}"
)
```

Longitud máxima: 128 tokens.

## Carga del modelo

```python
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = Path("backend/models/mailpyme_beto_model")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
```

## Inferencia

```python
def clasificar_correo(subject: str, sender: str, body: str) -> dict:
    texto = (
        f"Asunto: {subject} "
        f"Remitente: {sender} "
        f"Cuerpo: {body}"
    )

    inputs = tokenizer(
        texto,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    confidence, predicted_id = probabilities.max(dim=-1)

    return {
        "category": model.config.id2label[int(predicted_id.item())],
        "confidence": float(confidence.item()),
        "status": "classified",
    }
```

## Resultados internos

- Accuracy en test interno: 1.0000
- F1 macro en test interno: 1.0000
- Test: 84 correos
- Latencia media individual en RTX 4060 Laptop: 7.10 ms
- Estabilidad: 100 % en 500 ejecuciones

Estos resultados corresponden a un conjunto de prueba interno pequeño y no garantizan el mismo rendimiento sobre correos externos.

## Privacidad

El repositorio no debe contener correos reales, datasets privados ni archivos de entrenamiento con información sensible.

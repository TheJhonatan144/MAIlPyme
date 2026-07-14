# Modelo BETO entrenado - MailPyme AI

Este directorio contiene el modelo BETO entrenado para clasificar correos empresariales de MiPYMEs ecuatorianas.

## Modelo base

`dccuchile/bert-base-spanish-wwm-cased`

## Categorías oficiales

1. Contratos
2. Facturas
3. Colaboraciones
4. Clientes
5. Publicidad
6. Varios

## Entrada esperada

El texto de entrada debe construirse concatenando asunto, remitente y cuerpo:

```python
texto = f"Asunto: {subject} Remitente: {sender} Cuerpo: {body}"

## Archivos incluidos

model.safetensors: pesos del modelo entrenado.
config.json: configuración del modelo y etiquetas.
tokenizer.json: tokenizer del modelo.
tokenizer_config.json: configuración del tokenizer.
special_tokens_map.json: tokens especiales.
mailpyme_config.json: configuración usada en el proyecto.
Uso desde backend
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "backend/models/mailpyme_beto_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
Salida esperada

El backend debe devolver:

{
  "categoria": "Contratos",
  "confianza": 0.94,
  "estado": "Clasificado"
}

Luego:

```bash
git add backend/models/mailpyme_beto_model/README.md
git commit -m "Documentar uso del modelo BETO entrenado"
git push
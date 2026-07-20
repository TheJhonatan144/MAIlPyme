import torch

from app.schemas import Category
from app.model_loader import load_model


labels = [
    "Contratos",
    "Facturas",
    "Colaboraciones",
    "Clientes",
    "Publicidad",
    "Varios",
]


def classify_email(
    sender: str,
    subject: str,
    body: str
) -> tuple[Category, str]:

    tokenizer, model = load_model()

    text = (
        f"Asunto: {subject} "
        f"Remitente: {sender} "
        f"Cuerpo: {body}"
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    category = labels[prediction.item()]

    confidence_value = round(
        confidence.item(),
        4
    )

    return category, str(confidence_value)
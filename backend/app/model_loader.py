from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mailpyme_beto_model"


tokenizer = None
model = None


def load_model():
    global tokenizer, model

    if model is None:
        print("Cargando modelo BETO...")

        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True
        )

        model.eval()

        print("Modelo BETO cargado correctamente")

    return tokenizer, model
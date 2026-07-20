from fastapi import APIRouter


router = APIRouter(
    prefix="/model",
    tags=["Model"]
)


@router.get("/info")
def model_info():

    return {
        "model_name": "MailPyme BETO v2",
        "base_model": "dccuchile/bert-base-spanish-wwm-cased",
        "architecture": "BERT Sequence Classification",
        "categories": 6,
        "max_length": 128,
        "labels": [
            "Contratos",
            "Facturas",
            "Colaboraciones",
            "Clientes",
            "Publicidad",
            "Varios"
        ]
    }
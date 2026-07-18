from fastapi import APIRouter

router = APIRouter(
    prefix="/categories",
    tags=["Categories"],
)

CATEGORIES = [
    "Contratos",
    "Facturas",
    "Colaboraciones",
    "Clientes",
    "Publicidad",
    "Varios",
]


@router.get("/")
def get_categories():
    return {
        "categories": CATEGORIES
    }
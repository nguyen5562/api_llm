from fastapi import APIRouter
from utils.embedding import embedding

router = APIRouter()


@router.get("/embedding")
def embedding_endpoint(pdf_path):
    """
    Check if the PDF file is embedded with text.
    """
    try:
        result = embedding(pdf_path)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

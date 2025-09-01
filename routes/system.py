from fastapi import APIRouter
import settings

router = APIRouter()


@router.get("/")
def health_check():
    return {"status": "ok"}


@router.get("/root")
def info():
    return {
        "root_api": str(settings.ROOT_API),
        "root_project": str(settings.ROOT_PROJECT),
    }

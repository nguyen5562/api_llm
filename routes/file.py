import os
from fastapi import APIRouter, File, UploadFile, HTTPException
import settings
from utils.file import check_file_exists
from pathlib import Path

router = APIRouter()


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Kiểm tra định dạng file
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_FILE:
        raise HTTPException(
            status_code=400, detail="Chỉ cho phép tải lên file PDF, DOC hoặc DOCX."
        )

    # Kiểm tra file đã tồn tại chưa
    if check_file_exists(file.filename):
        raise HTTPException(status_code=400, detail="File đã tồn tại.")

    file_path = os.path.join(settings.FILE_FOLDER, file.filename)

    # Ghi file vào server
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    return {
        "message": "Upload thành công.",
        "filename": file.filename,
        "path": file_path,
    }

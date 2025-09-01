import os
import settings


def check_file_exists(filename: str) -> bool:
    """
    Kiểm tra file đã tồn tại trong thư mục UPLOAD_FOLDER chưa.

    Args:
        filename (str): Tên file, ví dụ "abc.pdf"

    Returns:
        bool: True nếu đã tồn tại, False nếu chưa
    """
    file_path = os.path.join(settings.FILE_FOLDER, filename)
    return os.path.exists(file_path)

from pathlib import Path
import sys

file_path = Path(__file__).resolve()
root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

# Đường dẫn gốc của API
ROOT_API = root_path
ROOT_PROJECT = root_path.parent

# Thư mục chứa file tải lên
FILE_FOLDER = ROOT_PROJECT / "data" / "file"

# Thư mục chứa model và kết quả embedding
EMBEDDING_FOLDER = ROOT_PROJECT / "embedding" / "results"
EMBEDDING_MODEL_FOLDER = ROOT_PROJECT / "embedding" / "Vietnamese_Embedding"
FAISS_PATH = EMBEDDING_FOLDER / "all_faiss.index"
PICKLE_PATH = EMBEDDING_FOLDER / "all_embeddings.pkl"

# Thư mục chứa mô hình LLM
LLM_MODEL_FOLDER = ROOT_PROJECT / "models"
MODEL = LLM_MODEL_FOLDER / "Vinallama-7b-Chat"

# Chỉ cho phép các định dạng file nhất định
ALLOWED_FILE = {".pdf", ".doc", ".docx"}

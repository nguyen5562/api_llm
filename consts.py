import settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Set device to GPU 0 (Tesla T4)
device = torch.device("cuda:0")

# Load llm model
llm_model_path = str(settings.MODEL)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
llm_tokenizer.pad_token = llm_tokenizer.eos_token  # Set padding token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_path,
    device_map="cuda:0",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)
llm_model.eval()  # Chuyển model sang chế độ đánh giá (không training)

# Load embedding model
embedding_model_path = str(settings.EMBEDDING_MODEL_FOLDER)
embedding_model = SentenceTransformer(embedding_model_path)
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)

# Check
if (
    embedding_model is None
    or embedding_tokenizer is None
    or llm_model is None
    or llm_tokenizer is None
):
    raise RuntimeError("❌ Không thể tải mô hình. Vui lòng kiểm tra cấu hình.")

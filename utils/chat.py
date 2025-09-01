import faiss
import pickle
import numpy as np
from transformers import TextIteratorStreamer
import threading
from consts import settings, llm_model, llm_tokenizer, embedding_model


def load_emedding_and_chunks():
    # Load FAISS index
    faiss_index = faiss.read_index(str(settings.FAISS_PATH))

    # Load chunks
    with open(settings.PICKLE_PATH, "rb") as f:
        data = pickle.load(f)
    all_chunks = []
    for item in data:
        all_chunks.extend(item["chunks"])

    return faiss_index, all_chunks


def get_relevant_chunks(
    query,
    faiss_index,
    chunks,
    top_k=3,
    max_tokens_per_chunk=512,
):
    query_vector = embedding_model.encode(
        [query]
    )  # dùng model embedding đã dùng trước đó
    D, I = faiss_index.search(np.array(query_vector).astype("float32"), top_k)

    context_chunks = []
    for i in I[0]:
        chunk = chunks[i]
        tokens = llm_tokenizer.tokenize(chunk)
        if len(tokens) > max_tokens_per_chunk:
            tokens = tokens[:max_tokens_per_chunk]
            chunk = llm_tokenizer.convert_tokens_to_string(tokens)
        context_chunks.append(chunk.strip())

    return context_chunks


def build_prompt(context_chunks, question):
    context = "\n---\n".join(context_chunks)
    return f"""<|im_start|>system
Bạn là một trợ lý AI tuyển sinh của Học viện Kỹ thuật Quân sự. Chỉ trả lời người dùng dựa trên thông tin được cung cấp dưới đây. Nếu không biết, hãy trả lời: "Tôi không có thông tin về câu hỏi này." Không được bịa.
<|im_end|>
<|im_start|>user
Thông tin:
{context}

Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
"""


def configure_generation():
    generation_config = llm_model.generation_config
    generation_config.max_new_tokens = 256  # đủ dài để trả lời rõ
    generation_config.num_beams = 3  # tăng tính chính xác (tìm tốt hơn)
    generation_config.early_stopping = True
    generation_config.do_sample = True  # dùng beam search (ổn định, không ngẫu nhiên)
    generation_config.num_return_sequences = 1

    # Các tham số sau sẽ bị **bỏ qua** vì `do_sample = False`
    # → bạn nên xóa hoặc để chú thích
    generation_config.temperature = None
    generation_config.top_p = 1.0

    # Thêm 2 tham số giúp giảm lặp, sát nghĩa hơn:
    # generation_config.repetition_penalty = 1.2
    # generation_config.no_repeat_ngram_size = 3

    return generation_config


def generate_answer(prompt):
    # Tokenize và đưa lên thiết bị
    encoding = llm_tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(llm_model.device)
    attention_mask = encoding["attention_mask"].to(llm_model.device)

    # Sinh văn bản
    outputs = llm_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=configure_generation(),
    )

    # Lấy phần model vừa sinh (bỏ phần input gốc)
    generated_only_ids = outputs[0][input_ids.shape[1] :]
    return llm_tokenizer.decode(generated_only_ids, skip_special_tokens=True)


def get_response(query: str):
    faiss_index, chunks = load_emedding_and_chunks()

    try:
        # Lấy các đoạn văn bản liên quan
        context_chunks = get_relevant_chunks(query, faiss_index, chunks)

        # Tạo prompt
        prompt = build_prompt(context_chunks, query)

        # Sinh câu trả lời
        full_output = generate_answer(prompt)

        # # Trích xuất phần trả lời
        # if "<|im_start|>assistant" in full_output:
        #     answer = full_output.split("<|im_start|>assistant")[-1].strip()
        # else:
        #     answer = full_output.strip()

        return full_output.strip()

    except Exception as e:
        print(f"❌ Lỗi trong quá trình xử lý: {e}")
        return "❌ Đã xảy ra lỗi trong quá trình xử lý câu hỏi. Vui lòng thử lại sau."

def generate_answer_stream(prompt):
    # Tokenize và đưa lên GPU
    encoding = llm_tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(llm_model.device)
    attention_mask = encoding["attention_mask"].to(llm_model.device)

    # Tạo streamer để lấy token sinh ra
    streamer = TextIteratorStreamer(
        llm_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Chạy sinh văn bản trong thread phụ
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "generation_config": configure_generation(),
        "streamer": streamer,
    }
    thread = threading.Thread(target=llm_model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield từng token
    for token in streamer:
        yield token


def get_response_stream(query: str):
    # Load dữ liệu
    faiss_index, chunks = load_emedding_and_chunks()

    try:
        # Lấy các đoạn ngữ cảnh liên quan
        context_chunks = get_relevant_chunks(query, faiss_index, chunks)

        # Tạo prompt hội thoại đầy đủ
        prompt = build_prompt(context_chunks, query)

        # Trả về generator từ mô hình
        return generate_answer_stream(prompt)

    except Exception as e:
        print(f"❌ Lỗi trong get_response_stream: {e}")
        yield "❌ Lỗi xử lý câu hỏi."
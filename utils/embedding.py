import os
import pickle
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
from datetime import datetime
import numpy as np
import re
from underthesea import sent_tokenize
import faiss
from consts import settings, embedding_model, embedding_tokenizer


def is_pdf_embedded(path):
    if not os.path.exists(settings.PICKLE_PATH):
        return False  # Chưa có dữ liệu chung => chắc chắn chưa nhúng gì

    pdf_name = os.path.splitext(os.path.basename(path))[0]

    with open(settings.PICKLE_PATH, "rb") as f:
        all_data = pickle.load(f)

    existing_pdf_names = {entry["pdf_name"] for entry in all_data}

    return pdf_name in existing_pdf_names


def preprocess_image(img):
    """
    Tiền xử lý ảnh để cải thiện OCR
    """
    if img.mode != "L":
        img = img.convert("L")

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    img = img.filter(ImageFilter.SHARPEN)

    return img


def ocr_pdf_to_text(pdf_path, output_dir):
    """
    OCR file PDF thành text
    """
    try:
        print(f"📖 Đang OCR file: {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        full_text = ""
        ocr_config = r"--oem 3 --psm 6 -l vie"

        for page_num in range(total_pages):
            print(f"🔄 Xử lý trang {page_num + 1}/{total_pages}...")

            page = doc.load_page(page_num)
            matrix = fitz.Matrix(2.5, 2.5)
            pix = page.get_pixmap(matrix=matrix)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            img = preprocess_image(img)

            try:
                page_text = pytesseract.image_to_string(img, config=ocr_config)
                full_text += page_text.strip()

            except Exception as e:
                print(f"   ❌ Lỗi OCR trang {page_num + 1}: {e}")

        doc.close()
        print(f"✅ Hoàn thành OCR {total_pages} trang")

        # 🧠 Tạo tên file JSON theo tên file PDF
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(output_dir, pdf_name, f"{pdf_name}_ocr.txt")

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Lưu file JSON
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"📄 Kết quả đã lưu vào: {output_path}")

        # Trả về danh sách các trang với nội dung
        return full_text

    except Exception as e:
        print(f"❌ Lỗi OCR: {e}")
        return None


def clean_text(text, pdf_path, output_dir):
    """
    Làm sạch một đoạn văn bản OCR (string)
    """
    # Loại ký tự không mong muốn (giữ lại tiếng Việt, toán học, đơn vị)
    text = re.sub(r"[^\w\s.,;:()\[\]?!\"\'\-–—…°%‰≥≤→←≠=+/*<>\n\r]", "", text)

    # Xử lý lỗi xuống dòng giữa từ hoặc giữa câu
    text = re.sub(r"-\n", "", text)  # nối từ bị gạch nối xuống dòng
    text = re.sub(r"\n(?=\w)", " ", text)  # dòng xuống không hợp lý → nối câu

    # Dấu chấm lặp vô nghĩa → ba chấm
    text = re.sub(r"\.{3,}", "...", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\n\s*\n", "\n\n", text)  # giữ ngắt đoạn
    text = re.sub(r"[ \t]+", " ", text)  # nhiều khoảng trắng → 1 dấu cách
    text = re.sub(r" *\n *", "\n", text)  # bỏ khoảng trắng đầu/cuối dòng

    # Lưu file
    clean_text = text.strip()

    # 🧠 Tạo tên file JSON theo tên file PDF
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, pdf_name, f"{pdf_name}_clean.txt")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Lưu file JSON
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"📄 Kết quả đã lưu vào: {output_path}")

    return clean_text


def split_sections(text):
    """
    Tách text thành các phần theo tiêu đề kiểu I., 1., a)
    """
    sections = re.split(r"\n(?=(?:[IVXLCDM]+\.)|(?:\d+\.)|(?:[a-z]\)))", text)
    return [s.strip() for s in sections if s.strip()]


def split_text_to_chunks_vi_tokenized_with_section(text, chunk_size=512, overlap=50):
    """
    Chia văn bản tiếng Việt thành các chunk dựa trên số token,
    giữ nguyên cấu trúc section và câu.
    """
    sections = split_sections(text)
    all_chunks = []

    for section in sections:
        sentences = sent_tokenize(section)
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            num_tokens = len(embedding_tokenizer.tokenize(sentence))

            if current_tokens + num_tokens > chunk_size:
                chunk_text = "\n".join(current_chunk).strip()
                all_chunks.append(chunk_text)

                # Overlap bằng token
                overlap_chunk = []
                total = 0
                for s in reversed(current_chunk):
                    toks = len(embedding_tokenizer.tokenize(s))
                    if total + toks > overlap:
                        break
                    overlap_chunk.insert(0, s)
                    total += toks

                current_chunk = overlap_chunk + [sentence]
                current_tokens = total + num_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += num_tokens

        if current_chunk:
            all_chunks.append(" ".join(current_chunk).strip())

    return all_chunks


def create_embeddings(chunks):
    """
    Tạo embeddings cho các text chunks
    """
    try:
        print(f"🔄 Tạo embeddings cho {len(chunks)} chunks...")
        embeddings = embedding_model.encode(chunks, show_progress_bar=True)

        print(f"✅ Hoàn thành tạo embeddings")
        return embeddings

    except Exception as e:
        print(f"❌ Lỗi tạo embeddings: {e}")
        return None


def save_embeddings(chunks, embeddings, pdf_path, output_dir):
    """
    Lưu embeddings và chunks vào file
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.join(output_dir, pdf_name), exist_ok=True)

    # Lưu dữ liệu
    data = {
        "pdf_name": pdf_name,
        "chunks": chunks,
        "embeddings": embeddings,
        "created_at": datetime.now().isoformat(),
    }

    # Lưu embeddings (pickle)
    pickle_path = os.path.join(output_dir, pdf_name, f"{pdf_name}_embeddings.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)

    # Lưu chunks (text file)
    chunks_path = os.path.join(output_dir, pdf_name, f"{pdf_name}_chunks.txt")
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(f"CHUNKS TỪ FILE: {pdf_name}.pdf\n")
        f.write(f"Tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tổng số chunks: {len(chunks)}\n")
        f.write("=" * 60 + "\n\n")

        for i, chunk in enumerate(chunks, 1):
            f.write(f"CHUNK {i}:\n")
            f.write("-" * 30 + "\n")
            f.write(chunk + "\n")
            f.write("-" * 30 + "\n\n")

    # Lưu thông tin embeddings (text file)
    embedding_info_path = os.path.join(
        output_dir, pdf_name, f"{pdf_name}_embedding_info.txt"
    )
    with open(embedding_info_path, "w", encoding="utf-8") as f:
        f.write(f"THÔNG TIN EMBEDDINGS: {pdf_name}.pdf\n")
        f.write(f"Tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"📊 THỐNG KÊ:\n")
        f.write(f"- Tổng số chunks: {len(chunks)}\n")
        f.write(f"- Kích thước embeddings: {embeddings.shape}\n")
        f.write(f"- Kiểu dữ liệu: {embeddings.dtype}\n")
        f.write(f"- Kích thước mỗi vector: {embeddings.shape[1]} dimensions\n\n")

        f.write(f"📝 PREVIEW EMBEDDINGS (5 chunks đầu):\n")
        f.write("-" * 50 + "\n")

        for i in range(min(5, len(chunks))):
            f.write(f"\nCHUNK {i+1}:\n")
            f.write(f"Text: {chunks[i][:100]}...\n")
            f.write(
                f"Embedding vector (10 giá trị đầu): {embeddings[i][:10].tolist()}\n"
            )
            f.write(f"Vector norm: {np.linalg.norm(embeddings[i]):.4f}\n")
            f.write("-" * 30 + "\n")

    # 4️⃣ Lưu FAISS index
    index_path = os.path.join(output_dir, pdf_name, f"{pdf_name}_faiss.index")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)

    # --- 🔁 Cập nhật FAISS chung ---
    all_faiss_path = os.path.join(output_dir, "all_faiss.index")
    if os.path.exists(all_faiss_path):
        index_all = faiss.read_index(all_faiss_path)
    else:
        index_all = faiss.IndexFlatL2(dim)

    index_all.add(embeddings.astype(np.float32))
    faiss.write_index(index_all, all_faiss_path)

    # --- 🔁 Cập nhật pickle chung ---
    all_pickle_path = os.path.join(output_dir, "all_embeddings.pkl")
    if os.path.exists(all_pickle_path):
        with open(all_pickle_path, "rb") as f:
            all_data = pickle.load(f)
    else:
        all_data = []

    all_data.append(data)

    with open(all_pickle_path, "wb") as f:
        pickle.dump(all_data, f)

    print(f"💾 Đã lưu embeddings: {pickle_path}")
    print(f"📄 Đã lưu chunks: {chunks_path}")
    print(f"📊 Đã lưu thông tin embeddings: {embedding_info_path}")
    print(f"📌 Đã lưu FAISS index: {index_path}")
    print(f"🔁 Cập nhật FAISS chung: {all_faiss_path}")
    print(f"📦 Cập nhật pickle chung: {all_pickle_path}")

    return pickle_path, index_path


def embedding(pdf_path):
    """
    Hàm chính để thực hiện embedding cho file PDF
    """
    # 0️⃣ Kiểm tra xem PDF đã được nhúng chưa
    if is_pdf_embedded(pdf_path):
        return "✅ PDF đã được embedding trước đó."

    # 1️⃣ Bắt đầu embedding
    try:
        # 2️⃣ OCR file PDF
        ocr_text = ocr_pdf_to_text(pdf_path, settings.EMBEDDING_FOLDER)
        if ocr_text is None:
            return "❌ Lỗi trong quá trình OCR file PDF."

        # 3️⃣ Làm sạch văn bản
        clean_text_content = clean_text(ocr_text, pdf_path, settings.EMBEDDING_FOLDER)

        # 4️⃣ Chia văn bản thành các chunk
        chunks = split_text_to_chunks_vi_tokenized_with_section(clean_text_content)

        # 5️⃣ Tạo embeddings
        embeddings = create_embeddings(chunks)
        if embeddings is None:
            return "❌ Lỗi trong quá trình tạo embeddings."

        # 6️⃣ Lưu embeddings và chunks
        save_embeddings(chunks, embeddings, pdf_path, settings.EMBEDDING_FOLDER)

        return "✅ Embedding hoàn tất thành công!"
    except Exception as e:
        print(f"❌ Lỗi trong quá trình embedding: {e}")
        return f"❌ Lỗi trong quá trình embedding: {e}"

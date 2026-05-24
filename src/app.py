import streamlit as st
import json
import faiss
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# --- TẦNG 1: KHỞI TẠO HỆ THỐNG TRUY XUẤT CACHE (BACKEND CỦA BẠN A) ---
@st.cache_resource
def init_rag_backend():
    """Hàm nạp dữ liệu chunks và lưu cấu trúc Vector DB vào bộ nhớ RAM"""
    chunks_file = 'chunks/uet_rag_chunks_dataset.json'
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        
    all_texts = [chunk["text"] for chunk in chunks_data]
    
    # Khởi tạo bộ nhúng vector phẳng nền (Dense Retriever)
    bi_encoder = SentenceTransformer('keepitreal/vietnamese-sbert')
    doc_embeddings = bi_encoder.encode(all_texts, convert_to_numpy=True)
    
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    
    # Khởi tạo bộ tối ưu hóa thứ hạng nâng cao (Cross-Encoder Reranker)
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    
    return index, bi_encoder, reranker, chunks_data, all_texts

# Kích hoạt hệ thống RAG Core
index, bi_encoder, reranker, chunks_data, all_texts = init_rag_backend()

# --- TẦNG 2: PIPELINE GENERATOR & PROMPT ENGINEERING (BẠN B) ---
def generate_answer(user_question, use_reranker=True, top_k=3):
    """Luồng RAG hoàn chỉnh xử lý truy xuất và gọi LLM sinh câu trả lời"""
    
    # 1. Tầng Retriever: Tìm kiếm 15 ứng viên thô bằng FAISS
    query_embed = bi_encoder.encode([user_question], convert_to_numpy=True)
    distances, indices = index.search(query_embed, k=15)
    candidate_chunks = [chunks_data[idx] for idx in indices[0]]
    
    # 2. Tầng Reranker: Kiểm tra cấu hình hệ thống
    if use_reranker:
        rerank_pairs = [[user_question, chunk["text"]] for chunk in candidate_chunks]
        rerank_scores = reranker.predict(rerank_pairs)
        sorted_indices = np.argsort(rerank_scores)[::-1]
        final_chunks = [candidate_chunks[idx] for idx in sorted_indices][:top_k]
    else:
        final_chunks = candidate_chunks[:top_k]
        
    # Hợp nhất danh sách văn bản bốc được thành chuỗi Ngữ cảnh (Context)
    context_text = "\n\n".join([chunk["text"] for chunk in final_chunks])
    
    # 3. Tầng Generator: Thiết lập Prompt Engineering theo chuẩn tối ưu SQuAD
    system_prompt = f"""Bạn là một trợ lý học vụ AI chuyên trách trích xuất thông tin của Trường Đại học Công nghệ (UET).
Nhiệm vụ của bạn là dựa VÀO VĂN BẢN HƯỚNG DẪN được cung cấp để trả lời câu hỏi của người dùng.

YÊU CẦU NGHIÊM NGẶT:
1. Chỉ trả lời dựa trên thông tin có trong VĂN BẢN HƯỚNG DẪN. 
2. Câu trả lời phải cực kỳ NGẮN GỌN, SÚC TÍCH, chỉ nêu từ khóa chính hoặc con số cụ thể (Tuyệt đối không lặp lại câu hỏi, không viết cả câu dài dòng).
3. Nếu thông tin không có trong văn bản, bắt buộc trả lời "Tôi không biết". Không tự ý bịa đặt thông tin.

VÍ DỤ MẪU:
Context: "Trường Đại học Công nghệ (UET) được thành lập theo Quyết định số 92/2004/QĐ-TTg ngày 25/05/2004 của Thủ tướng Chính phủ."
Question: "Trường UET được thành lập vào năm nào?"
Answer: 2004

---
VĂN BẢN HƯỚNG DẪN:
{context_text}

CÂU HỎI: {user_question}
Answer:"""

    try:
        # Gọi cục bộ mô hình Qwen đang chạy ngầm trên Ollama máy Local
        response = ollama.generate(
            model='qwen2.5:3b', 
            prompt=system_prompt,
            options={"temperature": 0.0} # Ép Độc tính sáng tạo về 0 để mô hình trích xuất chuẩn xác, không ảo tưởng
        )
        return response['response'].strip(), final_chunks
    except Exception as e:
        return f"[Lỗi kết nối Ollama Local]: {e}. Vui lòng kiểm tra lệnh 'ollama run qwen2.5:3b' trên Terminal!", []

# --- TẦNG 3: UI INTERFACE DEVELOPMENT (BẠN D) ---
st.set_page_config(page_title="UET Học Vụ Trợ Lý", page_icon="🤖", layout="wide")

st.title("🤖 Hệ thống Hỏi-Đáp Học vụ Thông minh UET (RAG QA)")
st.caption("Sản phẩm Bài tập lớn Nhóm - Hệ thống RAG thực nghiệm trên Quy chế và Tuyển sinh UET 2026")

# Thiết kế Sidebar điều khiển tham số
st.sidebar.header("🛠️ CẤU HÌNH HỆ THỐNG")
rag_mode = st.sidebar.selectbox(
    "Lựa chọn Biến thể RAG:",
    ("RAG Cơ bản (Baseline - Không Rerank)", "RAG Nâng cao (Advanced - Có Reranker)")
)
k_value = st.sidebar.slider("Số lượng ngữ cảnh truyền vào LLM (Top-K):", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.info("""
**Thành viên thực hiện:**
* Bạn A: Vector DB & Retriever
* Bạn B: Prompt & Generator
* Bạn C: Đánh giá & Báo cáo
* Bạn D: Giao diện Web App
""")

# Khung giao diện Chatbot chính
user_query = st.text_input("✍️ Nhập câu hỏi của ông vào đây về Quy chế hoặc Tuyển sinh UET:", placeholder="Ví dụ: Chỉ tiêu ngành CNTT năm 2026 là bao nhiêu?")

if st.button("🚀 Gửi câu hỏi"):
    if user_query.strip() == "":
        st.warning("Ông chưa gõ câu hỏi kìa!")
    else:
        is_reranker_active = True if "RAG Nâng cao" in rag_mode else False
        
        with st.spinner("🤖 Trợ lý đang đọc tài liệu quy chế để trích xuất đáp án..."):
            # Chạy toàn bộ luồng RAG
            answer, retrieved_sources = generate_answer(user_query, use_reranker=is_reranker_active, top_k=k_value)
            
        # Hiển thị câu trả lời ngắn gọn
        st.success("🎯 Câu trả lời trích xuất ngắn gọn:")
        st.subheader(f"{answer}")
        
        # Hiển thị tài liệu tham chiếu (Minh chứng hoạt động của bộ tìm kiếm của Bạn A)
        st.markdown("---")
        st.subheader("📋 Các đoạn văn bản nguồn được bộ truy xuất bốc lên:")
        
        for idx, chunk in enumerate(retrieved_sources):
            with st.expander(f"Đoạn nguồn #{idx+1}: {chunk['title']} (Mã: {chunk['chunk_id']})"):
                st.write(chunk['text'])
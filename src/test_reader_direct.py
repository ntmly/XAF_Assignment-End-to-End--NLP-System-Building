import json
from transformers import pipeline

# === CẤU HÌNH ===
chunks_path = "data/processed/chunks.json"
model_name = "deepset/bert-base-multilingual-cased-squad2"  # Đổi thành "distilbert-base-uncased-distilled-squad" nếu muốn test model cũ
question = "Học phần bắt buộc là gì?"

# === ĐỌC CHUNKS ===
with open(chunks_path, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Tìm chunk có chunk_id cụ thể (bạn có thể sửa ID)
target_chunk_id = "DOC_None_CHUNK_SO_TAY_HOC_VU_REG_000"
chunk_text = None
for chunk in chunks:
    if chunk.get("chunk_id") == target_chunk_id:
        chunk_text = chunk["text"]
        break

# Nếu không tìm thấy, lấy chunk đầu tiên (thường là chunk bạn vừa thêm)
if not chunk_text and len(chunks) > 0:
    chunk_text = chunks[0]["text"]
    print("⚠️ Không tìm thấy ID, dùng chunk đầu tiên:")

print("\n=== CONTEXT ===")
print(chunk_text[:500] + "...\n")

# === LOAD READER ===
print(f"Loading model: {model_name} ...")
qa = pipeline("question-answering", model="nguyenvulebinh/vi-mrc-large")

# === DỰ ĐOÁN ===
result = qa(question=question, context=chunk_text)
print(f"📝 QUESTION: {question}")
print(f"✅ ANSWER: {result['answer']}")
print(f"📊 SCORE: {result['score']:.4f}")
print(f"📍 POSITION: chars {result['start']} -> {result['end']}")
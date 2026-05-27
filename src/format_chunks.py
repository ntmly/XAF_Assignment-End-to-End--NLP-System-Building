import json

CHUNKS_PATH = "data/processed/chunks.json"

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

formatted_chunks = []

for idx, chunk in enumerate(chunks):

    formatted_chunk = {
        "chunk_id": idx,
        "title": chunk.get("title", ""),
        "text": chunk.get("text", "")
    }

    formatted_chunks.append(formatted_chunk)

with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        formatted_chunks,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Updated {len(formatted_chunks)} chunks in:")
print(CHUNKS_PATH)
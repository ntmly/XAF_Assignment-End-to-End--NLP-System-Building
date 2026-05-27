# query_rewriter.py

class QueryRewriter:

    def __init__(self, llm):

        self.llm = llm

    def rewrite(self, query: str):

        prompt = f"""
Bạn là hệ thống hỗ trợ retrieval cho RAG.

Nhiệm vụ:
Phân tích câu hỏi và sinh ra nhiều truy vấn tìm kiếm khác nhau để tối đa khả năng tìm đúng tài liệu.

Hãy:
- diễn đạt lại câu hỏi
- mở rộng semantic
- mở rộng lexical
- thêm synonym nếu cần
- thử nhiều cách hỏi khác nhau
- thử nhiều giả thuyết về ý nghĩa câu hỏi

Yêu cầu:
- Chỉ trả về query
- Mỗi dòng đúng 1 query
- Không giải thích
- Không đánh số

Câu hỏi:
{query}
"""

        try:

            output = self.llm(
                prompt,
                max_tokens=128,
                temperature=0.4,
                echo=False
            )

            text = output["choices"][0]["text"].strip()

            queries = [
                line.strip("-•123456789. ")
                for line in text.split("\n")
                if line.strip()
            ]

        except Exception:

            queries = []

        queries.insert(0, query)

        unique_queries = []

        seen = set()

        for q in queries:

            q_clean = q.strip()

            if len(q_clean) < 3:
                continue

            if len(q_clean.split()) > 15:
                continue

            q_lower = q_clean.lower()

            if q_lower not in seen:

                seen.add(q_lower)

                unique_queries.append(q_clean)

        return unique_queries[:10]
from pathlib import Path
from llama_cpp import Llama


class GenerativeReader:

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128
    ):

        BASE_DIR = Path(__file__).resolve().parent.parent

        MODEL_PATH = BASE_DIR / model_path

        print(f"[GenerativeReader] Đang load model:")
        print(MODEL_PATH)
        print(f"Model exists: {MODEL_PATH.exists()}")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model tại: {MODEL_PATH}"
            )

        self.llm = Llama(
            model_path=str(MODEL_PATH),

            n_ctx=1024,

            n_threads=4,
            n_threads_batch=4,

            n_batch=64,

            verbose=False
        )

        self.max_new_tokens = int(max_new_tokens)

        print("[GenerativeReader] Load model thành công.")

    def answer(self, query: str, retrieved_chunks: list) -> str:

        if not retrieved_chunks:
            return "UNKNOWN"

        context = "\n\n---\n\n".join(
            retrieved_chunks[:3]
        )

        prompt = (
            "<|im_start|>system\n"
            "Bạn là trợ lý AI trả lời câu hỏi dựa trên tài liệu.\n"
            "Quy tắc bắt buộc:\n"
            "1. Chỉ dùng thông tin trong phần TÀI LIỆU.\n"
            "2. Trả lời bằng 1-2 câu ngắn gọn, tiếng Việt.\n"
            "3. KHÔNG giải thích, KHÔNG lặp lại câu hỏi, KHÔNG thêm thông tin ngoài.\n"
            "4. Nếu tài liệu không có thông tin, trả lời đúng 1 từ: UNKNOWN<|im_end|>\n"
            "<|im_start|>user\n"
            f"TÀI LIỆU:\n{context}\n\n"
            f"CÂU HỎI: {query}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Câu trả lời:"
        )

        output = self.llm(
            prompt,

            max_tokens=self.max_new_tokens,

            stop=[
                "<|im_end|>",
                "<|im_start|>",
                "\n\n",
                "Câu hỏi:",
                "TÀI LIỆU:"
            ],

            echo=False,

            temperature=0.1,

            repeat_penalty=1.15
        )

        answer = output["choices"][0]["text"].strip()

        if answer:
            first_line = answer.split("\n")[0].strip()
            return first_line if first_line else answer

        return "UNKNOWN"
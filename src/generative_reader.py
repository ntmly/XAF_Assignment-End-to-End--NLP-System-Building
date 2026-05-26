from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GenerativeReader:
    def __init__(self, model_name: str, max_new_tokens: int = 256):
        print(f"Loading generative model {model_name} on CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        self.max_new_tokens = max_new_tokens
        
        # Đặt padding token nếu chưa có
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def answer(self, query: str, retrieved_chunks: list) -> str:
        if not retrieved_chunks:
            return "UNKNOWN"

        # 1. Tạo context từ các chunks đã retrieve
        context = "\n\n---\n\n".join(retrieved_chunks) # Phân cách rõ ràng giữa các đoạn
        
        # 2. Tạo prompt theo định dạng chat
        messages = [
            {"role": "system", "content": "Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ dựa trên thông tin được cung cấp."},
            {"role": "user", "content": f"Dựa vào các thông tin sau:\n{context}\n---\nHãy trả lời câu hỏi: {query}"}
        ]
        
        # Áp dụng chat template cho model
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 3. Tokenize và sinh câu trả lời
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cpu")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False, # Tắt sampling để có kết quả nhất quán
                temperature=0.1,
                repetition_penalty=1.1
            )
        
        # 4. Giải mã và trả về kết quả
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Chỉ lấy phần trả lời của assistant (phần sau prompt)
        answer = full_response[len(prompt):].strip()
        return answer if answer else "UNKNOWN"
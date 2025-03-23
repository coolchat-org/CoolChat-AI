
from typing import Tuple
import openai
from app.core.config import settings


def tokenize_and_summarize_openai(old_context, query, response, max_token: int = 2048) -> Tuple[str, bool]:
    new_context = f"{old_context + " " if old_context != "" else ""}Customer: {query}, AI: {response}"
    token_count = len(new_context.split())
    
    openai.api_key = settings.OPENAI_API_KEY
    if token_count >= max_token:
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Hoặc "gpt-4" nếu cần chất lượng cao hơn

                messages=[
                    {"role": "system", "content": "Bạn là một trợ lý AI chuyên tóm tắt văn bản và cuộc hội thoại bằng tiếng Việt."},
                    {"role": "user", "content": f"Tóm tắt văn bản sau, là một cuộc hội thoại giữa một nhân viên chăm sóc khách hàng và khách hàng: {new_context}"}
                ],
                max_tokens=max_token // 2,
                temperature=0.7,  # Điều chỉnh độ sáng tạo (0.7 là giá trị cân bằng)
                n=1,  # Số lượng kết quả trả về
                stop=None,  # Không cần chuỗi dừng cụ thể
            )
            # Trích xuất kết quả tóm tắt
            summary = response.choices[0].message.content.strip()
            return summary, True
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "", False

    return "", False
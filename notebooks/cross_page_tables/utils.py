import os
from google import genai
from google.genai import types
import json



def get_is_new_section_context(contexts):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=str(contexts)),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["is_new_section_context"],
            properties = {
                "is_new_section_context": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    items = genai.types.Schema(
                        type = genai.types.Type.BOOLEAN,
                    ),
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""Bạn là một chuyên gia phân tích cấu trúc tài liệu. Nhiệm vụ của bạn là xem xét một đoạn văn bản (CONTEXT_BEFORE) đứng ngay trước một bảng hoặc một mục, và xác định xem đoạn văn bản đó có chứa dấu hiệu rõ ràng cho thấy một phần (section), một mục hoặc một bảng mới sắp bắt đầu hay không.

Bạn sẽ được cung cấp một danh sách (List) các chuỗi CONTEXT_BEFORE. Bạn cần trả về một danh sách (List) các giá trị boolean (True hoặc False) có cùng độ dài, trong đó mỗi giá trị boolean tương ứng với quyết định của bạn cho chuỗi CONTEXT_BEFORE ở vị trí tương ứng.

Tiêu chí để quyết định True (Gợi ý mục/bảng mới):
- Tiêu đề rõ ràng
- Đề mục có cấu trúc
- Ngữ cảnh giới thiệu

Tiêu chí để quyết định False (KHÔNG gợi ý mục/bảng mới):
- Chuỗi rỗng
- Nội dung liền mạch
- Không có cấu trúc tiêu đề
- Chỉ là dữ liệu hoặc mô tả phụ

Yêu cầu:

Hãy phân tích cẩn thận từng CONTEXT_BEFORE trong danh sách đầu vào.
Áp dụng các tiêu chí trên để đưa ra quyết định True hoặc False cho mỗi context.
Luôn trả về False cho các giá trị CONTEXT_BEFORE rỗng.
Trả về kết quả dưới dạng một danh sách các giá trị boolean."""),
        ],
    )

    res =  client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return json.loads(res.text)



def get_is_has_header(rows):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=str(rows)),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            properties = {
                "is_has_header": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    items = genai.types.Schema(
                        type = genai.types.Type.BOOLEAN,
                    ),
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""Bạn là một chuyên gia phân tích cấu trúc dữ liệu dạng bảng. Nhiệm vụ của bạn là xem xét từng \"dòng\" (row) được cung cấp và xác định xem dòng đó có phải là một dòng tiêu đề (header) có ý nghĩa hay không. Một dòng tiêu đề có ý nghĩa là dòng chứa các tên cột mô tả loại dữ liệu sẽ xuất hiện trong các cột đó ở các dòng tiếp theo, chứ không phải là một dòng chứa dữ liệu cụ thể.

Bạn sẽ nhận được một danh sách (List) các dòng. Mỗi dòng trong danh sách này lại là một danh sách (List) các chuỗi ký tự (string), trong đó mỗi chuỗi đại diện cho nội dung của một ô (cell) trong dòng đó.

Bạn cần trả về một danh sách (List) các giá trị boolean (True hoặc False) có cùng độ dài với danh sách dòng đầu vào. Giá trị True có nghĩa là dòng tương ứng LÀ một header có ý nghĩa, và False có nghĩa là KHÔNG PHẢI.

Tiêu chí để xác định một dòng là Header có ý nghĩa (True):
- Tiêu chí để xác định một dòng là Header có ý nghĩa (True)
- Cấu trúc mô tả, không phải dữ liệu cụ thể
- Sử dụng tên cột chung chung
- Thường ngắn gọn và mang tính danh mục
- Không bắt đầu bằng dữ liệu số thứ tự hoặc ID cụ thể

Tiêu chí để xác định một dòng KHÔNG PHẢI là Header có ý nghĩa (False):
- Chứa dữ liệu cụ thể
- Bắt đầu bằng số thứ tự hoặc giá trị dữ liệu
- Là một câu hoàn chỉnh hoặc đoạn văn mô tả dài

Lưu ý quan trọng:

Sự hiện diện của tag HTML như <br> không làm thay đổi bản chất của ô là header hay data.
Đôi khi header có thể bị ngắt quãng hoặc có lỗi định dạng nhỏ (ví dụ: \"Title D\", \"irected by\"), hãy cố gắng nhận diện dựa trên các từ khóa chính và cấu trúc tổng thể.
Phân tích cẩn thận từng dòng một."""),
        ],
    )

    res = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return json.loads(res.text)
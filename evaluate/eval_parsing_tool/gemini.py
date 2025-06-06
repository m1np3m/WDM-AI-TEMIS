from google import genai
from google.genai import types
import base64
import json
from google.oauth2 import service_account

credentials_path = "C:/Users/PC/CODE/WDM-AI-TEMIS/unlimited-461914-b8ca7585af89.json"

# Thêm tham số scopes vào đây
credentials = service_account.Credentials.from_service_account_file(
    credentials_path,
    scopes=['https_://www.googleapis.com/auth/cloud-platform'] # <--- THÊM DÒNG NÀY
)

def generate(ground_truth, predict):
  client = genai.Client(
      vertexai=True,
      project="unlimited-461914", # Đảm bảo project ID này chính xác
      location="global",         # Cân nhắc sử dụng một region cụ thể hơn nếu "global" gây lỗi
                                 # ví dụ: "us-central1" hoặc region bạn thường dùng
      credentials=credentials,
  )


  msg1_text1 = types.Part.from_text(text=f"""You are provided with two markdown tables: a ground truth table and a predicted table.

Ground Truth Table:
{ground_truth}

Predicted Table:
{predict}

Evaluate the similarity between the two tables on a scale of 0 to 1, where 0 indicates no similarity and 1 indicates perfect match. For table fragments (tables without headers), prioritize the content and structure of the tables over the header values.

Provide a similarity score and a brief explanation of your reasoning.""")

  model = "gemini-2.5-flash-preview-05-20"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_text1
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 0,
    top_p = 1,
    seed = 0,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    response_mime_type = "application/json",
    response_schema = {"type":"OBJECT","properties":{"score":{"type":"NUMBER"}}},
    thinking_config=types.ThinkingConfig(
      thinking_budget=0,
    ),
  )

  res = client.models.generate_content(
    model = model,
    contents = contents,
    config = generate_content_config,
  )
  return json.loads(res.text)["score"]

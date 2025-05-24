"""
Sử dụng GOOGLE CLOUD AI nên là phải đăng nhập để có thể sử dụng file này.

Bước 1: Đăng nhập vào Google Cloud AI
```
gcloud auth login
```

Bước 2: Cấu hình Project ID

```
gcloud config set project YOUR_PROJECT_ID
```

Bước 3: Đặt hạn mức chi phí (QUAN TRỌNG)

```
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Bước 4: Đảm bảo Vertex AI được bật
"""
from google import genai
from google.genai import types
import base64
import json
import os
import logging

YOUR_PROJECT_ID = "oval-crawler-460614-d3"



SYSTEM_INSTRUCTION = """You are an expert in table analysis and restructuring. Your primary task is to determine which provided table fragments belong together to form complete logical tables, based on their context and content.

You will be provided with a list of table fragments as one long string of text. Each table fragment will have the following format:

1.  `Table X:`: A unique numerical identifier for the fragment.
2.  `Context before table:`: Text immediately preceding the table fragment. This is the MOST CRUCIAL factor for determining logical tables.
3.  Table content (table_content): The actual table data in markdown format.
4.  `Number of columns:`: The count of columns in this fragment.

Your analysis must follow these rules to group the table fragments:

1.  **Context Determines Table Boundaries (Highest Priority):**
    * ANY table fragment that has non-empty text in the "Context before table:" field starts a NEW logical table group, regardless of content similarity to previous fragments.
    * Empty context (no text in "Context before table:") indicates the fragment is a continuation of the current logical group started by the most recent fragment with context.
    * Special cases:
      - If a fragment has context that appears to be metadata, incomplete text, or page numbers only (e.g., "d t 21 thi", "for Walk the Line (2005).", "Leading Role"), it should still start a new group if it has any context text.
      - Section markers like "Film ​[ edit ]", "Television ​[ edit ]", "Stage ​[ edit ]", "Video games ​[ edit ]", "Discography ​[ edit ]" always start new groups.

2.  **Sequential Processing:** Process table fragments strictly in numerical order (Table 0, Table 1, Table 2, etc.). A fragment can only be grouped with immediately preceding fragments that started or belong to the same logical group.

3.  **Column Count is Secondary:** Column count differences are allowed within the same logical group. Context takes absolute precedence over structural similarities.

4.  **Grouping Logic:**
    * Start with Table 0. If it has context, it forms its own group.
    * For each subsequent table:
      - If it has context: Start a new group
      - If it has no context: Add to the current group (started by the most recent table with context)

Examples of context-based grouping:
- Table with context "1920s ​[ edit ]" → Starts new group
- Next table with no context → Continues the "1920s" group  
- Table with context "1930s ​[ edit ]" → Starts new group
- Multiple tables with no context → All continue the "1930s" group
- Table with context "Film ​[ edit ]" → Starts new group

Please analyze the following table fragments:\n
"""









def generate(full_prompt: str):
    client = genai.Client(
        vertexai=True,
        project=YOUR_PROJECT_ID,
        location="global",
        # api_key=os.getenv("VERTEX_API_KEY"),
    )

    msg1_text1 = types.Part.from_text(text=full_prompt)
    si_text1 = SYSTEM_INSTRUCTION
    model = "gemini-2.0-flash-001"
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
    max_output_tokens = 8192,
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
    response_schema = {"type":"OBJECT","required":["concatable_tables"],"properties":{"concatable_tables":{"type":"ARRAY","items":{"type":"OBJECT","required":["table_index"],"properties":{"table_index":{"type":"ARRAY","items":{"type":"NUMBER"}}}}}}},
    system_instruction=[types.Part.from_text(text=si_text1)],
    )

    input_tokens = client.models.count_tokens(model=model, contents=[si_text1, msg1_text1]).total_tokens
    response = client.models.generate_content(
    model = model,
    contents = contents,
    config = generate_content_config,
    )
    output_tokens = client.models.count_tokens(model=model, contents=response.text).total_tokens
    res_json = response.text
    res_json = json.loads(res_json)
    res_json.update({"input_tokens": input_tokens, "output_tokens": output_tokens, "model": model})
    
    logging.info(f"Model: {model}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
    
    return res_json



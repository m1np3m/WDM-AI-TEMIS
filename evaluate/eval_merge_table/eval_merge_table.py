# To run this code you need to install the following dependencies:
# pip install google-genai
import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2 import service_account
from merge_table_prompt import MERGE_TABLE_PROMPT


# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.WDMParser.extract_tables import full_pipeline, get_pdf_name, get_tables_from_pdf


load_dotenv()

# To use model
credentials_path = os.getenv("CREDENTIALS_PATH")
project_id = os.getenv("VERTEXAI_PROJECT_ID")

# Set up credentials with proper scopes for Vertex AI
scopes = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language",
]

credentials = service_account.Credentials.from_service_account_file(
    credentials_path, scopes=scopes
)


def generate(full_tables, merged_tables):
    client = genai.Client(
        # api_key=os.environ.get("GEMINI_API_KEY"),
        vertexai=True,
        project=project_id,
        location="us-central1",
        credentials=credentials,
    )

    input_prompt = MERGE_TABLE_PROMPT.format(
        full_tables=full_tables, merged_tables=merged_tables
    )

    model = "gemini-2.5-flash-preview-05-20"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["score"],
            properties={
                "score": genai.types.Schema(
                    type=genai.types.Type.NUMBER,
                ),
            },
        ),
    )

    res = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return json.loads(res.text), input_prompt


def prepare_markdown(tables):
    markdown = ""
    for idx, table in enumerate(tables):
        markdown += f"## Tables: {idx}\n\n"
        markdown += f"**Context before:** {table['context_before']}\n\n"
        markdown += f"**Page:** {table['page']}\n\n"
        markdown += f"{table['text']}\n"
        markdown += "\n" + "---" + "\n\n"
    return markdown


if __name__ == "__main__":
    pdf_path = "C:/Users/PC/CODE/WDM-AI-TEMIS/data/pdfs"

    source_pdfs = [
        os.path.join(pdf_path, file)
        for file in os.listdir(pdf_path)
        if file.endswith(".pdf")
    ]

    total_score = []

    for idx, source_pdf in enumerate(source_pdfs):
        
        print(f">> File {idx + 1}/{len(source_pdfs)}: {get_pdf_name(source_pdf)}")
        full_tables, merged_tables = full_pipeline(source_pdf, debug=False, return_full_tables=True, evaluate=True)
        # merged_tables = full_pipeline(source_pdf, debug=False, handle_merge_cell=False)

        print(f"    Tổng cộng có {len(full_tables)} bảng được trích xuất")
        print(f"    Số bảng còn lại sau khi gộp: {len(merged_tables)}")

        full_tables_markdown = prepare_markdown(full_tables)
        merged_tables_markdown = prepare_markdown(merged_tables)

        res, input_prompt = generate(full_tables_markdown, merged_tables_markdown)
        score = res["score"]
        if score > 0.7:
            print(f"✅    Score: {score}")
        else:
            print(f"❌    Score: {score}")
        total_score.append(score)
        # break

    # tạo dict với key là tên file, value là điểm
    score_dict = {get_pdf_name(source_pdf): score for source_pdf, score in zip(source_pdfs, total_score)}

    print(f"Tổng điểm: {sum(total_score)}")
    print(f"Điểm trung bình: {sum(total_score) / len(source_pdfs)}")
    with open("total_score_2.txt", "w", encoding="utf-8") as f:
        f.write(f"Tổng điểm: {sum(total_score)}\n")
        f.write(f"Điểm trung bình: {sum(total_score) / len(source_pdfs)}\n")
        f.write(f"List điểm: {total_score}")
    with open("score_dict_2.json", "w", encoding="utf-8") as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=4)

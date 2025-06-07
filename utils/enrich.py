import os
import time
import json
import base64
import requests

from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

MAX_REQUESTS_PER_KEY = 50

class Enrich_Openrouter:
    def __init__(self, api_url="https://openrouter.ai/api/v1/chat/completions"):
        self.api_url = api_url
        self.summary_text = """
        This image contains a data table or a keyboard shortcut matrix. Please analyze and describe it thoroughly based on the following instructions:

        #### Table Structure:
        - **How many rows and columns are there?**
          - Provide the total number of rows and columns in the table.

        - **What are the headers of each column?**
          - List the names or labels of each column header.

        - **Is there a total row, footer, or any notes?**
          - Indicate if there is a summary row (e.g., total), footer, or additional notes at the bottom of the table.

        #### Data Overview:
        - **List the data row by row if possible.**
          - Present the data in each row, ideally in a structured format.

        - **Explain the meaning of each value in the table cells.**
          - Describe what each value represents (e.g., numerical data, categorical data, etc.).

        - **Clarify any special symbols (e.g., %, $, color codes, icons).**
          - Explain the significance of any symbols, icons, or formatting used in the table.

        #### In-depth Analysis:
        - **Which rows or columns stand out as significant?**
          - Identify rows or columns that contain important information or trends.

        - **Are there any relationships, patterns, or correlations between columns?**
          - Analyze whether certain columns are related or show dependencies.

        - **Identify any upward/downward trends.**
          - Highlight any noticeable trends in the data over time or across categories.

        - **Point out any anomalies or outliers in the data.**
          - Note any unusual values or outliers that deviate from the norm.

        #### Comparison and Interpretation:
        - **Compare values across rows or groups.**
          - Compare data across different rows or groups to identify similarities or differences.

        - **Identify the highest and lowest values.**
          - Determine the maximum and minimum values in the dataset.

        - **What does the table suggest or imply overall?**
          - Summarize the main insights or conclusions drawn from the table.

        #### If the table contains keyboard shortcuts or commands:
        - **Describe the action associated with each shortcut key combination.**
          - Explain what each shortcut does (e.g., Ctrl + C for copy).

        - **Group the shortcuts by functionality if possible (e.g., navigation, editing, system-level commands).**
          - Organize shortcuts into categories based on their purpose.

        #### Presentation:
        - **Please present your response in a well-structured and easy-to-read format.**
          - Use bullet points, numbered lists, or tables where appropriate to enhance readability.
        """

    def get_valid_key(self, request_counters):
        for key, count in request_counters.items():
            if count + 2 <= MAX_REQUESTS_PER_KEY:
                return key
        raise Exception("Tất cả các API key đều đã vượt quá giới hạn request.")

    def prompt_for_summary(self, model, base64_image, prompt_text):
        return json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
        })

    def table_markdown_context(self, model, base64_image, markdown_content, summary_content):
        return json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
        You are given three sources of information related to a single table:

        1. **Raw Extracted Markdown Table**: This is the output from automated table extraction tools. While it contains all the necessary data, its structure may be incorrect — column headers may be misaligned, rows may not match properly, and formatting can be inconsistent.
        Use this only to reference the raw data values, not the structure. Here is the raw extracted markdown table below:
        {markdown_content}
        2. **Table Summary**:This is a detailed description of the original table’s purpose, layout, and the meaning of each column and row.
        Use this as your primary source to guide the table’s structure. Here is the summary of the table:
        {summary_content}
        3. **Table Image (Visual Shortcut)**: This is a visual representation of the actual table. Use it to validate the layout, verify column headers, row alignments, and relationships between data entries.
        This image acts as your ground truth.
        Here is the image:
        ![Table image]

        ---

        ### Your task:
        Based on these three inputs, reconstruct a **well-formatted Markdown table** with accurate column headers, rows, alignment, and structure. Ensure that:

        - All data points from the raw table are included.
        - Pay special attention to:

        * Merged columns (colspan): If a cell like "ABC" spans across multiple columns (e.g., columns A–C), this is a clear signal to merge columns.
        * Merged rows (rowspan): If a value extends downward across multiple rows, reflect this properly in the Markdown layout.
        - The final structure adheres to the format described in the summary.
        - Ensure the total number of rows and columns matches what is shown in the image and described in the summary.
        - Any inconsistencies are resolved using the image as reference.
        - The output is only the corrected Markdown table and nothing else.

        Return only the fixed and properly structured Markdown table, dont use space or enter char.
        """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        })

    def enrich_image(self, api_key, base64_image, markdown_content):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        summary_response = requests.post(
            url=self.api_url,
            headers=headers,
            data=self.prompt_for_summary(
                model="qwen/qwen2.5-vl-32b-instruct:free",
                base64_image=base64_image,
                prompt_text=self.summary_text
            )
        )
        summary_content = summary_response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        time.sleep(3)

        final_markdown_response = requests.post(
            url=self.api_url,
            headers=headers,
            data=self.table_markdown_context(
                model="mistralai/mistral-small-3.1-24b-instruct:free",
                base64_image=base64_image,
                markdown_content=markdown_content,
                summary_content=summary_content
            )
        )
        time.sleep(3)
        return final_markdown_response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

    def full_pipeline(self, file_path, extract_table_markdown, result_path, list_keys):
        results = []
        filename = os.path.basename(file_path)
        request_counters = {key: 0 for key in list_keys}
        api_key = self.get_valid_key(request_counters)
        if not file_path.lower().endswith(".png"):
            print("Không phải ảnh PNG")
            return

        try:
            with open(file_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            enriched_markdown = self.enrich_image(api_key=api_key, base64_image=base64_image, markdown_content=extract_table_markdown)
            request_counters[api_key] += 2
            results.append({
                "image_path": filename,
                "markdown_content": enriched_markdown
            })

        except Exception as e:
            print(f"Lỗi với ảnh {filename}: {e}")
            print("Lưu tiến độ hiện tại...")
            with open(result_path, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, indent=2, ensure_ascii=False)
            return []

        with open(result_path, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=2, ensure_ascii=False)

        return results







class Enrich_VertexAI:
    def __init__(self, model_name="gemini-1.5-flash", credentials_path=None):
        if credentials_path is None:
            raise ValueError("Bạn cần cung cấp đường dẫn tới credentials_path")

        self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.llm = ChatVertexAI(
            model=model_name,
            temperature=0.4,
            max_tokens=2048,
            credentials=self.credentials,
        )

        self.output_parser = StrOutputParser()

        self.summary_text = """
        This image contains a data table or a keyboard shortcut matrix. Please analyze and describe it thoroughly based on the following instructions:
        1. Summarize the table's structure, content, and headers.
        2. Identify repeated patterns, data types, or hierarchical categories.
        3. Highlight any special formatting, such as merged cells, bold/italicized text, or color coding.
        4. Describe whether the table is horizontal, vertical, or matrix-like.
        5. Mention any missing values, inconsistencies, or notes.
        Your response should be detailed and help reconstruct the table's structure later.
        """

    def _decode_image(self, base64_image):
        return base64.b64decode(base64_image)

    def prompt_for_summary(self, base64_image):
        image_bytes = self._decode_image(base64_image)
        prompt = [
            HumanMessage(
                content=[
                    {"type": "text", "text": self.summary_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            )
        ]
        return self.output_parser.invoke(self.llm.invoke(prompt))

    def table_markdown_context(self, base64_image, markdown_content, summary_content):
        context_prompt = f"""
        You are given three sources of information related to a single table:
        1. **Raw Extracted Markdown Table**:
        {markdown_content}
        2. **Table Summary**:
        {summary_content}
        3. **Table Image**: (see below)
        ### Your task:
        Based on these three inputs, reconstruct a **well-formatted Markdown table** with accurate column headers, rows, alignment, and structure. Return only the fixed and properly structured Markdown table without spaces or line breaks.
        """

        prompt = [
            HumanMessage(
                content=[
                    {"type": "text", "text": context_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            )
        ]
        return self.output_parser.invoke(self.llm.invoke(prompt))

    def enrich_image(self, base64_image, markdown_content):
        summary_content = self.prompt_for_summary(base64_image)
        time.sleep(2)  
        return self.table_markdown_context(base64_image, markdown_content, summary_content)

    def full_pipeline(self, file_path, extract_table_markdown, result_path, verbose=1):
        results = []
        filename = os.path.basename(file_path)
        if not file_path.lower().endswith(".png"):
            print("Không phải ảnh PNG")
            return

        try:
            with open(file_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            enriched_markdown = self.enrich_image(base64_image=base64_image, markdown_content=extract_table_markdown)

            results.append({
                "image_path": filename,
                "markdown_content": enriched_markdown
            })

        except Exception as e:
            print(f"Lỗi với ảnh {filename}: {e}")
            print("Lưu tiến độ hiện tại...")
            with open(result_path, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, indent=2, ensure_ascii=False)
            return []

        with open(result_path, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=2, ensure_ascii=False)

        return results



if __name__ == "__main__":
    import json
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    import os

    image_path = "C:/Users/Admin/Data/WDM-AI-TEMIS/data/data-finetune/final_data/extracted_images/aca6e4ba-2349-425b-ba92-9ed6e2747a3d.png"
    markdown_content = "|March24,2010|☑Mexico|0–0|☑Iceland|InternationalFriendly|63,227|\n|--|--|--|--|--|--|\n|June 9,2011|☑Costa Rica|1–1|☑El Salvador|2011CONCACAFGold CupGroup A|46,012|\n|June 9,2011|☑Mexico|5–0|☑Cuba|2011CONCACAFGold CupGroup A|46,012|\n|August2, 2014|☑Liverpool|2–0|☑Milan|2014InternationalChampionsCup|69,364|\n|July 15,2015|☑Cuba|1–0|☑Guatemala|2015CONCACAFGold CupGroup C|55,823|\n|July 15,2015|☑Mexico|4–4|☑Trinidad and Tobago|2015CONCACAFGold CupGroup C|55,823|\n|July 25,2015|☑Chelsea|1–1(6–5 pen.)|☑Paris Saint-Germain|2015InternationalChampionsCup|61,224|\n|July 30,2016|☑BayernMunich|4–1|☑Inter Milan|2016InternationalChampionsCup|53,629|\n|July 22,2018|☑BorussiaDortmund|3–1|☑Liverpool|2018InternationalChampionsCup|55,447|\n|June23,2019|☑Canada|7–0|☑Cuba|2019CONCACAFGold CupGroup A|59,283|\n|June23,2019|☑Mexico|3–2|☑Martinique|2019CONCACAFGold CupGroup A|59,283|\n|July 20,2019|☑Arsenal|3–0|☑Fiorentina|2019InternationalChampionsCup|34,902|\n|October3, 2019|☑United Stateswomen|2–0|☑South Koreawomen|Women’sInternationalFriendly|30,071|\n"
    
    print("******************************")

    processor = Enrich_Openrouter()

    result_path = "./processed_results.json"
    list_keys = [
            os.getenv('API_OPENROUTE_locdinh'),
            os.getenv('API_OPENROUTE_pjx'),
            os.getenv('API_OPENROUTE_ueh'),
            os.getenv('API_OPENROUTE_dynamic'),
            os.getenv('API_OPENROUTE_innolab')
        ]

    results = processor.full_pipeline(image_path, markdown_content, result_path, list_keys)
    print("Pipeline completed. Results saved to:", result_path)
    print("Results:", results)

    print("******************************")
    processor = Enrich_VertexAI(
        credentials_path="C:/Users/Admin/Data/multimodal-rag-baseline/gdsc2025-74596a254ab4.json"
    )


    result_path = "./vertex_chat_results.json"

    results = processor.full_pipeline(image_path, markdown_content, result_path)
    print("Pipeline completed. Results saved to:", result_path)
    print("Results:", results)

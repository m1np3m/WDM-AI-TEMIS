import os
import time
import json
import base64
import requests

from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from .utils_retry import retry_network_call, retry_vertex_ai_call

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
        2. **Table Summary**:This is a detailed description of the original table's purpose, layout, and the meaning of each column and row.
        Use this as your primary source to guide the table's structure. Here is the summary of the table:
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

    @retry_network_call
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
            ),
            timeout=60  # Add timeout to prevent hanging
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
            ),
            timeout=60  # Add timeout to prevent hanging
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
    def __init__(self, model_name="gemini-2.0-flash-001", credentials_path=None):
        if credentials_path is None:
            raise ValueError(
                "credentials_path is required. "
                "Please provide the path to your Google Cloud service account JSON key file. "
                "Example: Enrich_VertexAI(credentials_path='/path/to/your/service-account-key.json')"
            )
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found at: {credentials_path}. "
                "Please ensure the path is correct and the file exists."
            )

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
        6. Are ther any merge collumns or rows in the table, describe it carefully?
        Your response should be detailed and help reconstruct the table's structure later.
        """

    def _decode_image(self, base64_image):
        return base64.b64decode(base64_image)

    @retry_vertex_ai_call
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
        result = self.output_parser.invoke(self.llm.invoke(prompt))
        # Add small delay after successful call to avoid overwhelming API
        time.sleep(2)
        return result

    @retry_vertex_ai_call
    def table_markdown_context(self, base64_image, markdown_content, summary_content):
        context_prompt = f"""
        You are given three sources of information related to a single table:
        1. **Raw Extracted Markdown Table**:
        {markdown_content}
        2. **Table Summary**:
        {summary_content}
        3. **Table Image**: (see below)
        ### Note:
        Note that, when the table has merged rows, the Markdown format will not show the duplicate rows or columns for the merged cells. Instead, it will show the first row or column with the content, and the subsequent rows or columns will be left empty.
        Look from pdf it look like this:
        | STT | Họ tên       | Môn học      | Điểm |
        |-----|--------------|--------------|------|
        | 1   | Nguyễn Văn A | Toán         | 8    |
        |     |              | Lý           | 7    |
        |     |              | Hóa          | 9    |
        | 2   | Trần Thị B   | Toán         | 8.5  |
        |     |              | Lý           | 6.5  |
        But If table merged!Output rows when you returns need to look like this, we need all meaning from the table:
        | STT | Họ tên       | Môn học      | Điểm |
        |-----|--------------|--------------|------|
        | 1   | Nguyễn Văn A | Toán         | 8    |
        |     | Nguyễn Văn A | Lý           | 7    |
        |     | Nguyễn Văn A | Hóa          | 9    |
        | 2   | Trần Thị B   | Toán         | 8.5  |
        |     | Trần Thị B   | Lý           | 6.5  |
        If the table merged columns, it will look like this:
        Input table:
        | STT | 2023         | 2024         | Tổng điểm |
        |-----|--------------|--------------|-----------|
        | 1   | Nguyễn Văn A | Trần Thị B   | 16        |
        | 2   | Lê Văn C                    | 15        |
        Output (Expected Markdown):
        | STT | 2023         | 2024         | Tổng điểm |
        |-----|--------------|--------------|-----------|
        | 1   | Nguyễn Văn A | Trần Thị B   | 16        |
        | 2   | Lê Văn C     | Lê Văn C     | 15        |
        ### Your task:
        Please, based on these three inputs, reconstruct a **well-formatted Markdown table** with accurate column headers, rows, alignment, and structure. Return only the fixed and properly structured Markdown table without spaces or line breaks.
        Reconstruct a well-formatted Markdown table that reflects the original visual layout and full data of the source table, including all values in merged rows and columns.
        Do not change the number of rows or columns .
        Fill in empty cells caused by merging with the correct duplicated or inferred values.
        Maintain Markdown syntax and alignment .
        Return only the final clean Markdown table , no extra text.
        """

        prompt = [
            HumanMessage(
                content=[
                    {"type": "text", "text": context_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            )
        ]
        result = self.output_parser.invoke(self.llm.invoke(prompt))
        # Add small delay after successful call to avoid overwhelming API
        time.sleep(2)
        return result

    def enrich_image(self, base64_image, markdown_content):
        summary_content = self.prompt_for_summary(base64_image)
        # Small delay between API calls
        time.sleep(1)  
        return self.table_markdown_context(base64_image, markdown_content, summary_content)

    def full_pipeline(self, file_path, extract_table_markdown, result_path, verbose=1, return_markdown=False):
        results = []
        filename = os.path.basename(file_path)
        if not file_path.lower().endswith(".png"):
            print("Không phải ảnh PNG")
            return

        try:
            with open(file_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            enriched_markdown = self.enrich_image(base64_image=base64_image, markdown_content=extract_table_markdown)

            if return_markdown:
                return enriched_markdown

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

import time
import requests
import json
import os
import random
import base64
MAX_REQUESTS_PER_KEY = 50
summary_text = """
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

def get_valid_key(request_counters):
    """
    Lấy key còn quota để sử dụng, ưu tiên theo thứ tự trong list_keys.
    """
    for key, count in request_counters.items():
        if count + 2 <= MAX_REQUESTS_PER_KEY:
            return key
    raise Exception("Tất cả các API key đều đã vượt quá giới hạn request.")


def prompt_for_summary(model = "qwen/qwen2.5-vl-32b-instruct:free", base64_image = None, prompt_text= summary_text):
    return json.dumps({
            "model": model,
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                    }
                }
                ]
            }
            ],
            
        })

def table_markdown_context(model, base64_image, markdown_content , summary_content):
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
            ],
            
        })


def VLM_enrichdata(
        url= "https://openrouter.ai/api/v1/chat/completions", 
        key= None,
        summary_context= None, 
        base64_image=None,
        markdown_content=''
        ):
    
    headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
    
    summary_content=requests.post(
        url=url,
        headers=headers,
        data= prompt_for_summary(model="qwen/qwen2.5-vl-32b-instruct:free", base64_image=base64_image, prompt_text=summary_context)
        )
    summary_content = summary_content.json().get("choices", [{}])[0].get("message", {}).get("content", "")

    time.sleep(3)  

    time.sleep(3)  


    table_markdown = requests.post(
        url=url,
        headers=headers,
        data= table_markdown_context("mistralai/mistral-small-3.1-24b-instruct:free", base64_image, markdown_content, summary_content)
        )
    
    time.sleep(3)  
    return table_markdown.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    

def full_pipeline_vlm(source_path, markdown_map, result_path, list_keys, verbose=1):
    """
    Hàm xử lý toàn bộ pipeline với VLM.
    Lưu kết quả hiện tại vào file JSON nếu có lỗi xảy ra.
    """
    image_extensions = ('.png',)
    results = []
    output_json_path = result_path
    request_counters = {key: 0 for key in list_keys}

    for filename in os.listdir(source_path):
        file_path = os.path.join(source_path, filename)

        # key = random.choice([
        #     os.getenv('API_OPENROUTE_locdinh'),
        #     os.getenv('API_OPENROUTE_pjx'),
        #     os.getenv('API_OPENROUTE_ueh'),
        #     os.getenv('API_OPENROUTE_dynamic'),
        #     os.getenv('API_OPENROUTE_innolab')
        # ])

        if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
            if verbose >= 1:
                print(f"Đang xử lý ảnh: {filename}")

            try:
                extract_table_markdown = markdown_map.get(filename)
                print(f"Extracted markdown for {filename}: {extract_table_markdown}")

                if not extract_table_markdown:
                    print(f"Không tìm thấy dữ liệu cho ảnh: {filename}")
                    continue

                if extract_table_markdown == "No suitable table found meeting threshold":
                    continue
                api_key = get_valid_key(request_counters)
                with open(file_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                table_markdown = VLM_enrichdata(
                    key=api_key,
                    base64_image=base64_image,
                    summary_context=summary_text,
                    markdown_content=extract_table_markdown
                )
                request_counters[api_key] += 2

                result_item = {
                    "image_path": filename,
                    "markdown_content": table_markdown,
                }
                print(result_item)
                results.append(result_item)

            except Exception as e:
                print(f"Lỗi xảy ra với ảnh {filename}: {e}")
                print("Lưu tiến độ hiện tại vào file JSON...")
                
                json_result = json.dumps(results, indent=2, ensure_ascii=False)
                with open(output_json_path, 'w', encoding='utf-8') as json_file:
                    json_file.write(json_result)
                
                return [] 

    json_result = json.dumps(results, indent=2, ensure_ascii=False)
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_result)

    return results

if __name__ == "__main__":
    source_path = "data/images"
    markdown_map = {
        "example_image.png": "| Header1 | Header2 |\n|---------|---------|\n| Data1   | Data2   |"
    }
    result_path = "data/results/vlm_enrich_results.json"

    results = full_pipeline_vlm(source_path, markdown_map, result_path, verbose=1)
    print("Pipeline completed. Results saved to:", result_path)
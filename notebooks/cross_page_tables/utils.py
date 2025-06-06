import json
import os
from typing import List

from google import genai
from google.genai import types
from google.oauth2 import service_account
from dotenv import load_dotenv

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


def get_is_new_section_context(contexts: List[str], return_prompt: bool = False):
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location="us-central1",  # Changed from "global" to a specific region
        credentials=credentials,
    )

    si_text1 = (
        "You are an expert in document structure analysis. Your task is to examine text segments that appear "
        "immediately before tables or sections, and determine if they clearly indicate the start of a new section, "
        "item, or table.\n\n"
        "You will be provided with a numbered list of contexts (Context 1, Context 2, etc.). Each context is the text "
        "that appears immediately before a table in a document. You need to return a list of boolean values (True or "
        "False) of the same length, where each boolean corresponds to your decision for the context at the respective "
        "position (Context 1 → first boolean, Context 2 → second boolean, etc.).\n\n"
        "Criteria for deciding True (Indicates new section/table):\n"
        "- Clear title or heading\n"
        '- Structured heading (e.g., "Chapter 1", "Section A", "Table 1: ...")\n'
        "- Introductory context that clearly introduces a new topic/section\n\n"
        "Criteria for deciding False (Does NOT indicate new section/table):\n"
        "- Empty context (marked as [EMPTY])\n"
        "- Seamless content continuation from previous text\n"
        "- No structured heading or title\n"
        "- Just data or supplementary description\n"
        "- Fragment of previous content\n\n"
        "Requirements:\n"
        "- Analyze each numbered context individually\n"
        "- Apply the above criteria to decide True or False for each context\n"
        "- Always return False for [EMPTY] contexts\n"
        "- Return the result as a list of boolean values in the same order as the input contexts\n"
        "- The output list must have exactly the same length as the input list"
    )

    model = "gemini-2.0-flash-001"

    # Format contexts with clear indexing
    formatted_contexts = [
        f"Context {i}:\n{context.strip() if context.strip() else '[EMPTY]'}"
        for i, context in enumerate(contexts, 1)
    ]

    contexts_text = "\n\n".join(formatted_contexts)

    input_text = f"\n\n### List of Contexts Before Tables:\n\n{contexts_text}\n\n### Total number of contexts: {len(contexts)}"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=input_text)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        top_p=1,
        max_output_tokens=8192,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
            ),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "required": ["is_new_section_context"],
            "properties": {
                "is_new_section_context": {
                    "type": "ARRAY",
                    "items": {"type": "BOOLEAN"},
                }
            },
        },
        system_instruction=[types.Part.from_text(text=si_text1)],
    )

    res = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    if return_prompt:
        return json.loads(res.text), si_text1 + input_text
    return json.loads(res.text)


def get_is_has_header(
    rows: List[List[str]], first_3_rows: List[str], return_prompt: bool = False
):
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location="us-central1",  # Changed from "global" to a specific region
        credentials=credentials,
        #
    )

    model = "gemini-2.0-flash"

    # Format table information with both headers and context
    formatted_tables = []
    for i, (header_row, table_context) in enumerate(zip(rows, first_3_rows), 1):
        # Format header row
        if header_row:
            formatted_cells = []
            for cell in header_row:
                cell_content = str(cell).strip() if cell else ""
                formatted_cells.append(cell_content if cell_content else "[EMPTY_CELL]")
            header_text = " | ".join(formatted_cells)
        else:
            header_text = "[NO_HEADER_EXTRACTED]"

        # Format table context (first 3 rows in markdown)
        table_preview = table_context.strip() if table_context else "[EMPTY_TABLE]"

        formatted_table = f"""\nTable {i}:
Header Row: {header_text}
Table Preview (First 3 rows):
{table_preview}"""

        formatted_tables.append(formatted_table)

    tables_text = "\n\n" + "=" * 50 + "\n\n".join(formatted_tables)

    input_text = f"\n\n### Tables Analysis:\n{tables_text}\n\n### Total number of tables: {len(rows)}"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=input_text)],
        ),
    ]

    si_text = """You are an expert in analyzing table data structures. Your task is to examine tables and determine if their first row contains meaningful column headers.

You will receive information about multiple tables. For each table, you'll see:
1. "Header Row": The extracted first row that might be headers
2. "Table Preview": A markdown preview of the first 3 rows to provide context

You need to return a list of boolean values (True or False) of the same length, where each boolean corresponds to your analysis of whether the table at the respective position has a meaningful header row (Table 1 → first boolean, Table 2 → second boolean, etc.).

A meaningful header row contains column names that describe the type of data that will appear in those columns in subsequent rows, rather than specific data values.

**Criteria for determining a header row as meaningful (True):**
- Contains descriptive column names (e.g., "Name", "Date", "Amount", "Description", "Status")
- Uses generic categorical terms rather than specific data values
- Typically concise and descriptive labels
- Does not contain specific identifiers, dates, numbers, or actual data values
- May contain formatting indicators like "Title", "Category", "Type", etc.
- Headers are consistent with the data pattern shown in the table preview

**Criteria for determining a header row is NOT meaningful (False):**
- Contains specific data values instead of column names (e.g., "John Smith", "2023-01-01", "1000", specific IDs)
- Starts with ordinal numbers, dates, or specific identifiers
- Contains complete sentences or long descriptive paragraphs
- Contains actual data that should be in body rows
- [EMPTY_CELL] for most or all cells
- [NO_HEADER_EXTRACTED] indicates no header was found
- The header row looks like data when compared to subsequent rows in the preview

**Important Analysis Guidelines:**
- Compare the "Header Row" with the actual data shown in "Table Preview"
- If the header row contains the same type of content as subsequent rows, it's likely data, not headers
- Use the table preview to understand the data pattern and validate if the header makes sense
- Headers should be descriptive labels, not data entries
- Consider the overall structure and consistency of the table

**Output Requirements:**
- Analyze each table individually using both the header row and table preview
- Return exactly one boolean per table in the same order as input
- The output list must have exactly the same length as the input list
- Be conservative: when in doubt, prefer False unless clearly header-like content"""

    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        top_p=1,
        max_output_tokens=8192,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
            ),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "required": ["is_has_header"],
            "properties": {
                "is_has_header": {"type": "ARRAY", "items": {"type": "BOOLEAN"}}
            },
        },
        system_instruction=[types.Part.from_text(text=si_text)],
    )

    res = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    if return_prompt:
        return json.loads(res.text), si_text + input_text
    return json.loads(res.text)


if __name__ == "__main__":
    ### TESTING CASES

    print("TEST CONTEXT BEFORE TABLE")

    contexts = [
        "Table 2: Population by continent (2023 estimate)",
        "Bảng 3: Danh sách các trường đại học hàng đầu thế giới",
        "The quick brown fox jumps over the lazy dog.",
        "Yesterday, it rained heavily in the city center.",
    ]

    rows1 = [
        ["Year", "Population", "Continent"],
        ["2023", "7.9 billion", "Asia"],
    ]

    first_3_rows1 = [
        "| Year | Population | Continent |\n| 2023 | 7.9 billion | Asia |\n| 2022 | 7.8 billion | Africa |",
        "|2023 | 7.9 billion | Asia |\n|2022 | 7.8 billion | Africa |",
    ]

    res, prompt = get_is_new_section_context(contexts, return_prompt=True)
    print("=====\tPrompt====\n", prompt)
    print("=====\tResult====\n", res)

    res, prompt = get_is_has_header(rows1, first_3_rows1, return_prompt=True)
    print("=====\tPrompt====\n", prompt)
    print("=====\tResult====\n", res)

    print("=====\tTESTING CASES END====\n")

import os
from google import genai
from google.genai import types
import json
from typing import List


def get_is_new_section_context(contexts: List[str]):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    
    # Format contexts with clear indexing
    formatted_contexts = []
    for i, context in enumerate(contexts, 1):
        formatted_contexts.append(f"Context {i}:\n{context.strip() if context.strip() else '[EMPTY]'}")
    
    contexts_text = "\n\n".join(formatted_contexts)
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"### List of Contexts Before Tables:\n\n{contexts_text}\n\n### Total number of contexts: {len(contexts)}"),
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
            types.Part.from_text(text="""You are an expert in document structure analysis. Your task is to examine text segments that appear immediately before tables or sections, and determine if they clearly indicate the start of a new section, item, or table.

You will be provided with a numbered list of contexts (Context 1, Context 2, etc.). Each context is the text that appears immediately before a table in a document. You need to return a list of boolean values (True or False) of the same length, where each boolean corresponds to your decision for the context at the respective position (Context 1 → first boolean, Context 2 → second boolean, etc.).

Criteria for deciding True (Indicates new section/table):
- Clear title or heading
- Structured heading (e.g., "Chapter 1", "Section A", "Table 1: ...")
- Introductory context that clearly introduces a new topic/section

Criteria for deciding False (Does NOT indicate new section/table):
- Empty context (marked as [EMPTY])
- Seamless content continuation from previous text
- No structured heading or title
- Just data or supplementary description
- Fragment of previous content

Requirements:
- Analyze each numbered context individually
- Apply the above criteria to decide True or False for each context
- Always return False for [EMPTY] contexts
- Return the result as a list of boolean values in the same order as the input contexts
- The output list must have exactly the same length as the input list"""),
        ],
    )

    res =  client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return json.loads(res.text)



def get_is_has_header(rows: List[List[str]]):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    
    # Format header rows with clear indexing and structure
    formatted_rows = []
    for i, row in enumerate(rows, 1):
        # Handle empty rows or rows with empty cells
        formatted_cells = []
        for cell in row:
            cell_content = str(cell).strip() if cell else ""
            formatted_cells.append(cell_content if cell_content else "[EMPTY_CELL]")
        
        row_text = " | ".join(formatted_cells)
        formatted_rows.append(f"Header Row {i}: {row_text}")
    
    rows_text = "\n\n".join(formatted_rows)
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"### Header Rows from Different Tables:\n\n{rows_text}\n\n### Total number of header rows: {len(rows)}"),
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
            types.Part.from_text(text="""You are an expert in analyzing table data structures. Your task is to examine header rows extracted from different tables and determine if they contain meaningful column headers.

You will receive a numbered list of header rows (Header Row 1, Header Row 2, etc.). Each header row contains cells separated by " | ". Each row represents the first row extracted from a different table. You need to return a list of boolean values (True or False) of the same length, where each boolean corresponds to your analysis of the header row at the respective position (Header Row 1 → first boolean, Header Row 2 → second boolean, etc.).

A meaningful header row contains column names that describe the type of data that will appear in those columns in subsequent rows, rather than specific data values.

Criteria for determining a header row as meaningful (True):
- Contains descriptive column names (e.g., "Name", "Date", "Amount", "Description", "Status")
- Uses generic categorical terms rather than specific data
- Typically concise and descriptive
- Does not contain specific ordinal data, ID numbers, or actual data values
- May contain formatting indicators like "Title", "Category", "Type", etc.

Criteria for determining a header row is NOT meaningful (False):
- Contains specific data values instead of column names
- Starts with ordinal numbers, dates, or specific identifiers (e.g., "001", "2023-01-01", "John Smith")
- Contains complete sentences or long descriptive paragraphs
- Contains actual data that should be in body rows
- Contains [EMPTY_CELL] for most cells

Important notes:
- Analyze each numbered header row individually
- The presence of HTML tags like <br> does not change the nature of a cell being a header or data
- Minor formatting errors in headers (e.g., "Title D", "irected by") should still be considered as headers if the intent is clear
- [EMPTY_CELL] indicates an empty cell in the original table
- Return the result as a list of boolean values in the same order as the input header rows
- The output list must have exactly the same length as the input list"""),
        ],
    )

    res = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return json.loads(res.text)
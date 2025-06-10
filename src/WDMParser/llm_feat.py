import json
import os
import re
from typing import List

from google.oauth2 import service_account

from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate


from .utils_retry import retry_network_call

_credentials = None
_project_id = None


class IsNewSectionContext(BaseModel):
    is_new_section_context: List[bool] = Field(
        description="List of boolean values indicating whether the context indicates a new section, item, or table"
    )


class IsHasHeader(BaseModel):
    is_has_header: List[bool] = Field(
        description="List of boolean values indicating whether the table has a header"
    )


def clean_json_response(response: str) -> str:
    """
    Clean JSON response by removing markdown code blocks and extra whitespace.
    
    Args:
        response: Raw response string that may contain markdown formatting
        
    Returns:
        Cleaned JSON string
    """
    # Remove markdown code block markers
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*$', '', response)
    
    # Convert Python-style booleans to JSON-style booleans
    response = re.sub(r'\bTrue\b', 'true', response)
    response = re.sub(r'\bFalse\b', 'false', response)
    
    # Remove any leading/trailing whitespace
    response = response.strip()
    
    return response


def parse_json_response(response_text: str, model_class):
    """
    Parse JSON response with fallback for markdown-wrapped JSON.
    
    Args:
        response_text: Raw response from LLM
        model_class: Pydantic model class to parse into
        
    Returns:
        Parsed model instance
    """
    # First try to parse as-is
    try:
        json_data = json.loads(response_text)
        return model_class(**json_data)
    except json.JSONDecodeError:
        # Try cleaning markdown formatting
        cleaned_text = clean_json_response(response_text)
        try:
            json_data = json.loads(cleaned_text)
            return model_class(**json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nOriginal text: {response_text}\nCleaned text: {cleaned_text}")


def _get_credentials():
    """Lazy load credentials when needed."""
    global _credentials, _project_id

    if _credentials is None:
        credentials_path = os.getenv("CREDENTIALS_PATH")
        _project_id = os.getenv("VERTEXAI_PROJECT_ID")

        if not credentials_path:
            raise ValueError(
                "CREDENTIALS_PATH environment variable is not set. "
                "Please set it to the path of your Google Cloud service account JSON key file."
            )

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found at: {credentials_path}. "
                "Please ensure the path is correct and the file exists."
            )

        # Set up credentials with proper scopes for Vertex AI
        scopes = [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/generative-language",
        ]

        _credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=scopes
        )

    return _credentials, _project_id


@retry_network_call
def get_is_new_section_context(contexts: List[str], return_prompt: bool = False):
    credentials, project_id = _get_credentials()
    model = "gemini-2.0-flash-001"

    llm = ChatVertexAI(
        model=model,
        temperature=0,
        max_tokens=8192,
        credentials=credentials,
    )

    template = """You are an expert in document structure analysis. Your task is to examine text segments that appear 
immediately before tables or sections, and determine if they clearly indicate the start of a new section, 
item, or table.

You will be provided with a numbered list of contexts (Context 1, Context 2, etc.). Each context is the text 
that appears immediately before a table in a document. You need to return a list of boolean values (True or 
False) of the same length, where each boolean corresponds to your decision for the context at the respective 
position (Context 1 → first boolean, Context 2 → second boolean, etc.).

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
- The output list must have exactly the same length as the input list
- Return ONLY a JSON object in this exact format: {{"is_new_section_context": [true, false, ...]}}
- Do NOT wrap the JSON in markdown code blocks or any other formatting

### List of Contexts Before Tables:

{contexts_text}

### Total number of contexts: {len_contexts}

Return ONLY the JSON response without any additional text or formatting."""

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["contexts_text", "len_contexts"],
    )

    # Format contexts with clear indexing
    formatted_contexts = [
        f"Context {i}:\n{context.strip() if context.strip() else '[EMPTY]'}"
        for i, context in enumerate(contexts, 1)
    ]
    contexts_text = "\n\n".join(formatted_contexts)

    # Create the chain and invoke
    chain = prompt_template | llm
    response = chain.invoke(
        input={
            "contexts_text": contexts_text,
            "len_contexts": len(contexts),
        }
    )
    
    # Parse the response
    result = parse_json_response(response.content, IsNewSectionContext)
    
    if return_prompt:
        return result.is_new_section_context, prompt_template.format(
            contexts_text=contexts_text,
            len_contexts=len(contexts),
        )

    return result.is_new_section_context


@retry_network_call
def get_is_has_header(
    rows: List[List[str]], first_3_rows: List[str], return_prompt: bool = False
):
    credentials, project_id = _get_credentials()
    model = "gemini-2.0-flash-001"

    llm = ChatVertexAI(
        model=model,
        temperature=0,
        max_tokens=8192,
        credentials=credentials,
    )

    template = """You are an expert in analyzing table data structures. Your task is to examine tables and determine if their first row contains meaningful column headers.

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
- Be conservative: when in doubt, prefer False unless clearly header-like content
- Return ONLY a JSON object in this exact format: {{"is_has_header": [true, false, ...]}}
- Do NOT wrap the JSON in markdown code blocks or any other formatting

### Tables Analysis:

{tables_text}

### Total number of tables: {len_rows}

Return ONLY the JSON response without any additional text or formatting."""

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["tables_text", "len_rows"],
    )

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

        formatted_table = f"""Table {i}:
Header Row: {header_text}
Table Preview (First 3 rows):
{table_preview}"""

        formatted_tables.append(formatted_table)

    tables_text = "\n\n".join(formatted_tables)

    # Create the chain and invoke
    chain = prompt_template | llm
    response = chain.invoke(input={"tables_text": tables_text, "len_rows": len(rows)})
    
    # Parse the response
    result = parse_json_response(response.content, IsHasHeader)

    if return_prompt:
        return result.is_has_header, prompt_template.format(
            tables_text=tables_text, len_rows=len(rows)
        )

    return result.is_has_header

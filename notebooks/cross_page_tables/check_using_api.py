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


YOUR_PROJECT_ID = "oval-crawler-460614-d3"


import json
import logging

from google import genai
from google.genai import types


def generate(full_prompt: str):
    client = genai.Client(
        vertexai=True,
        project=YOUR_PROJECT_ID,
        location="global",
    )

    msg1_text1 = types.Part.from_text(text=full_prompt)
    si_text1 = """You are an AI assistant specialized in analyzing and grouping data tables for vertical concatenation. Your primary goal is to identify sets of tables that can be meaningfully combined based on their context, structure, and headers. You will be provided with information for multiple tables, presented sequentially.

Input Format:
Each "Table X:" block provides several pieces of information:
Context before table (CBT): (Optional) Text preceding the table, often indicating its subject matter, title, or the section it belongs to. This context is critically important for grouping.
Table index: A numerical identifier (e.g., Table 0).
Markdown table content: One or more lines of table content in Markdown format. This may include Markdown header lines (text before a separator line), separator lines (e.g., |:---|:---|), and data lines. Some table blocks might primarily define headers, while others are predominantly data segments.
Number of columns (NOC): The count of columns in the table, provided externally.
Column types: A list of data types for the columns, provided externally.

Your Tasks (leading to a structured JSON output):

First: Identify Tables Establishing Effective Headers
For each table block (e.g., Table 0, Table 1), determine its effective column headers.
An "effective header row" is the row within the Markdown content that best defines the column names for that table or initiates the headers for a group.
1. Examine the Markdown content of the current table block.
2. If the Markdown content contains a standard header structure, which typically includes a row of potential names, followed by a Markdown separator line (e.g., |:---|:---| or similar), followed by what appears to be the first row of data:
    a. Crucially, if this first row of data (immediately after the separator line) clearly represents a list of actual column names (e.g., like "| Model | Year | Notes |" as often seen in the first table of a series that defines the structure for subsequent data), then these names from this first data row are considered the "effective column headers". This table is considered to establish these headers.
    b. If rule (2a) does not apply, then the row *before* the Markdown separator line is considered the "effective column headers". This table is considered to establish these headers.
3. If a table block's Markdown content does *not* begin with its own distinct header structure as described in (2) (i.e., it starts directly with data rows, even if a Markdown separator line appears at the *end* of its data segment but not at its beginning following a header row), it does not establish its own new effective headers. It is presumed to inherit headers from a preceding table in its group.
Compile a list of table indexes that establish their own effective column headers based on these rules. This list will form the `has_headers` field in the output JSON.

Second: Identify Concatable Table Groups
Your main task is to group tables that can be vertically concatenated. This process must result in every table index being assigned to a group in the `concatable_tables` output, even if a group contains only a single table due to its unique characteristics. Apply the following criteria hierarchically:
1. Primary Separator - Context before table (CBT): This is the most important criterion for separating groups.
    a. When a table is encountered, compare its CBT (if present) with the CBT of the active or immediately preceding table/group.
    b. A new or significantly different CBT must initiate a new table group. This rule takes precedence over structural similarities.
    c. Tables with no CBT are generally considered continuations of the current CBT group, provided other criteria (NOC, Headers) match.
2. Structural Compatibility (within a CBT-defined group): Once tables are tentatively associated by CBT (or lack of new CBT), they must also be structurally compatible to belong to the same concatenation group:
    a. Number of Columns (NOC): All tables within a single concatenation group must have the exact same externally provided `Number of columns`. A change in NOC, even if the CBT implies continuation, forces a new group.
    b. Consistent Header Structure: All tables within a group must share the same effective column headers (as determined in "Task First" or inherited).
        i. Parse the column names from their "effective column headers". For tables to be in the same group, their parsed column names (and their order) must be strictly identical after applying any specified column-name-specific normalizations (like the 'Source' rule detailed below).
        ii. A difference in any column name not covered by a specific normalization rule (e.g., "Crew role, notes" vs. "Notes") makes the headers incompatible, forcing a new group.
        iii. Normalization Rule (Specific Cases): The only current normalization is for 'Source' columns: header names like Source[13], Source[XYZ], Source[13][35] etc., should be treated as a canonical "Source" when comparing headers for this specific column name. This rule applies only to columns named 'Source' or its direct variants. It does not make entire header sets compatible if other column names differ.

Third: Determine Headers and "Guessed" Status for All Tables
For every table, provide its definitive column headers and whether these headers were established by the table itself or inferred/inherited. This will inform the `headers_info` field in the output.
1. For tables that established their own "effective column headers" (as per "Task First"):
    `headers`: Use the exact column names parsed (applying the 'Source' normalization if applicable to a column name).
    `is_header_guessed`: `false`.
2. For tables that did *not* establish their own "effective column headers" (i.e., they are continuations or lack a clear self-defined header):
    `headers`:
        a. If it is part of a group where headers are established by another table's "effective column headers" (typically the first such table in that group, like `Table 0` in your example), it inherits those (potentially normalized) headers.
        b. If it is part of a group where no table possesses "effective column headers" (e.g., the group starts due to a CBT change, and the first table in this new group lacks clear headers as per Task First rules, and subsequent tables also lack them but match on CBT and NOC), then all tables in this group will use consistently applied generic headers like ["Column 1", "Column 2", ...] based on the group's NOC.
        c. If it forms a new group by itself (e.g., due to a NOC change not aligning it with any group that has established headers, or it is a standalone table with unique CBT and no clear "effective column headers"), use generic headers as above based on its NOC.
    `is_header_guessed`: `true`.

Finally: Consolidate Header Information Output
For the `headers_info` part of the JSON output: if multiple table indexes (particularly those within the same `concatable_tables` group) share the exact same list of headers AND the same `is_header_guessed` status, consolidate them into a single `headers_info` entry. This entry's `table_index` field will be an array listing all such table indexes. Tables not fitting this consolidation pattern will have their own `headers_info` entries.

Critical Guidelines for JSON Output Generation:
When generating the final JSON output, pay strict attention to the following to ensure its validity:
String Formatting: All string values (e.g., column names in the headers list) must be correctly formatted.
Enclose all strings in double quotes (").
Inside any string value, special characters must be escaped:
A literal double quote (") must be written as \\".
A literal backslash (\) must be written as \\\\.
A newline character must be written as \n.
A tab character must be written as \t.
A carriage return character must be written as \r.
A backspace character must be written as \\b.
A form feed character must be written as \\f.
Ensure every string opened with a double quote is properly terminated with a closing double quote. Adhering to this will prevent Unterminated string errors and ensure the JSON is parsable.

Below is the table data:"""

    model = "gemini-2.0-flash-lite-001"
    contents = [
        types.Content(role="user", parts=[msg1_text1]),
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
            "required": ["concatable_tables"],
            "properties": {
                "concatable_tables": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "required": ["table_index"],
                        "properties": {
                            "table_index": {
                                "type": "ARRAY",
                                "items": {"type": "NUMBER"},
                            }
                        },
                    },
                }
            },
        },
        system_instruction=[types.Part.from_text(text=si_text1)],
    )

    input_tokens = client.models.count_tokens(
        model=model, contents=[si_text1, msg1_text1]
    ).total_tokens
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    output_tokens = client.models.count_tokens(
        model=model, contents=response.text
    ).total_tokens
    res_json = response.text
    res_json = json.loads(res_json)
    res_json.update(
        {"input_tokens": input_tokens, "output_tokens": output_tokens, "model": model}
    )

    logging.info(
        f"Model: {model}, Input tokens: {input_tokens}, Output tokens: {output_tokens}"
    )

    return res_json

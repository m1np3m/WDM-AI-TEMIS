import os
from google import genai
from google.genai import types
import json
from typing import List


def get_is_new_section_context(contexts):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="\n\n".join(contexts) + "\nNumber of contexts: " + str(len(contexts))),
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
            types.Part.from_text(text="""You are an expert in document structure analysis. Your task is to examine a text segment (CONTEXT_BEFORE) that appears immediately before a table or section, and determine if it clearly indicates the start of a new section, item, or table.

You will be provided with a list of CONTEXT_BEFORE strings. You need to return a list of boolean values (True or False) of the same length, where each boolean corresponds to your decision for the CONTEXT_BEFORE at the respective position.

Criteria for deciding True (Indicates new section/table):
- Clear title
- Structured heading
- Introductory context

Criteria for deciding False (Does NOT indicate new section/table):
- Empty string
- Seamless content
- No structured heading
- Just data or supplementary description

Requirements:

- Carefully analyze each CONTEXT_BEFORE in the input list.
- Apply the above criteria to decide True or False for each context.
- Always return False for empty CONTEXT_BEFORE values.
- Return the result as a list of boolean values.
- Always return a complete list of boolean values for the input list."""),
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
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="\n\n".join("\t".join(row) for row in rows) + "\nNumber of elements: " + str(len(rows))),
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
            types.Part.from_text(text="""You are an expert in analyzing table data structures. Your task is to examine each provided "row" and determine if it is a meaningful header row. A meaningful header row contains column names that describe the type of data that will appear in those columns in subsequent rows, rather than specific data.

You will receive a list of rows. Each row in this list is itself a list of strings, where each string represents the content of a cell in that row.

You need to return a list of boolean values (True or False) of the same length as the input list of rows. A value of True means the corresponding row IS a meaningful header, and False means it is NOT.

Criteria for determining a row as a meaningful Header (True):
- Descriptive structure, not specific data
- Uses generic column names
- Typically concise and categorical
- Does not start with specific ordinal data or ID

Criteria for determining a row is NOT a meaningful Header (False):
- Contains specific data
- Starts with ordinal or data value
- Is a complete sentence or long descriptive paragraph

Important notes:

- The presence of HTML tags like <br> does not change the nature of a cell being a header or data.
- Sometimes headers may be interrupted or have minor formatting errors (e.g., "Title D", "irected by"), try to identify based on key terms and overall structure.
- Always return enough boolean values for the input list.
- Carefully analyze each row one by one."""),
        ],
    )

    res = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return json.loads(res.text)
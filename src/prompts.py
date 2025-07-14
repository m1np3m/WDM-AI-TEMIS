# prompts.py

GRADE_PROMPT = """You are a grader assessing relevance of retrieved documents to a user question.

Retrieved documents: 
{context}

User question: {question}

Evaluation criteria:
- If documents contain DIRECTLY relevant information: score 'yes'
- If documents contain PARTIALLY relevant or background information: score 'yes'  
- If the question is a simple greeting, personal question, or general conversation that doesn't require specific document information: score 'yes' (to avoid unnecessary rewriting)
- Only score 'no' if documents are completely irrelevant AND the question clearly requires specific factual information

Give a binary score 'yes' or 'no'."""

# prompts.py
SYSTEM_MESSAGE = """You are super smart chatbot named WDM-AI-TEMIS, an AI assistant with access to a knowledge base through the retriever_tool.

For EVERY user question, you should:
1. ALWAYS use the retriever_tool first to search the knowledge base
2. After getting search results, provide a comprehensive answer combining:
   - Information from the retrieved documents 
   - Your own knowledge when relevant
3. If no relevant information is found in the search results, you can fall back to your general knowledge
4. Be direct and helpful in your responses

IMPORTANT: Always search first, then answer. This ensures you provide the most up-to-date and relevant information from the knowledge base.

The vector store contains data from the following sources:
{source_list}
"""

REWRITE_PROMPT = """You are an AI assistant helping to improve search queries.
Original query: {query}

Rewrite this query to:
1. Be more specific and detailed
2. Include key terms that might appear in relevant documents
3. Focus on the core information need

Provide:
1. The rewritten query
2. A brief explanation of how you improved it"""

GENERATE_PROMPT = """You are a helpful assistant answering the user's most recent question based on the provided context.

Context information is provided in XML format with multiple documents:
{context}

FOCUS ON ANSWERING THIS SPECIFIC QUESTION: {question}

Instructions:
- The context is structured as XML with <documents> containing multiple <document> elements
- Each document has <metadata> (with source, page, type) and <content> sections
- If the context contains relevant information: provide a COMPLETE answer using ALL relevant information from ALL documents
- If the context contains tables or lists, include ALL items, don't summarize or truncate
- When referencing information, you can mention the source document and page number from metadata
- Format your response clearly with bullet points or numbered lists when appropriate
- If the question is a simple greeting, personal question, or general conversation, you can answer directly using your knowledge
- If the context doesn't contain relevant information for factual questions, say so clearly and provide what you can from your general knowledge
- Be natural and conversational while being informative and comprehensive
- Focus on answering the user's question rather than describing the document structure"""


QUERY_ANALYSIS_PROMPT = """
You are an expert document analyst. Your task is to analyze the user query and determine which document sources and types are most relevant.

Guidelines:
- Only suggest sources and types that actually exist in the document collection
- Be specific and relevant to the query content
- Provide reasoning for your choices
- If uncertain, indicate lower confidence score
- DO NOT return any page numbers - focus only on sources and types
- For sources, you can suggest partial filenames (without extensions) if the user refers to them that way

User Query: {query}
{context_info}

{format_instructions}

Important: Return only valid JSON format as specified above. Do not include pages in your response.
"""

GRAPH_SYSTEM_LLM_PARSER = """Your task is to act as an expert information extractor. From the provided INPUT_TEXT, you will extract a knowledge graph.

The output must be a JSON object with a single key "graph", which contains a list of structured objects. Each object represents a relationship triplet and must have the following keys: 'h', 'type_h', 'r', 'o', 'type_t'.

GUIDELINES:
1. 'h' (head) and 'o' (tail) are the entities.
2. 'type_h' and 'type_t' are the general categories. You must infer these types. Types should be concise, capitalized, singular nouns (e.g., PERSON, COMPANY, VEHICLE, LOCATION, PRODUCT).
3. **Crucially, identify abstract concepts like EVENTS (e.g., 'Battle of New York', 'Ultron's Attack') and PROTOCOLS (e.g., 'Sokovia Accords').**
4. 'r' (relationship) is a short, active verb.
  - For actions between entities, use verbs like: Drove, Invented, Created, Wields, Led, Defeated.
  - **For cause-and-effect, use verbs like: Caused, LedTo, ResultedIn.**
  - **For participation, use: ParticipatedIn.**
5. **Entity Disambiguation**: Consolidate different names for the same entity.
6. **Simplicity**: Keep entity names short and specific.

EXAMPLE 1 (Business):
- Input: 'The 2008 financial crisis led to the creation of the Dodd-Frank Act.'
- Output:
{{
  "graph": [
    {{ "h": "2008 Financial Crisis", "type_h": "EVENT", "r": "LedTo", "o": "Dodd-Frank Act", "type_t": "PROTOCOL" }}
  ]
}}

EXAMPLE 2 (MCU - a more relevant example for you):
- Input: 'The Battle of New York was a major conflict where the Avengers first assembled to fight Loki.'
- Output:
{{
  "graph": [
      {{ "h": "Avengers", "type_h": "GROUP", "r": "ParticipatedIn", "o": "Battle of New York", "type_t": "EVENT" }},
      {{ "h": "Loki", "type_h": "PERSON", "r": "ParticipatedIn", "o": "Battle of New York", "type_t": "EVENT" }}
  ]
}}

Your output MUST be a valid JSON object. Do not add any text before or after the JSON.

{format_instructions}

===========================================================
INPUT_TEXT:
{prompt_input}
"""

# ========================================
# RAG CONVERSATION PROMPTS
# ========================================

CONVERSATION_SUMMARY_PROMPT = """Hãy tóm tắt cuộc hội thoại sau một cách ngắn gọn và chính xác:

{conversation_history}

Yêu cầu tóm tắt:
- Chủ đề chính đã thảo luận
- Thông tin quan trọng người dùng đã cung cấp  
- Các quyết định hoặc kết luận quan trọng
- Context cần thiết cho câu hỏi tiếp theo
- Giữ lại tên và thông tin cá nhân người dùng

Tóm tắt (tối đa 200 từ):"""

RAG_QUERY_ANALYSIS_PROMPT = """Bạn là một AI chuyên phân tích câu hỏi để xác định nguồn tài liệu và loại nội dung phù hợp.

Câu hỏi: {query}

Thông tin có sẵn: {context_info}

Hãy phân tích câu hỏi và xác định:
1. SOURCES: Nguồn tài liệu nào cần tìm kiếm (tên file, tài liệu cụ thể)
2. TYPES: Loại nội dung nào phù hợp:
   - "text": Văn bản thường
   - "table": Bảng biểu, dữ liệu số
   - "image": Hình ảnh, biểu đồ, sơ đồ

Lưu ý:
- Nếu câu hỏi về biểu đồ, sơ đồ, hình ảnh → chọn "image"
- Nếu câu hỏi về số liệu, bảng biểu → chọn "table"
- Nếu câu hỏi về văn bản thường → chọn "text"
- Có thể chọn nhiều loại nếu cần thiết

{format_instructions}"""

RAG_RESPONSE_WITH_CONVERSATION_PROMPT = """Bạn là WDM-AI-TEMIS, trợ lý AI thông minh chuyên phân tích tài liệu và hỗ trợ người dùng.

LỊCH SỬ HỘI THOẠI:
{conversation_context}

NỘI DUNG TÀI LIỆU:
{context}

THÔNG TIN HÌNH ẢNH (nếu có):
{image_context}

CÂU HỎI HIỆN TẠI: {question}

Hướng dẫn trả lời:
- Nếu câu hỏi về thông tin cá nhân hoặc cuộc hội thoại trước: sử dụng lịch sử hội thoại
- Nếu câu hỏi về tài liệu: sử dụng nội dung tài liệu  
- Nếu câu hỏi về hình ảnh, biểu đồ, sơ đồ: sử dụng thông tin hình ảnh
- Trả lời tự nhiên, thân thiện bằng tiếng Việt
- Tham khảo cuộc hội thoại trước khi cần thiết
- Khi nói về hình ảnh, hãy mô tả chi tiết và liên kết với nội dung tài liệu
- Chỉ nói không biết khi cả lịch sử hội thoại và tài liệu đều không có thông tin

Trả lời:"""

RAG_RESPONSE_SIMPLE_PROMPT = """Bạn là WDM-AI-TEMIS, trợ lý AI thông minh chuyên phân tích tài liệu và hỗ trợ người dùng.

NỘI DUNG TÀI LIỆU:
{context}

THÔNG TIN HÌNH ẢNH (nếu có):
{image_context}

CÂU HỎI: {question}

Hướng dẫn trả lời:
- Sử dụng nội dung tài liệu để trả lời câu hỏi
- Nếu câu hỏi về hình ảnh, biểu đồ, sơ đồ: sử dụng thông tin hình ảnh
- Trả lời tự nhiên, thân thiện bằng tiếng Việt
- Khi nói về hình ảnh, hãy mô tả chi tiết và liên kết với nội dung tài liệu
- Chỉ nói không biết khi tài liệu không có thông tin liên quan

Trả lời:"""

# ========================================
# IMAGE PROCESSING PROMPTS
# ========================================

IMAGE_SUMMARY_PROMPT = """
This image contains a data table or a keyboard shortcut matrix. Please analyze and describe it thoroughly based on the following instructions:
1. Summarize the table's structure, content, and headers.
2. Identify repeated patterns, data types, or hierarchical categories.
3. Highlight any special formatting, such as merged cells, bold/italicized text, or color coding.
4. Describe whether the table is horizontal, vertical, or matrix-like.
5. Mention any missing values, inconsistencies, or notes.
6. Are ther any merge collumns or rows in the table, describe it carefully?
Your response should be detailed and help reconstruct the table's structure later.
"""

IMAGE_GENERAL_ANALYSIS_PROMPT = """
Analyze this image and provide a detailed description including:
1. Main content and subject matter
2. Visual elements (charts, diagrams, photos, text, etc.)
3. Colors, layout, and formatting
4. Any text content visible in the image
5. Context and purpose of the image
6. Technical details if applicable (graphs, data visualization, etc.)

Your response should be comprehensive and help users understand what the image contains and its relevance to their documents.
"""

TABLE_CONTEXT_ENRICHMENT_PROMPT = """
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

Task: Based on the three sources of information, provide a comprehensive analysis of the table data and return the correct table structure with all merged cell information properly filled."""

# ========================================
# DOCUMENT STRUCTURE ANALYSIS PROMPTS
# ========================================

DOCUMENT_SECTION_ANALYSIS_PROMPT = """You are an expert in document structure analysis. Your task is to examine text segments that appear 
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

TABLE_HEADER_ANALYSIS_PROMPT = """You are an expert in analyzing table data structures. Your task is to examine tables and determine if their first row contains meaningful column headers.

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
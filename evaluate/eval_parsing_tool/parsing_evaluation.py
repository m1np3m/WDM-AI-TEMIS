import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Lỗi: OPENAI_API_KEY không được tìm thấy.")
else:
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0,
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )
    print("DEBUG: LLM initialized with JSON mode.")
    


def evaluate_table_extraction(ground_truth: str, predict: str, llm_model = None, debug: bool = False) -> any: # Thay đổi kiểu trả về nếu cần
   parser = JsonOutputParser()

   prompt_template_str = """
   Evaluate the accuracy of a table extraction tool by comparing two markdown tables.

   Consider the following criteria:
   - **Content**: Ensure matches and events in GROUND TRUTH are accurately represented in PREDICT.
   - **Formatting**: Check formatting consistency, including spacing and alignment.
   - **Data Integrity**: Verify accurate data representation of dates, teams, scores, match titles, and attendance.
   - **Order**: Confirm event sequence consistency.

   # Steps

   1. **Content Comparison**: Match rows and confirm data accuracy (team names, scores, etc.).
   2. **Formatting Review**: Check column alignment and use of whitespace.
   3. **Data Integrity Check**: Identify missing data or inconsistencies.
   4. **Order Verification**: Ensure consistent event sequences.
   5. **Assessment**: Note discrepancies and provide a score (0-1).

   # Output Format

   You must respond with a JSON object containing only a score field with a float value between 0 and 1.
   For example:
   {{"score": 0.85}}

   ### GROUND TRUTH
   {ground_truth}

   ### PREDICT
   {predict}

   Remember: Your entire response must be a valid JSON object with only a "score" field containing a number between 0 and 1.
   """
   prompt = PromptTemplate.from_template(template=prompt_template_str)

   try:
      chain = prompt | llm_model | parser
      if debug:
         print("DEBUG: Chain created successfully.")

      invocation_payload = {
         "ground_truth": ground_truth,
         "predict": predict
      }
      if debug:
         print(f"DEBUG: Invoking chain with payload (first 100 chars of each): \nGround truth: {ground_truth[:100]}...\nPredict: {predict[:100]}...")

      # Để kiểm tra riêng lẻ LLM (bỏ qua parser tạm thời):
      # formatted_prompt = prompt.invoke(invocation_payload)
      # print(f"DEBUG: Formatted prompt sent to LLM:\n{formatted_prompt.to_string()}") # Hoặc .text tùy phiên bản
      # llm_output = llm_model.invoke(formatted_prompt)
      # print(f"DEBUG: Raw output from LLM:\n{llm_output}")
      # res = parser.parse(llm_output) # Parse thủ công nếu muốn kiểm tra parser

      res = chain.invoke(invocation_payload)
      if debug:
         print(f"DEBUG: Chain invocation successful. Result (res): {res}")
         print(f"DEBUG: Type of res: {type(res)}")
      return res

   except Exception as e:
      if debug:   
         print(f"ERROR in evaluate_table_extraction: {e}")
         import traceback
         traceback.print_exc() # In ra chi tiết lỗi và dòng gây lỗi
      return "" # Hoặc None, hoặc một giá trị báo lỗi cụ thể
    
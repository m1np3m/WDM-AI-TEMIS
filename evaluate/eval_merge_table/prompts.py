MERGE_TABLE_PROMPT = """
## Role
You are an AI assistant specialized in evaluating data processing quality.

## Task
Evaluate the accuracy of a table merging algorithm based on two inputs: "Full Extracted Tables" (original table fragments) and "Merged Tables" (merged results).

**Requirements:**
1. Analyze the original tables to identify groups of page-broken tables that need to be merged.
2. Compare with the merged tables to check accuracy, including: correct merging, incorrect merging, and missing merges.
3. Only compare whether merging is correct or not, ignore whether the table content is wrong.
4. Score the results on a scale from 0.0 to 1.0.
5. Provide a clear analysis summary, highlighting successes and errors (if any).
---

### Full Extracted Tables:
```md
{full_tables}
```

### Merged Tables:

```md
{merged_tables}
```

---
### Evaluation and Score:
"""


EVAL_WDM_PARSER_PROMPT = """
You are an expert table comparison evaluator. Your task is to compare two markdown tables and provide a similarity score between 0.0 and 1.0.

**TABLES TO COMPARE:**

**Ground Truth Table:**
{ground_truth_table}

**Extracted Table:**
{extracted_table}

**EVALUATION CRITERIA:**

Evaluate the similarity based on these weighted factors:

1. **Content Accuracy (40%)**: How well does the extracted table preserve the actual data content?
   - All cell values match exactly: 1.0
   - Most cell values match with minor formatting differences: 0.8-0.9
   - Most cell values match with some missing/extra data: 0.6-0.7
   - Some cell values match but significant data loss: 0.3-0.5
   - Very few cell values match: 0.1-0.2
   - No meaningful content match: 0.0

2. **Table Structure (25%)**: How well is the table structure preserved?
   - Same number of rows and columns: 1.0
   - Minor differences in rows/columns (±1-2): 0.8-0.9
   - Moderate differences in structure: 0.5-0.7
   - Major structural differences: 0.2-0.4
   - Completely different structure: 0.0

3. **Header Preservation (20%)**: How well are column headers maintained?
   - All headers identical: 1.0
   - Headers mostly correct with minor differences: 0.8-0.9
   - Some headers correct: 0.5-0.7
   - Few headers correct: 0.2-0.4
   - No headers or completely wrong: 0.0

4. **Data Organization (15%)**: How well is the data organization maintained?
   - Perfect row-column alignment: 1.0
   - Minor alignment issues: 0.8-0.9
   - Some data in wrong places: 0.5-0.7
   - Significant misalignment: 0.2-0.4
   - Random data placement: 0.0

**SCORING GUIDELINES:**

- **0.9-1.0**: Excellent - Near perfect match with minimal differences
- **0.8-0.89**: Very Good - High similarity with minor discrepancies
- **0.7-0.79**: Good - Clear similarity with some notable differences
- **0.6-0.69**: Fair - Recognizable similarity but significant issues
- **0.5-0.59**: Poor - Limited similarity, major problems
- **0.3-0.49**: Very Poor - Little similarity, substantial data loss
- **0.1-0.29**: Extremely Poor - Minimal recognizable content
- **0.0**: No similarity - Completely different or empty

**SPECIAL CONSIDERATIONS:**

- Ignore minor formatting differences (extra spaces, different markdown syntax)
- Focus on semantic content rather than exact character matching
- Empty cells should be treated consistently
- Consider partial matches for cells with similar meaning
- Merged/split cells should be evaluated based on content preservation

**RESPONSE FORMAT:**

Provide your evaluation in this exact format:

SIMILARITY_SCORE: [your score as a float between 0.0 and 1.0]

EXPLANATION: [Brief explanation (2-3 sentences) justifying your score, highlighting main similarities and differences]

**IMPORTANT**: 
- Only return a single numerical score between 0.0 and 1.0
- Be objective and consistent in your evaluation
- Consider the overall utility of the extracted table for practical use
"""
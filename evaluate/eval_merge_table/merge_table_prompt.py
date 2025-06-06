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

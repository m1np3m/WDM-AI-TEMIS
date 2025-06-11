import json
import os
import re
from typing import List
from utils import find_most_similar_table_ensemble, evaluate_table_similarity
from dotenv import load_dotenv
from tqdm import tqdm
# sleep
import time

load_dotenv()


def extract_markdown_tables(file_path: str) -> list[str]:
    """
    Extract all markdown tables from a markdown file and return them as strings.

    Args:
        file_path (str): Path to the markdown file

    Returns:
        list[str]: List of markdown tables as strings
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        lines = content.split("\n")
        tables = []
        i = 0

        while i < len(lines):
            # Look for potential table start (line with pipes)
            if "|" in lines[i].strip() and lines[i].strip():
                table_lines = []

                # Collect consecutive lines that look like table rows
                while i < len(lines) and is_table_line(lines[i]):
                    table_lines.append(lines[i])
                    i += 1

                # Check if we have a valid table (at least 2 lines)
                if len(table_lines) >= 2:
                    table_string = "\n".join(table_lines)
                    tables.append(table_string)

                continue

            i += 1

        return tables

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []


def is_table_line(line: str) -> bool:
    """
    Check if a line looks like part of a markdown table.

    Args:
        line (str): Line to check

    Returns:
        bool: True if line appears to be part of a table
    """
    stripped = line.strip()
    if not stripped:
        return False

    # Must contain at least one pipe
    if "|" not in stripped:
        return False

    return True


def get_tables_from_json_source(json_data, source: str) -> List[str]:
    source_json_data = None
    for data in json_data:
        if data["source"] == source:
            source_json_data = data
            break

    if source_json_data is None:
        return []

    tables = []
    for table in source_json_data["tables"]:
        tables.append(table["text"])

    return tables


# Example usage
if __name__ == "__main__":
    md_path = "/home/thangquang/code/WDM-AI-TEMIS/data/data-finetune/final_data/md/"
    md_files = os.listdir(md_path)
    md_files = [file for file in md_files if file.endswith(".md")]
    
    json_data = json.loads(
        open(
            "/home/thangquang/code/WDM-AI-TEMIS/data/full_content_chatdoc_eval.json",
            "r",
            encoding="utf-8",
        ).read()
    )
    
    # Store results for each file
    file_results = {}
    overall_total_score = 0
    overall_total_success = 0
    overall_score_list = []
    
    print(f"Starting evaluation of {len(md_files)} markdown files...")
    print("=" * 80)
    
    for i, md_file in enumerate(tqdm(md_files, desc="Processing files")):
        print(f"\n[{i+1}/{len(md_files)}] Evaluating: {md_file}")
        
        # Extract tables from current file
        ground_truth_tables = extract_markdown_tables(md_path + md_file)
        wdm_tables = get_tables_from_json_source(json_data, md_file.replace(".md", ".pdf"))
        
        # Initialize file-specific metrics
        file_score = 0
        file_success = 0
        file_score_list = []
        file_total_wdm_tables = len(wdm_tables)
        
        print(f"  Ground truth tables: {len(ground_truth_tables)}")
        print(f"  WDM extracted tables: {file_total_wdm_tables}")
        
        if file_total_wdm_tables == 0:
            print(f"  ⚠️  No WDM tables found for {md_file}")
            file_results[md_file] = {
                "ground_truth_count": len(ground_truth_tables),
                "wdm_tables_count": 0,
                "matched_tables": 0,
                "total_score": 0,
                "average_score": 0,
                "scores": [],
                "status": "no_wdm_tables"
            }
            continue
        
        # Evaluate each WDM table against ground truth
        for j, wdm_table in enumerate(wdm_tables):
            print(f"  Evaluating table {j+1}/{file_total_wdm_tables}...", end=" ")
            
            # Find most similar ground truth table
            idx = find_most_similar_table_ensemble(
                wdm_table, ground_truth_tables, min_similarity_threshold=0.4, verbose=False
            )
            
            if idx != -1:
                ground_truth_table = ground_truth_tables[idx]
                # Get LLM evaluation score
                score = evaluate_table_similarity(ground_truth_table, wdm_table)
                
                file_score += score
                file_score_list.append(score)
                file_success += 1
                
                overall_total_score += score
                overall_score_list.append(score)
                overall_total_success += 1
                
                print(f"✅ Score: {score:.3f}")
                time.sleep(0.5)  # Rate limiting
            else:
                print("❌ No match found")
        
        # Calculate file metrics
        file_average_score = file_score / file_success if file_success > 0 else 0
        file_match_rate = (file_success / file_total_wdm_tables) * 100
        
        # Store file results
        file_results[md_file] = {
            "ground_truth_count": len(ground_truth_tables),
            "wdm_tables_count": file_total_wdm_tables,
            "matched_tables": file_success,
            "total_score": file_score,
            "average_score": file_average_score,
            "match_rate": file_match_rate,
            "scores": file_score_list,
            "status": "completed"
        }
        
        # Print file summary
        print(f"  📊 File Summary:")
        print(f"     - Matched tables: {file_success}/{file_total_wdm_tables} ({file_match_rate:.1f}%)")
        print(f"     - Average score: {file_average_score:.3f}")
        print(f"     - Total score: {file_score:.3f}")
        print(f"     - Individual scores: {[f'{s:.3f}' for s in file_score_list]}")

    # Print overall results
    print("\n" + "=" * 80)
    print("OVERALL EVALUATION RESULTS")
    print("=" * 80)
    
    overall_average = overall_total_score / overall_total_success if overall_total_success > 0 else 0
    
    print(f"Total files processed: {len([f for f in file_results.values() if f['status'] == 'completed'])}")
    print(f"Total successful matches: {overall_total_success}")
    print(f"Overall average score: {overall_average:.3f}")
    print(f"Total score: {overall_total_score:.3f}")
    
    # Print per-file results table
    print("\n📋 PER-FILE DETAILED RESULTS:")
    print("-" * 120)
    print(f"{'File':<40} {'GT':<4} {'WDM':<4} {'Match':<6} {'Rate%':<6} {'Avg':<6} {'Total':<6} {'Scores'}")
    print("-" * 120)
    
    for file_name, results in file_results.items():
        if results['status'] == 'completed':
            scores_str = ', '.join([f'{s:.2f}' for s in results['scores'][:3]])  # Show first 3 scores
            if len(results['scores']) > 3:
                scores_str += f" ... (+{len(results['scores'])-3} more)"
            
            print(f"{file_name:<40} {results['ground_truth_count']:<4} {results['wdm_tables_count']:<4} "
                  f"{results['matched_tables']:<6} {results['match_rate']:<6.1f} "
                  f"{results['average_score']:<6.3f} {results['total_score']:<6.3f} {scores_str}")
        else:
            print(f"{file_name:<40} {results['ground_truth_count']:<4} {results['wdm_tables_count']:<4} "
                  f"{'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<6} No WDM tables")

    # Save results to JSON file
    output_file = "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "overall_metrics": {
                "total_files": len(md_files),
                "completed_files": len([f for f in file_results.values() if f['status'] == 'completed']),
                "total_successful_matches": overall_total_success,
                "overall_average_score": overall_average,
                "total_score": overall_total_score,
                "all_scores": overall_score_list
            },
            "file_results": file_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print(f"📊 Score distribution: min={min(overall_score_list):.3f}, max={max(overall_score_list):.3f}, "
          f"median={sorted(overall_score_list)[len(overall_score_list)//2]:.3f}")

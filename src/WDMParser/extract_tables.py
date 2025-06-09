import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2 import service_account
from loguru import logger
from markdown import markdown
from tqdm import tqdm

load_dotenv()

from .enrich import Enrich_VertexAI
from .credential_helper import validate_credentials_path, print_credentials_help, setup_default_credentials
from .llm_feat import get_is_new_section_context, get_is_has_header
from .utils_retry import retry_network_call

IMAGE_OUTPUT_DIR = "C:/Users/Admin/Data/WDM-AI-TEMIS/test_images"


class WDMTable(TypedDict):
    text: str
    page: int
    source: str
    n_rows: int
    n_columns: int
    bbox: Tuple[float, float, float, float]
    context_before: str
    is_new_section_context: bool
    is_has_header: bool
    image_path: str


class WDMMergedTable(TypedDict):
    text: str
    page: List[int]
    source: str
    bbox: List[Tuple[float, float, float, float]]
    headers: List[str]  # Textual headers
    n_rows: int  # Sum of n_rows from original tables in the group
    n_columns: int  # Max n_columns from original tables, text padded to this
    context_before: str
    image_paths: List[str]




def get_pdf_name(source: str) -> str:
    """
    Lấy tên của file pdf từ đường dẫn

    Args:
        source (str): Tên đường dẫn đến file pdf

    Returns:
        str: Tên của file pdf
    """
    pdf_name = os.path.basename(source)
    file_name_part, file_extension_part = os.path.splitext(pdf_name)
    return file_name_part


def get_headers_from_markdown(markdown_text: str) -> List[str]:
    """
    Extracts headers from a markdown table.

    Args:
        markdown_text (str): The markdown text containing the table.

    Returns:
        List[str]: A list of headers extracted from the markdown table.
    """
    lines = markdown_text.split("\n")
    headers = []
    for line in lines:
        if line.startswith("|") and not line.startswith("|---"):
            headers = [header.strip() for header in line.split("|") if header.strip()]
            break
    return headers


def solve_non_header_table(
    df: pd.DataFrame, target_headers: List[str], log: bool = False
) -> pd.DataFrame:
    if not isinstance(target_headers, list):
        if log:
            logger.warning(
                "Warning: target_headers không phải là list. Trả về DataFrame gốc."
            )
        return df.copy()
    df_copy = df.copy()
    first_row_data_values: List[Any] = []
    for col_original_name in df_copy.columns:
        if isinstance(col_original_name, str) and col_original_name.startswith(
            "Unnamed:"
        ):
            first_row_data_values.append(np.nan)
        else:
            first_row_data_values.append(str(col_original_name))
    if len(first_row_data_values) != len(target_headers):
        if len(first_row_data_values) > len(target_headers):
            first_row_data_values = first_row_data_values[: len(target_headers)]
        else:
            first_row_data_values.extend(
                [np.nan] * (len(target_headers) - len(first_row_data_values))
            )
    new_first_row_df = pd.DataFrame([first_row_data_values], columns=target_headers)
    num_target_cols = len(target_headers)
    current_data_cols = df_copy.shape[1]
    if num_target_cols == 0:
        if current_data_cols > 0:
            df_copy = pd.DataFrame(index=df_copy.index)
    else:
        if current_data_cols < num_target_cols:
            for i in range(num_target_cols - current_data_cols):
                df_copy[f"__temp_added_col_{i}"] = np.nan
        elif current_data_cols > num_target_cols:
            df_copy = df_copy.iloc[:, :num_target_cols]
    df_copy.columns = target_headers
    result_df = pd.concat([new_first_row_df, df_copy], ignore_index=True)
    return result_df.reset_index(drop=True)


def get_context_before_table(
    doc: pymupdf.Document,
    table_page_num_0_indexed: int,
    table_bbox: Tuple[float, float, float, float],
    max_vertical_gap_on_same_page: float = 50.0,
    search_upward_pixels_on_same_page: float = 300.0,
    try_previous_page_if_no_context: bool = True,
    search_bottom_pixels_on_prev_page: float = 200.0,
    prev_page_all_table_bboxes: Optional[
        List[Tuple[float, float, float, float]]
    ] = None,
    log: bool = False,
) -> str:
    """
    Tìm văn bản ngữ cảnh đứng ngay trước một bảng đã cho.
    Ưu tiên tìm trên cùng trang (trong phạm vi chiều rộng của bảng).
    Nếu không tìm thấy, có thể thử tìm ở cuối trang trước đó.

    Args:
        doc (pymupdf.Document): Đối tượng tài liệu PyMuPDF đã được mở.
        table_page_num_0_indexed (int): Số trang (bắt đầu từ 0) của bảng mục tiêu.
        table_bbox (Tuple[float, float, float, float]): Bounding box (x0, y0, x1, y1) của bảng mục tiêu.
        max_vertical_gap_on_same_page (float): Khoảng cách dọc tối đa từ đáy văn bản đến đỉnh bảng trên cùng trang.
        search_upward_pixels_on_same_page (float): Khoảng cách (pixel) từ đỉnh bảng lên trên để xác định vùng tìm kiếm ban đầu trên cùng trang.
        try_previous_page_if_no_context (bool): Nếu True, sẽ tìm ở trang trước nếu không có ngữ cảnh trên cùng trang.
        search_bottom_pixels_on_prev_page (float): Xác định chiều cao vùng ở cuối trang trước sẽ được quét.
        prev_page_all_table_bboxes (Optional[List[Tuple]]): Danh sách các bbox của tất cả bảng trên trang trước.
                                                            Nếu được cung cấp, ngữ cảnh sẽ được tìm *sau* bảng cuối cùng đó.
        log (bool): Bật ghi log chi tiết.

    Returns:
        str: Chuỗi văn bản ngữ cảnh được nối lại, hoặc chuỗi rỗng nếu không tìm thấy.
    """
    if not doc or not (0 <= table_page_num_0_indexed < len(doc)):
        if log:
            logger.error(
                f"Tài liệu không hợp lệ hoặc số trang không hợp lệ: {table_page_num_0_indexed}."
            )
        return ""

    tx0, ty0, tx1, _ = table_bbox

    final_context_parts: List[str] = []

    # Tìm ngữ cảnh trên cùng trang với bảng
    try:
        page = doc.load_page(table_page_num_0_indexed)

        search_y_start_same_page = max(0.0, ty0 - search_upward_pixels_on_same_page)
        search_y_end_same_page = ty0 - 1e-4

        if search_y_end_same_page > search_y_start_same_page:
            clip_rect_same_page = pymupdf.Rect(
                tx0, search_y_start_same_page, tx1, search_y_end_same_page
            )

            blocks_on_same_page = page.get_text(
                "blocks", clip=clip_rect_same_page, sort=True
            )

            candidate_blocks_info_same_page: List[Dict[str, Any]] = []
            for (
                bx0_orig,
                by0_orig,
                bx1_orig,
                by1_orig,
                btext_clipped,
                _,
                btype,
            ) in blocks_on_same_page:
                if btype != 0:
                    continue
                btext_cleaned = btext_clipped.strip()
                if not btext_cleaned:
                    continue

                if (
                    by1_orig <= search_y_end_same_page
                    and (ty0 - by1_orig) <= max_vertical_gap_on_same_page
                ):
                    candidate_blocks_info_same_page.append(
                        {
                            "y0_orig": by0_orig,
                            "x0_orig": bx0_orig,
                            "text": btext_cleaned,
                        }
                    )

            candidate_blocks_info_same_page.sort(
                key=lambda b: (b["y0_orig"], b["x0_orig"])
            )

            current_page_context_list = [
                b["text"] for b in candidate_blocks_info_same_page
            ]
            if current_page_context_list:
                final_context_parts.append("\n".join(current_page_context_list))

            if log and current_page_context_list:
                logger.debug(
                    f"Ngữ cảnh tìm thấy trên cùng trang (P{table_page_num_0_indexed + 1}) "
                    f"trước bảng tại y={ty0:.2f}:\n'{final_context_parts[-1]}'"
                )
            elif log:
                logger.debug(
                    f"Không có ngữ cảnh phù hợp trên cùng trang (P{table_page_num_0_indexed + 1})."
                )

    except Exception as e_same_page:
        if log:
            logger.error(
                f"Lỗi khi xử lý cùng trang {table_page_num_0_indexed + 1} để tìm ngữ cảnh: {e_same_page}"
            )

    # Tìm ngữ cảnh ở cuối trang trước
    if (
        try_previous_page_if_no_context
        and not final_context_parts
        and table_page_num_0_indexed > 0
    ):
        prev_page_num = table_page_num_0_indexed - 1
        try:
            prev_page = doc.load_page(prev_page_num)
            if log:
                logger.debug(
                    f"Không có ngữ cảnh trên P{table_page_num_0_indexed + 1}. "
                    f"Đang thử tìm ở cuối trang trước (P{prev_page_num + 1})."
                )

            search_y_start_prev_page = (
                prev_page.rect.height - search_bottom_pixels_on_prev_page
            )

            if prev_page_all_table_bboxes:
                bottom_of_last_table_on_prev = 0.0
                for prev_table_bbox_on_this_prev_page in prev_page_all_table_bboxes:
                    bottom_of_last_table_on_prev = max(
                        bottom_of_last_table_on_prev,
                        prev_table_bbox_on_this_prev_page[3],
                    )

                search_y_start_prev_page = max(
                    search_y_start_prev_page, bottom_of_last_table_on_prev + 1e-4
                )
                if log:
                    logger.debug(
                        f"Trang trước: tìm kiếm văn bản sau y={search_y_start_prev_page:.2f} (sau bảng cuối cùng / vùng cuối trang)."
                    )

            search_y_end_prev_page = prev_page.rect.height

            if search_y_end_prev_page > search_y_start_prev_page:
                clip_rect_prev_page = pymupdf.Rect(
                    tx0, search_y_start_prev_page, tx1, search_y_end_prev_page
                )

                blocks_on_prev_page = prev_page.get_text(
                    "blocks", clip=clip_rect_prev_page, sort=True
                )

                candidate_blocks_info_prev_page: List[Dict[str, Any]] = []
                for (
                    bx0_orig,
                    by0_orig,
                    bx1_orig,
                    by1_orig,
                    btext_clipped,
                    _,
                    btype,
                ) in blocks_on_prev_page:
                    if btype != 0:
                        continue
                    btext_cleaned = btext_clipped.strip()
                    if not btext_cleaned:
                        continue

                    if by0_orig >= search_y_start_prev_page:
                        candidate_blocks_info_prev_page.append(
                            {
                                "y0_orig": by0_orig,
                                "x0_orig": bx0_orig,
                                "text": btext_cleaned,
                            }
                        )

                candidate_blocks_info_prev_page.sort(
                    key=lambda b: (b["y0_orig"], b["x0_orig"])
                )

                prev_page_context_list = [
                    b["text"] for b in candidate_blocks_info_prev_page
                ]
                if prev_page_context_list:
                    final_context_parts = prev_page_context_list
                    if log:
                        context_text = "\n".join(prev_page_context_list)
                        logger.debug(
                            f"Ngữ cảnh tìm thấy ở cuối trang trước (P{prev_page_num + 1}):\n'{context_text}'"
                        )
                elif log:
                    logger.debug(
                        f"Không có ngữ cảnh phù hợp ở cuối trang trước (P{prev_page_num + 1})."
                    )

        except Exception as e_prev_page:
            if log:
                logger.error(
                    f"Lỗi khi xử lý trang trước {prev_page_num + 1} để tìm ngữ cảnh: {e_prev_page}"
                )

    final_result_text = ""
    if final_context_parts:
        final_result_text = final_context_parts[0]

    if log and not final_result_text:
        logger.info(
            f"Không tìm thấy ngữ cảnh nào cho bảng tại trang {table_page_num_0_indexed + 1} (bbox: {table_bbox})."
        )
    elif log and final_result_text:
        logger.info(
            f"Ngữ cảnh cuối cùng cho bảng tại trang {table_page_num_0_indexed + 1}:\n-----\n{final_result_text}\n-----"
        )

    return final_result_text.strip()


def convert_markdown_to_df(markdown_text: str) -> pd.DataFrame:
    try:
        html_table = markdown(markdown_text, extensions=["markdown.extensions.tables"])
        dfs = pd.read_html(StringIO(f"<table>{html_table}</table>"))
        if dfs:
            return dfs[0]
        else:
            # Return empty DataFrame if no tables found
            return pd.DataFrame()
    except Exception as e:
        # Return empty DataFrame if conversion fails
        return pd.DataFrame()


def process_single_page(
    page_info: Tuple[int, str, str], log: bool = False
) -> List[Dict]:
    """
    Process a single page of the PDF document to extract tables.

    Args:
        page_info: Tuple containing (page_idx, pdf_path, source)
        log (bool): If True, enables logging.

    Returns:
        List of table objects found on the page
    """
    try:
        page_idx, pdf_path, source = page_info
        # Open document in each process
        doc = pymupdf.open(pdf_path)
        page = doc.load_page(page_idx)
        tables = page.find_tables(strategy="lines_strict").tables
        page_tables = []

        if tables:
            for idx, table in enumerate(tables):
                bbox = table.bbox
                # lấy hình ảnh ra
                pix = page.get_pixmap(clip=bbox, dpi=200)
                # bắt đầu lưu hình ảnh
                output_filename = (
                    f"{IMAGE_OUTPUT_DIR}/{source}_{page_idx + 1}_table_{idx}.png"
                )
                pix.save(output_filename)
                n_rows = table.row_count
                n_columns = table.col_count
                text = table.to_markdown()

                table_obj = {
                    "text": text,
                    "page": page_idx + 1,
                    "source": source,
                    "n_rows": n_rows,
                    "n_columns": n_columns,
                    "bbox": bbox,
                    "context_before": "",  # Will be filled later
                    "is_new_section_context": False,
                    "image_path": output_filename,
                }
                page_tables.append(table_obj)

        # Close document in this process
        doc.close()
        return page_tables
    except Exception as e:
        if log:
            logger.error(f"Error processing page {page_idx + 1}: {str(e)}")
        return []


def get_n_rows_from_markdown(markdown_text: str, n_rows: int) -> str:
    """
    Extract the first n rows (including header) from a markdown table string.

    Args:
        markdown_text: The markdown table as a string.
        n_rows: Number of rows to extract (including header).

    Returns:
        A markdown string containing only the first n rows of the table,
        with all Col1, Col2, Col3, ... removed from the start of each cell.
    """
    lines = [line for line in markdown_text.strip().splitlines() if line.strip()]
    if not lines:
        return ""

    # Find the header and separator lines
    header_idx = None
    sep_idx = None
    for idx, line in enumerate(lines):
        if "|" in line:
            if header_idx is None:
                header_idx = idx
            elif sep_idx is None and set(line.replace("|", "").strip()) <= set("-: "):
                sep_idx = idx
                break

    if header_idx is None or sep_idx is None:
        # Not a valid markdown table
        return ""

    # The table starts at header_idx, separator at sep_idx
    table_lines = lines[header_idx:]
    # Always include header and separator
    result_lines = table_lines[:2]
    # Add up to n_rows-1 data rows (since header is already included)
    data_lines = table_lines[2 : 2 + max(0, n_rows - 1)]
    result_lines += data_lines

    # Remove Col1, Col2, Col3, ... from the start of each cell in every line
    cleaned_lines = []
    for line in result_lines:
        # Only process lines that look like table rows (contain at least one |)
        if "|" in line:
            # Remove Col\d+ at the start of each cell (after | or at start)
            # This regex replaces occurrences of Col\d+ at the start of a cell
            cleaned = re.sub(r"(\||^)\s*Col\d+\s*", r"\1", line)
            cleaned_lines.append(cleaned)
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def get_tables_from_pdf(
    doc: Union[str, pymupdf.Document],
    pages: List[int] = None,
    debug: bool = False,
    debug_level: int = 1,
    enrich: bool = False,
    use_ai_analysis: bool = True,
) -> List[WDMTable]:
    # Convert Document object to file path if needed
    if isinstance(doc, pymupdf.Document):
        pdf_path = doc.name
        doc.close()  # Close the original document
    else:
        pdf_path = doc
        doc = pymupdf.open(pdf_path)

    source = get_pdf_name(pdf_path)
    if pages is None:
        pages = list(range(1, doc.page_count + 1))

    # Close the document after getting page count
    doc.close()

    # Prepare page info for parallel processing
    page_infos = [(page - 1, pdf_path, source) for page in pages]

    # Determine number of processes (use 75% of available CPU cores)
    n_processes = max(1, int(cpu_count() * 0.75))
    if debug:
        logger.info(f"Using {n_processes} processes for parallel processing")

    # Process pages in parallel
    total_tables = []
    with Pool(processes=n_processes) as pool:
        # Use tqdm to show progress
        results = list(
            tqdm(
                pool.imap(process_single_page, page_infos),
                total=len(page_infos),
                desc="Processing pages",
            )
        )

    # Combine results from all pages
    for page_tables in results:
        total_tables.extend(page_tables)

    # Sort tables by page number and vertical position
    total_tables.sort(key=lambda t: (t["page"], t["bbox"][1]))

    # Reopen document for context processing
    doc = pymupdf.open(pdf_path)

    # Process contexts for all tables
    for i, table in enumerate(total_tables):
        target_table_page_0_indexed = table["page"] - 1
        actual_prev_page_0_indexed = target_table_page_0_indexed - 1
        filtered_prev_page_table_bboxes = []

        if actual_prev_page_0_indexed >= 0:
            for t_prev in total_tables:
                if t_prev["page"] - 1 == actual_prev_page_0_indexed:
                    filtered_prev_page_table_bboxes.append(t_prev["bbox"])

        context = get_context_before_table(
            doc=doc,
            table_page_num_0_indexed=target_table_page_0_indexed,
            table_bbox=table["bbox"],
            prev_page_all_table_bboxes=filtered_prev_page_table_bboxes,
        )
        total_tables[i]["context_before"] = context

    # Close document after context processing
    doc.close()

    # Process contexts for new section detection
    # Process headers (needed for both AI and non-AI analysis)
    headers = [get_headers_from_markdown(table["text"]) for table in total_tables]
    # Post process: remove Col1, Col2, Col3, etc.
    headers = [[re.sub(r"^Col\d+", "", col) for col in header] for header in headers]
    
    # Initialize debug variables
    prompt_contexts = ""
    prompt_headers = ""
    
    # AI-powered analysis (requires credentials)
    if use_ai_analysis:
        try:
            validate_credentials_path()
            
            contexts = [
                (i, table["context_before"])
                for i, table in enumerate(total_tables)
                if table["context_before"] != ""
            ]

            # Retry logic for get_is_new_section_context
            max_retries = 5
            for attempt in range(max_retries):
                res, prompt_contexts = get_is_new_section_context(
                    [context for _, context in contexts], return_prompt=True
                )
                if len(res["is_new_section_context"]) == len(contexts):
                    break
                if debug:
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for get_is_new_section_context due to length mismatch."
                    )
            else:
                if debug:
                    logger.error(
                        "Failed to get correct response length from get_is_new_section_context after retries."
                    )

            for (i, _), is_new in zip(contexts, res["is_new_section_context"]):
                total_tables[i]["is_new_section_context"] = is_new

            # Retry logic for get_is_has_header
            first_3_rows = [
                get_n_rows_from_markdown(table["text"], 3) for table in total_tables
            ]
            for attempt in range(max_retries):
                res, prompt_headers = get_is_has_header(
                    headers, first_3_rows, return_prompt=True
                )
                if len(res["is_has_header"]) == len(headers):
                    break
                if debug:
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for get_is_has_header due to length mismatch."
                    )
            else:
                if debug:
                    logger.error(
                        "Failed to get correct response length from get_is_has_header after retries."
                    )

            for table, is_has in zip(total_tables, res["is_has_header"]):
                table["is_has_header"] = is_has
                
        except (ValueError, FileNotFoundError) as e:
            if debug:
                logger.warning(f"AI analysis disabled due to credentials issue: {e}")
            # Set default values when AI analysis is not available
            for table in total_tables:
                table["is_new_section_context"] = False
                table["is_has_header"] = False
    else:
        # Set default values when AI analysis is disabled
        for table in total_tables:
            table["is_new_section_context"] = False
            table["is_has_header"] = False

    if debug:
        for idx, table in enumerate(total_tables):
            if debug_level >= 1:
                print(f"Table {idx + 1}:")
                print(f"    Page: {table['page']}")
                # print(f"Bbox: {table['bbox']}")
                print(f"    N_rows: {table['n_rows']}")
                print(f"    N_columns: {table['n_columns']}")
                print(f"    First row (can be header): {headers[idx]}")
                print(f"    Is has header: {table['is_has_header']}")
                print(f"    Is new section context: {table['is_new_section_context']}")
                print(f"    Context before: {table['context_before']}")
                # print(f"    Text: {table['text']}")
                print("-" * 100)
        if debug_level >= 2:
            print(
                f"CONTEXTS PROMPT:\n{prompt_contexts}\n\nHEADERS PROMPT:\n{prompt_headers}"
            )

    # Enrich tables if requested
    if enrich and total_tables:
        if debug:
            logger.info(f"Starting enrichment of {len(total_tables)} tables...")

        try:
            credentials_path = validate_credentials_path()
            
            start_time = time.time()
            if debug:
                logger.info(
                    f"Processing {len(total_tables)} table(s) for enrichment with parallel processing"
                )

            processor = Enrich_VertexAI(credentials_path=credentials_path)
            result_path = "vertex_chat_results.json"

            # Initialize enriched_markdowns list with None values to maintain order
            enriched_markdowns = [None] * len(total_tables)

            # Use ThreadPoolExecutor for parallel processing with max 2 workers to avoid rate limits
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(
                        enrich_single_table_markdown,
                        processor,
                        table,
                        result_path,
                        i,
                        debug,
                    ): i
                    for i, table in enumerate(total_tables)
                }

                # Process completed tasks with progress tracking
                completed_count = 0
                for future in as_completed(future_to_index):
                    try:
                        table_index, enriched_markdown = future.result()
                        enriched_markdowns[table_index] = enriched_markdown
                        completed_count += 1
                        if debug:
                            logger.info(
                                f"Completed {completed_count}/{len(total_tables)} table enrichments"
                            )
                    except Exception as exc:
                        table_index = future_to_index[future]
                        if debug:
                            logger.error(
                                f"Table {table_index + 1} generated an exception: {exc}"
                            )
                        # Use original markdown as fallback
                        enriched_markdowns[table_index] = total_tables[table_index][
                            "text"
                        ]

            if debug:
                elapsed_time = time.time() - start_time
                logger.info(
                    f"All table enrichments completed in {elapsed_time:.2f}s (avg: {elapsed_time / len(total_tables):.2f}s per table)"
                )

            # Ensure no None values in enriched_markdowns (fallback for any failed enrichments)
            for i, enriched_markdown in enumerate(enriched_markdowns):
                if enriched_markdown is None:
                    if debug:
                        logger.warning(
                            f"Table {i + 1} enrichment was None, using original markdown"
                        )
                    enriched_markdowns[i] = total_tables[i]["text"]

            # Update tables with enriched markdown
            for i, enriched_markdown in enumerate(enriched_markdowns):
                total_tables[i]["text"] = enriched_markdown

            if debug:
                logger.info("All tables updated with enriched content")

        except Exception as e:
            if debug:
                logger.error(
                    f"Error during enrichment: {e}, keeping original markdown"
                )

    return total_tables


def clean_markdown_fences(markdown_content: str) -> str:
    """
    Remove markdown code fences (``` or ```md) from the content if present.

    Args:
        markdown_content: The markdown table content that may contain code fences

    Returns:
        The cleaned markdown content with code fences removed
    """
    # Remove ```md or ``` or ```markdown from start
    markdown_content = re.sub(r"^```(?:md|markdown)?\n", "", markdown_content)

    # Remove ``` from end
    markdown_content = re.sub(r"\n```$", "", markdown_content)

    return markdown_content


def enrich_single_table(
    processor: Enrich_VertexAI,
    table: WDMTable,
    result_path: str,
    table_index: int,
    debug: bool = False,
) -> Tuple[int, pd.DataFrame]:
    """
    Enrich a single table using VertexAI processor.

    Args:
        processor: VertexAI processor instance
        table: Table object to enrich
        result_path: Path to save results
        table_index: Index of the table for logging
        debug: Enable debug logging

    Returns:
        Tuple of (table_index, enriched_dataframe)
    """

    start_time = time.time()
    try:
        if debug:
            logger.info(
                f"Processing table {table_index + 1} from image: {table['image_path']}"
            )

        enriched_markdown = processor.full_pipeline(
            file_path=table["image_path"],
            extract_table_markdown=table["text"],
            result_path=result_path,
            return_markdown=True,
        )
        cleaned_markdown = clean_markdown_fences(enriched_markdown)
        df = convert_markdown_to_df(cleaned_markdown)

        if debug:
            elapsed = time.time() - start_time
            logger.info(
                f"Table {table_index + 1} enriched successfully in {elapsed:.2f}s, shape: {df.shape}"
            )

        return table_index, df
    except Exception as e:
        if debug:
            elapsed = time.time() - start_time
            logger.error(
                f"Error enriching table {table_index + 1} after {elapsed:.2f}s: {e}"
            )
        # Return original markdown as fallback
        df = convert_markdown_to_df(table["text"])
        return table_index, df


def enrich_single_table_markdown(
    processor: Enrich_VertexAI,
    table: WDMTable,
    result_path: str,
    table_index: int,
    debug: bool = False,
) -> Tuple[int, str]:
    """
    Enrich a single table using VertexAI processor and return enriched markdown.

    Args:
        processor: VertexAI processor instance
        table: Table object to enrich
        result_path: Path to save results
        table_index: Index of the table for logging
        debug: Enable debug logging

    Returns:
        Tuple of (table_index, enriched_markdown)
    """

    start_time = time.time()
    try:
        if debug:
            logger.info(
                f"Enriching table {table_index + 1} from image: {table['image_path']}"
            )

        # Add a small delay before processing to help with rate limiting
        if table_index > 0:  # Don't delay the first table
            time.sleep(1)
            
        enriched_markdown = processor.full_pipeline(
            file_path=table["image_path"],
            extract_table_markdown=table["text"],
            result_path=result_path,
            return_markdown=True,
        )
        cleaned_markdown = clean_markdown_fences(enriched_markdown)

        if debug:
            elapsed = time.time() - start_time
            logger.info(
                f"Table {table_index + 1} enriched successfully in {elapsed:.2f}s"
            )

        return table_index, cleaned_markdown
    except Exception as e:
        if debug:
            elapsed = time.time() - start_time
            logger.error(
                f"Error enriching table {table_index + 1} after {elapsed:.2f}s: {e}"
            )
        # Return original markdown as fallback
        return table_index, table["text"]


def find_spanned_table_groups(tables: List[WDMTable]) -> List[List[WDMTable]]:
    """
    Tìm các nhóm bảng kéo dài qua nhiều trang (span multipage).

    Args:
        tables: Danh sách các bảng đã được sắp xếp theo thứ tự xuất hiện trong tài liệu

    Returns:
        Danh sách các nhóm bảng, mỗi nhóm là một list các bảng liên tục
    """
    if not tables:
        return []

    # Sắp xếp tables theo page và position nếu chưa được sắp xếp
    sorted_tables = sorted(
        tables, key=lambda t: (t["page"], t["bbox"][1])
    )  # sort by page and y-coordinate

    groups = []
    current_group = [sorted_tables[0]]

    for i in range(1, len(sorted_tables)):
        current_table = sorted_tables[i]
        prev_table = sorted_tables[i - 1]

        if _should_group_with_previous(current_table, prev_table):
            current_group.append(current_table)
        else:
            # Kết thúc group hiện tại và bắt đầu group mới
            groups.append(current_group)
            current_group = [current_table]

    # Thêm group cuối cùng
    groups.append(current_group)

    return groups


def _should_group_with_previous(current_table: WDMTable, prev_table: WDMTable) -> bool:
    """
    Quyết định liệu current_table có nên được nhóm với prev_table hay không.

    Logic dựa trên flowchart đã phân tích:
    1. Kiểm tra trang kế tiếp
    2. Kiểm tra context tiêu đề mới
    3. Kiểm tra header riêng
    4. Phân biệt dựa trên context_before
    """

    # Điều kiện tiên quyết: phải là trang kế tiếp
    if current_table["page"] != prev_table["page"] + 1:
        return False

    # Nếu context trước current_table là tiêu đề cho bảng/phần mới -> không group
    if current_table["is_new_section_context"]:
        return False

    # Nếu current_table có header riêng -> không group (thường là bảng mới)
    if current_table["is_has_header"]:
        return False

    # Nếu current_table không có header riêng -> có khả năng cao là span
    if not current_table["is_has_header"]:
        # Trường hợp 1: Span rõ ràng (context_before rỗng)
        if not current_table["context_before"].strip():
            return True

        # Trường hợp 2: Span với context "nhiễu"
        # (context_before không rỗng nhưng không phải tiêu đề mới)
        # Thêm kiểm tra tương thích cấu trúc cột để tăng độ tin cậy
        if _is_compatible_structure(current_table, prev_table):
            return True

    # Các trường hợp khác: không group
    return False


def _is_compatible_structure(current_table: WDMTable, prev_table: WDMTable) -> bool:
    """
    Kiểm tra tính tương thích về cấu trúc giữa hai bảng.
    Cho phép sai lệch 1-2 cột do lỗi extract tool.
    """

    # Cho phép sai lệch tối đa 2 cột
    col_diff = abs(current_table["n_columns"] - prev_table["n_columns"])

    # Nếu số cột giống nhau hoặc chênh lệch không quá 2 cột
    if col_diff <= 2:
        return True

    # Thêm các kiểm tra khác nếu cần:
    # - Kiểm tra độ rộng bảng (bbox width)
    current_width = current_table["bbox"][2] - current_table["bbox"][0]
    prev_width = prev_table["bbox"][2] - prev_table["bbox"][0]
    width_ratio = min(current_width, prev_width) / max(current_width, prev_width)

    # Nếu độ rộng tương tự (>= 80% overlap) thì vẫn có thể group
    if width_ratio >= 0.8:
        return True

    return False


def print_groups_summary(groups: List[List[WDMTable]], debug: bool = False) -> None:
    """In tóm tắt các nhóm bảng để debug"""
    if debug:
        logger.info(f"Tìm thấy {len(groups)} nhóm bảng:")

        for i, group in enumerate(groups, 1):
            if len(group) == 1:
                table = group[0]
                logger.info(
                    f"  Nhóm {i}: Bảng đơn (Trang {table['page']}, {table['n_columns']} cột)"
                )
            else:
                pages = [t["page"] for t in group]
                cols = [t["n_columns"] for t in group]
                logger.info(
                    f"  Nhóm {i}: Span {len(group)} bảng (Trang {min(pages)}-{max(pages)}, Cột: {cols})"
                )


def merge_tables(
    tables: List[WDMTable],
    debug: bool = False,
) -> WDMMergedTable:
    try:
        # Convert each table's markdown text to a DataFrame
        # Tables are now pre-enriched if enrich was requested in get_tables_from_pdf_2
        dfs = [convert_markdown_to_df(table["text"]) for table in tables]

        # Handle single-row tables that might have data as column names
        processed_dfs = []
        for i, df in enumerate(dfs):
            # Check if this DataFrame might have data as column names
            # This happens when a table has only 1 data row and pandas treats it as headers
            if len(df) <= 1:
                # Check if we have only one row and it's all NaN values
                if len(df) == 1 and df.iloc[0].isna().all():
                    # Get column names (which might be the actual data)
                    col_names = list(df.columns)

                    # Convert column names back to a data row
                    data_row = pd.DataFrame([col_names])
                    # Give it generic column names
                    data_row.columns = [f"Col_{j}" for j in range(len(col_names))]
                    processed_dfs.append(data_row)
                    if debug:
                        logger.debug(
                            f"Table {i}: Converted misinterpreted headers back to data row"
                        )
                else:
                    processed_dfs.append(df)
            else:
                processed_dfs.append(df)

        # Update dfs to use processed versions
        dfs = processed_dfs

        # Filter out empty DataFrames
        non_empty_dfs = [df for df in dfs if not df.empty and df.shape[1] > 0]

        if not non_empty_dfs:
            # If all DataFrames are empty, return a fallback merged table
            if debug:
                logger.warning("All DataFrames are empty, returning fallback table")
            first_table = tables[0]
            return WDMMergedTable(
                text=first_table["text"],
                page=[first_table["page"]],
                source=first_table["source"],
                bbox=[first_table["bbox"]],
                headers=[get_headers_from_markdown(first_table["text"])],
                n_rows=first_table["n_rows"],
                n_columns=first_table["n_columns"],
                context_before=first_table["context_before"],
                image_paths=[first_table["image_path"]],
            )

        # Determine the maximum number of columns across all DataFrames
        max_cols = max(df.shape[1] for df in non_empty_dfs)
        dfs = non_empty_dfs

        # Get headers from the first table to establish consistent column structure
        target_headers = get_headers_from_markdown(tables[0]["text"])

        # If target_headers length doesn't match max_cols, create a consistent column structure
        if len(target_headers) < max_cols:
            target_headers.extend(
                [f"Col_{i}" for i in range(len(target_headers), max_cols)]
            )
        elif len(target_headers) > max_cols:
            target_headers = target_headers[:max_cols]

        # Process DataFrames using the safer approach
        normalized_dfs = []

        # First DataFrame: keep as is (this is our standard with proper headers)
        first_df = dfs[0].copy()

        # Ensure first DataFrame has the right number of columns
        if first_df.shape[1] < max_cols:
            for j in range(first_df.shape[1], max_cols):
                first_df[f"temp_col_{j}"] = ""
        elif first_df.shape[1] > max_cols:
            first_df = first_df.iloc[:, :max_cols]

        # Set proper column names for first DataFrame
        first_df.columns = target_headers[: first_df.shape[1]]
        normalized_dfs.append(first_df)

        # Process remaining DataFrames: set numeric column names to avoid header confusion
        for i in range(1, len(dfs)):
            df_copy = dfs[i].copy()

            # Ensure it has the right number of columns first
            if df_copy.shape[1] < max_cols:
                for j in range(df_copy.shape[1], max_cols):
                    df_copy[j] = ""  # Use numeric column names
            elif df_copy.shape[1] > max_cols:
                df_copy = df_copy.iloc[:, :max_cols]

            # Set NUMERIC column names (0, 1, 2, 3...) to avoid header confusion
            df_copy.columns = list(range(df_copy.shape[1]))

            # Now safely rename to target headers
            df_copy.columns = target_headers[: df_copy.shape[1]]

            normalized_dfs.append(df_copy)

        # Now concatenate the normalized DataFrames
        merged_df = pd.concat(normalized_dfs, ignore_index=True)

        # Post-process: Remove any duplicate header rows (but be more careful)
        # Only remove header rows if we have enough data rows
        if len(merged_df) > 2:  # Only try to remove headers if we have more than 2 rows
            # Check if any row has the same values as the target headers
            header_mask = merged_df.apply(
                lambda row: all(
                    str(row.iloc[i]).strip().lower()
                    == str(target_headers[i]).strip().lower()
                    for i in range(min(len(row), len(target_headers)))
                    if target_headers[i].strip()
                ),
                axis=1,
            )

            # Remove header duplicate rows (but keep at least one row)
            if header_mask.sum() > 0 and len(merged_df) > header_mask.sum():
                # Keep only non-header rows
                merged_df = merged_df[~header_mask]
                # If we accidentally removed all rows, keep the original
                if len(merged_df) == 0:
                    merged_df = pd.concat(normalized_dfs, ignore_index=True)

        # Post-process the merged DataFrame
        # Convert all columns to string type before filling NaN
        merged_df = merged_df.astype(str)
        # Fill NaN values with empty strings
        merged_df = merged_df.replace("nan", "")
        # Remove columns with names starting with 'Col' and replace their values with empty strings
        merged_df.columns = [re.sub(r"^Col\d+", "", col) for col in merged_df.columns]
        merged_df = merged_df.replace(r"^Col\d+", "", regex=True)

        # Create a MergedTable with the combined data
        merged_table = WDMMergedTable(
            text=merged_df.to_markdown(index=False),
            page=[table["page"] for table in tables],
            source=tables[0]["source"],
            bbox=[table["bbox"] for table in tables],
            headers=[get_headers_from_markdown(table["text"]) for table in tables],
            n_rows=merged_df.shape[0],
            n_columns=merged_df.shape[1],
            context_before=tables[0]["context_before"],
            image_paths=[table["image_path"] for table in tables],
        )

        return merged_table

    except Exception as e:
        if debug:
            logger.error(f"Error merging tables: {str(e)}")
        # Return a fallback merged table with just the first table
        first_table = tables[0]
        return WDMMergedTable(
            text=first_table["text"],
            page=[first_table["page"]],
            source=first_table["source"],
            bbox=[first_table["bbox"]],
            headers=[get_headers_from_markdown(first_table["text"])],
            n_rows=first_table["n_rows"],
            n_columns=first_table["n_columns"],
            context_before=first_table["context_before"],
            image_paths=[first_table["image_path"]],
        )


def full_pipeline(
    doc: Union[List[str], List[pymupdf.Document]],
    pages: List[int] = None,
    debug: bool = False,
    debug_level: int = 1,
    return_full_tables: bool = False,
    evaluate: bool = False,
    enrich: bool = False,
) -> Union[List[WDMMergedTable], Tuple[List[WDMTable], List[WDMMergedTable]]]:
    merged_tables = []

    # Convert single string to list if needed
    if isinstance(doc, str):
        doc = [doc]

    for d in doc:
        try:
            if isinstance(d, str):
                # Validate file path
                if not os.path.exists(d):
                    if debug:
                        logger.error(f"File not found: {d}")
                    continue
                if debug:
                    logger.info(f"Processing document: {get_pdf_name(d)}")
            else:
                if debug:
                    logger.info(f"Processing document: {d.name}")

            if debug:
                logger.info("   Extracting tables...")
            # For full_pipeline (merge_span_tables=True), always use AI analysis
            tables = get_tables_from_pdf(d, pages, debug, debug_level, enrich, use_ai_analysis=True)
            if evaluate:
                # bỏ đi table cuối cùng
                tables = tables[:-1]
            if debug:
                logger.info("   Finding spanned table groups...")
            table_groups = find_spanned_table_groups(tables)
            print_groups_summary(table_groups, debug)
            if debug:
                logger.info("   Merging tables...")
            for group in table_groups:
                merged_table = merge_tables(group, debug)
                merged_tables.append(merged_table)
            if debug:
                logger.info("   Done!")
        except Exception as e:
            if debug:
                logger.error(f"Error processing document: {str(e)}")
            continue

    if return_full_tables:
        return tables, merged_tables
    return merged_tables

import os
import re
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import numpy as np

import pandas as pd
import pymupdf  # PyMuPDF
import pymupdf4llm
from loguru import logger
from markdown import markdown
from tqdm import tqdm

from utils import get_is_has_header, get_is_new_section_context


class Table(TypedDict):
    text: str
    page: int
    source: str
    n_rows: int
    n_columns: int
    bbox: Tuple[float, float, float, float]
    context_before: str
    is_new_section_context: bool
    is_has_header: bool


class MergedTable(TypedDict):
    text: str
    page: List[int]
    source: str
    bbox: List[Tuple[float, float, float, float]]
    headers: List[str]  # Textual headers
    n_rows: int  # Sum of n_rows from original tables in the group
    n_columns: int  # Max n_columns from original tables, text padded to this
    context_before: str


def get_pdf_name(source: str) -> str:
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


def solve_non_header_table(df: pd.DataFrame, target_headers: List[str]) -> pd.DataFrame:
    if not isinstance(target_headers, list):
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

def split_markdown_into_tables(markdown_text, debug=False):
    """
    Splits a single string containing multiple Markdown tables into a list of strings,
    where each string represents a separate table. Tables are assumed to be
    separated by two or more newline characters.

    Args:
        markdown_text (str): The input string potentially containing multiple
                             Markdown tables.
        debug (bool): If True, enables detailed debug logging during the process.
                      Defaults to False.

    Returns:
        list: A list of strings, where each string is a distinct Markdown table
              found in the input. Returns an empty list if the input is empty,
              contains only whitespace, or no valid table blocks are found.
    """
    # Initial check for empty or whitespace-only input.
    if not markdown_text or not markdown_text.strip():
        if debug:
            logger.debug(
                "Input markdown_text is empty or consists only of whitespace. Returning an empty list."
            )
        return []

    # Remove leading/trailing whitespace (including newlines) from the entire input text.
    # This helps in correctly splitting and avoids empty entries at the list ends if the input
    # has extraneous blank lines at the very beginning or end.
    stripped_text = markdown_text.strip()

    # If stripping results in an empty string (e.g., input was just "\n\n\n" or similar).
    if not stripped_text:
        if debug:
            logger.debug(
                "Input markdown_text became empty after stripping. Returning an empty list."
            )
        return []

    # Split the text into blocks using two or more newline characters as delimiters.
    # This pattern is the primary mechanism for separating distinct Markdown tables.
    potential_table_blocks = re.split(r"\n{2,}", stripped_text)

    extracted_tables = []
    if debug:
        logger.debug(
            f"Split input into {len(potential_table_blocks)} potential block(s). Now validating each block..."
        )

    for i, block in enumerate(potential_table_blocks):
        # Clean whitespace from each individual block before validation.
        # This ensures that checks like startswith('|') are accurate even if a block has padding.
        cleaned_block = block.strip()

        # A valid Markdown table block, when separated this way, must not be empty
        # and should start with the pipe character ('|').
        if cleaned_block and cleaned_block.startswith("|"):
            extracted_tables.append(cleaned_block)
            if debug:
                # Provide a preview of the extracted table for debugging purposes.
                # Replacing newlines with '↵' for a compact log preview.
                preview = cleaned_block[:70].replace("\n", "↵")
                if len(cleaned_block) > 70:
                    preview += "..."
                logger.debug(
                    f"Block {i + 1}: Valid table extracted (length: {len(cleaned_block)} chars). Preview: '{preview}'"
                )
        elif cleaned_block and debug:
            # Log if a non-empty block was discarded because it didn't meet the table criteria.
            preview = cleaned_block[:70].replace("\n", "↵")
            if len(cleaned_block) > 70:
                preview += "..."
            logger.warning(
                f"Block {i + 1}: Discarded (length: {len(cleaned_block)} chars) as it does not start with '|' "
                f"after stripping. Preview: '{preview}'"
            )
        elif not cleaned_block and debug:
            # Log if a block (after stripping) was empty.
            # This should be less common if `stripped_text` itself is not empty.
            logger.debug(
                f"Block {i + 1}: Discarded as it was empty after stripping individual block whitespace."
            )

    if debug:
        logger.info(f"Finished processing. Extracted {len(extracted_tables)} table(s).")

    return extracted_tables


def get_context_before_table(
    doc: pymupdf.Document,
    table_page_num_0_indexed: int,
    table_bbox: Tuple[float, float, float, float],
    # Thông số cho việc tìm kiếm trên cùng trang
    max_vertical_gap_on_same_page: float = 50.0,
    search_upward_pixels_on_same_page: float = 300.0,
    # Thông số cho việc tìm kiếm trên trang trước
    try_previous_page_if_no_context: bool = True,
    search_bottom_pixels_on_prev_page: float = 200.0,
    prev_page_all_table_bboxes: Optional[
        List[Tuple[float, float, float, float]]
    ] = None,
    debug: bool = False,
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
        debug (bool): Bật ghi log chi tiết.

    Returns:
        str: Chuỗi văn bản ngữ cảnh được nối lại, hoặc chuỗi rỗng nếu không tìm thấy.
    """
    if not doc or not (0 <= table_page_num_0_indexed < len(doc)):
        if debug:
            logger.error(
                f"Tài liệu không hợp lệ hoặc số trang không hợp lệ: {table_page_num_0_indexed}."
            )
        return ""

    tx0, ty0, tx1, _ = (
        table_bbox  # ty1 là table_bbox[3] - không dùng trực tiếp trong định nghĩa vùng tìm kiếm phía trên
    )

    # Danh sách để lưu trữ các phần văn bản ngữ cảnh (từ trang trước và trang hiện tại)
    final_context_parts: List[str] = []

    # --- Phần 1: Tìm ngữ cảnh trên CÙNG TRANG với bảng ---
    try:
        page = doc.load_page(table_page_num_0_indexed)

        # Xác định vùng tìm kiếm trên cùng trang:
        # Theo chiều dọc: từ một khoảng phía trên bảng (`search_upward_pixels_on_same_page`)
        #                đến ngay sát phía trên đỉnh của bảng (`ty0`).
        # Theo chiều ngang: **nghiêm ngặt trong giới hạn chiều rộng của bảng (tx0, tx1).**
        search_y_start_same_page = max(0.0, ty0 - search_upward_pixels_on_same_page)
        search_y_end_same_page = (
            ty0 - 1e-4
        )  # Một epsilon nhỏ để đảm bảo nằm hoàn toàn phía trên bảng

        if search_y_end_same_page > search_y_start_same_page:
            # Vùng clip cho PyMuPDF sẽ sử dụng trực tiếp tx0, tx1 của bảng cho giới hạn ngang.
            clip_rect_same_page = pymupdf.Rect(
                tx0, search_y_start_same_page, tx1, search_y_end_same_page
            )

            # page.get_text("blocks", clip=...) trả về các block có phần giao với clip_rect.
            # Quan trọng: nội dung `btext_clipped` đã được PyMuPDF cắt theo chiều ngang của `clip_rect_same_page`.
            # Tọa độ (bx0_orig,...) là của block gốc chưa bị cắt.
            # `sort=True` sắp xếp các block theo thứ tự đọc (y0, rồi x0).
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
                    continue  # Chỉ xử lý text blocks
                btext_cleaned = btext_clipped.strip()
                if not btext_cleaned:
                    continue

                # Lọc dựa trên khoảng cách dọc từ đáy block gốc (by1_orig) đến đỉnh bảng (ty0).
                # Block phải kết thúc trong vùng tìm kiếm đã clip (by1_orig <= search_y_end_same_page).
                # Và khoảng cách đến đỉnh bảng không quá lớn.
                if (
                    by1_orig <= search_y_end_same_page
                    and (ty0 - by1_orig) <= max_vertical_gap_on_same_page
                ):
                    candidate_blocks_info_same_page.append(
                        {
                            "y0_orig": by0_orig,  # Dùng y0_orig để sắp xếp theo thứ tự đọc
                            "x0_orig": bx0_orig,
                            "text": btext_cleaned,
                        }
                    )

            # Sắp xếp lại các block ứng viên theo thứ tự đọc (từ trên xuống, trái sang phải)
            # Mặc dù `sort=True` đã sắp xếp, việc sắp xếp lại sau khi lọc đảm bảo thứ tự cho tập con.
            candidate_blocks_info_same_page.sort(
                key=lambda b: (b["y0_orig"], b["x0_orig"])
            )

            current_page_context_list = [
                b["text"] for b in candidate_blocks_info_same_page
            ]
            if current_page_context_list:
                final_context_parts.append("\n".join(current_page_context_list))

            if debug and current_page_context_list:
                logger.debug(
                    f"Ngữ cảnh tìm thấy trên cùng trang (P{table_page_num_0_indexed + 1}) "
                    f"trước bảng tại y={ty0:.2f}:\n'{final_context_parts[-1]}'"
                )
            elif debug:
                logger.debug(
                    f"Không có ngữ cảnh phù hợp trên cùng trang (P{table_page_num_0_indexed + 1})."
                )

    except Exception as e_same_page:
        if debug:
            logger.error(
                f"Lỗi khi xử lý cùng trang {table_page_num_0_indexed + 1} để tìm ngữ cảnh: {e_same_page}"
            )

    # --- Phần 2: Tìm ngữ cảnh ở cuối TRANG TRƯỚC ---
    # Thực hiện nếu không có ngữ cảnh trên trang hiện tại, được phép thử, và không phải trang đầu tiên.
    if (
        try_previous_page_if_no_context
        and not final_context_parts
        and table_page_num_0_indexed > 0
    ):
        prev_page_num = table_page_num_0_indexed - 1
        try:
            prev_page = doc.load_page(prev_page_num)
            if debug:
                logger.debug(
                    f"Không có ngữ cảnh trên P{table_page_num_0_indexed + 1}. "
                    f"Đang thử tìm ở cuối trang trước (P{prev_page_num + 1})."
                )

            # Xác định vùng tìm kiếm Y trên trang trước:
            # Mặc định là một vùng ở cuối trang.
            search_y_start_prev_page = (
                prev_page.rect.height - search_bottom_pixels_on_prev_page
            )

            # Nếu biết bbox của các bảng trên trang trước, tìm văn bản *sau* bảng cuối cùng.
            if prev_page_all_table_bboxes:
                bottom_of_last_table_on_prev = 0.0
                for prev_table_bbox_on_this_prev_page in prev_page_all_table_bboxes:
                    # Giả định rằng prev_page_all_table_bboxes chỉ chứa các bảng thuộc prev_page_num
                    bottom_of_last_table_on_prev = max(
                        bottom_of_last_table_on_prev,
                        prev_table_bbox_on_this_prev_page[3],
                    )  # y1

                # Bắt đầu tìm kiếm sau bảng cuối cùng đó, nhưng không cao hơn vùng cuối trang đã xác định.
                search_y_start_prev_page = max(
                    search_y_start_prev_page, bottom_of_last_table_on_prev + 1e-4
                )
                if debug:
                    logger.debug(
                        f"Trang trước: tìm kiếm văn bản sau y={search_y_start_prev_page:.2f} (sau bảng cuối cùng / vùng cuối trang)."
                    )

            search_y_end_prev_page = prev_page.rect.height  # Tìm đến hết cuối trang

            if search_y_end_prev_page > search_y_start_prev_page:
                # Vùng tìm kiếm ngang trên trang trước cũng nên căn theo chiều rộng của bảng hiện tại.
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

                    # Chỉ lấy các block bắt đầu (by0_orig) trong vùng tìm kiếm xác định ở cuối trang.
                    # Điều này đảm bảo chúng ta lấy văn bản ở phần dưới cùng mong muốn.
                    if by0_orig >= search_y_start_prev_page:
                        candidate_blocks_info_prev_page.append(
                            {
                                "y0_orig": by0_orig,
                                "x0_orig": bx0_orig,
                                "text": btext_cleaned,
                            }
                        )

                # Sắp xếp theo thứ tự đọc (từ trên xuống, trái sang phải)
                candidate_blocks_info_prev_page.sort(
                    key=lambda b: (b["y0_orig"], b["x0_orig"])
                )

                prev_page_context_list = [
                    b["text"] for b in candidate_blocks_info_prev_page
                ]
                if prev_page_context_list:
                    # Nối ngữ cảnh từ trang trước vào *trước* ngữ cảnh (nếu có) từ trang hiện tại.
                    # Vì hiện tại final_context_parts đang rỗng, nên đây sẽ là ngữ cảnh duy nhất.
                    final_context_parts = prev_page_context_list
                    if debug:
                        context_text = "\n".join(prev_page_context_list)
                        logger.debug(
                            f"Ngữ cảnh tìm thấy ở cuối trang trước (P{prev_page_num + 1}):\n'{context_text}'"
                        )
                elif debug:
                    logger.debug(
                        f"Không có ngữ cảnh phù hợp ở cuối trang trước (P{prev_page_num + 1})."
                    )

        except Exception as e_prev_page:
            if debug:
                logger.error(
                    f"Lỗi khi xử lý trang trước {prev_page_num + 1} để tìm ngữ cảnh: {e_prev_page}"
                )

    # Kết hợp các phần ngữ cảnh (nếu có nhiều hơn một, ví dụ sau này có thể thêm ngữ cảnh từ nhiều nguồn)
    # Hiện tại, final_context_parts sẽ chỉ có 0 hoặc 1 phần tử dạng list đã join.
    # Nếu logic thay đổi để `final_context_parts` chứa nhiều chuỗi, `"\n\n".join` sẽ phù hợp.
    # Với logic hiện tại, nếu `final_context_parts` có phần tử, đó là một chuỗi đã join.

    final_result_text = ""
    if final_context_parts:
        final_result_text = final_context_parts[
            0
        ]  # Vì chúng ta chỉ điền một khối văn bản (từ cùng trang hoặc trang trước)

    if debug and not final_result_text:
        logger.info(
            f"Không tìm thấy ngữ cảnh nào cho bảng tại trang {table_page_num_0_indexed + 1} (bbox: {table_bbox})."
        )
    elif debug and final_result_text:
        logger.info(
            f"Ngữ cảnh cuối cùng cho bảng tại trang {table_page_num_0_indexed + 1}:\n-----\n{final_result_text}\n-----"
        )

    return final_result_text.strip()



def convert_markdown_to_df(markdown_text: str) -> pd.DataFrame:
    html_table = markdown(markdown_text, extensions=['markdown.extensions.tables'])
    dfs = pd.read_html(StringIO(f"<table>{html_table}</table>"))
    if dfs:
        df = dfs[0]
    else:
        print("Không tìm thấy bảng nào.")
    return df

def fix_merged_row(df_col1, df_col2) -> pd.Series:
    def find_consecutive_true_indices(series):
        consecutive_groups = []
        current_streak = []

        for i, value in enumerate(series):
            if value:
                current_streak.append(i)
            elif len(current_streak) >= 2:
                consecutive_groups.append(current_streak)
                current_streak = []

        if len(current_streak) >= 2:
            consecutive_groups.append(current_streak)

        return consecutive_groups

    for group in find_consecutive_true_indices(df_col1 == df_col2):
        if group[0] > 0:
            df_col2.iloc[group] = df_col2.iloc[group[0] - 1]
    return df_col2

def fix_merged_row_df(df: pd.DataFrame) -> pd.DataFrame:
    test_df = df.copy()
    n_cols = test_df.shape[1]

    for i in range(1, n_cols - 1):
        test_df.iloc[:, i + 1] = fix_merged_row(test_df.iloc[:, i], test_df.iloc[:, i + 1])

    return test_df


def get_tables_from_pdf(
    doc=Union[str, pymupdf.Document],
    image_path: str = "images",
    write_images: bool = False,
    pages: List[int] = None,
) -> List[Table]:
    if isinstance(doc, str):
        doc = pymupdf.open(doc)
    md_text = pymupdf4llm.to_markdown(
        doc=doc, image_path=image_path, page_chunks=True, write_images=write_images
    )

    total_tables: List[Table] = []

    for page in md_text:
        metadata = page["metadata"]
        page_idx = metadata["page"]
        source = get_pdf_name(metadata["file_path"])
        tables_metadata = page["tables"]
        tables_text = split_markdown_into_tables(page["text"])
        for table_metadata, table_text in zip(tables_metadata, tables_text):
            bbox = table_metadata["bbox"]
            n_rows = table_metadata["rows"]
            n_columns = table_metadata["columns"]
            # find context before
            target_table_page_0_indexed = page_idx - 1
            actual_prev_page_0_indexed = target_table_page_0_indexed - 1
            filtered_prev_page_table_bboxes = []
            if actual_prev_page_0_indexed >= 0:
                for t_prev in total_tables:
                    if (
                        t_prev["page"] - 1 == actual_prev_page_0_indexed
                    ):  # So sánh trang 0-indexed
                        filtered_prev_page_table_bboxes.append(t_prev["bbox"])

            context = get_context_before_table(
                doc=doc,
                table_page_num_0_indexed=target_table_page_0_indexed,
                table_bbox=bbox,
                prev_page_all_table_bboxes=filtered_prev_page_table_bboxes,
                # debug=True
            )

            table: Table = {
                "text": table_text,
                "page": page_idx,
                "source": source,
                "n_rows": n_rows,
                "n_columns": n_columns,
                "bbox": bbox,
                "context_before": context,
            }
            total_tables.append(table)
    # thêm is_new_section_context
    contexts = [table["context_before"] for table in total_tables]
    res = get_is_new_section_context(contexts)
    for table, is_new in zip(total_tables, res["is_new_section_context"]):
        table["is_new_section_context"] = is_new

    # thêm is_has_header
    headers = [get_headers_from_markdown(table["text"]) for table in total_tables]
    res = get_is_has_header(headers)
    for table, is_has in zip(total_tables, res["is_has_header"]):
        table["is_has_header"] = is_has
    return total_tables


def get_tables_from_pdf_2(
    doc: Union[str, pymupdf.Document],
    pages: List[int] = None,
) -> List[Table]:
    if isinstance(doc, str):
        doc = pymupdf.open(doc)
    total_tables = []

    source = get_pdf_name(doc.name)
    if pages is None:
        pages = list(range(1, doc.page_count + 1))

    for page in tqdm(pages, desc="Processing pages"):
        page_idx = page - 1
        page = doc.load_page(page_idx)
        tables = page.find_tables(strategy="lines_strict").tables
        if tables:
            for table in tables:
                bbox = table.bbox
                n_rows = table.row_count
                n_columns = table.col_count
                text = table.to_markdown()

                target_table_page_0_indexed = page_idx
                actual_prev_page_0_indexed = target_table_page_0_indexed - 1
                filtered_prev_page_table_bboxes = []
                if actual_prev_page_0_indexed >= 0:
                    for t_prev in total_tables:
                        if t_prev["page"] - 1 == actual_prev_page_0_indexed:
                            filtered_prev_page_table_bboxes.append(t_prev["bbox"])

                context = get_context_before_table(
                    doc=doc,
                    table_page_num_0_indexed=target_table_page_0_indexed,
                    table_bbox=bbox,
                    prev_page_all_table_bboxes=filtered_prev_page_table_bboxes,
                )

                table_obj = {
                    "text": text,
                    "page": page_idx + 1,
                    "source": source,
                    "n_rows": n_rows,
                    "n_columns": n_columns,
                    "bbox": bbox,
                    "context_before": context,
                    "is_new_section_context": False,
                }
                total_tables.append(table_obj)

    contexts = [
        (i, table["context_before"])
        for i, table in enumerate(total_tables)
        if table["context_before"] != ""
    ]

    # Retry logic for get_is_new_section_context
    max_retries = 5
    for attempt in range(max_retries):
        res = get_is_new_section_context([context for _, context in contexts])
        if len(res["is_new_section_context"]) == len(contexts):
            break
        logger.warning(f"Retry {attempt + 1}/{max_retries} for get_is_new_section_context due to length mismatch.")
    else:
        logger.error("Failed to get correct response length from get_is_new_section_context after retries.")

    for (i, _), is_new in zip(contexts, res["is_new_section_context"]):
        total_tables[i]["is_new_section_context"] = is_new

    headers = [get_headers_from_markdown(table["text"]) for table in total_tables]

    # Retry logic for get_is_has_header
    for attempt in range(max_retries):
        res = get_is_has_header(headers)
        if len(res["is_has_header"]) == len(headers):
            break
        logger.warning(f"Retry {attempt + 1}/{max_retries} for get_is_has_header due to length mismatch.")
    else:
        logger.error("Failed to get correct response length from get_is_has_header after retries.")

    for table, is_has in zip(total_tables, res["is_has_header"]):
        table["is_has_header"] = is_has
    return total_tables


def find_spanned_table_groups(
    tables: List[Table], debug: bool = False
) -> List[List[Table]]:
    """
    Identifies groups of tables that span across multiple pages from a list of tables.

    The logic follows specific rules to determine if a table is a continuation
    of the previous one on the next page, considering factors like the context
    before the table, the presence of headers, and new section markers. Tables
    are first sorted by page number and then by their vertical position on the page.

    Args:
        tables: A list of Table objects. These tables will be sorted internally
                before processing.
        debug: If True, enables detailed debug logging to stderr. This will also
               disable the tqdm progress bar to prevent log clutter.

    Returns:
        A list of table groups. Each group is a list of Table objects
        that are considered part of the same logical table spanning
        across one or more pages. An empty list is returned if the input
        `tables` list is empty.
    """
    if not tables:
        return []

    # Sort tables by page and then by vertical position (top of bbox y0).
    # Assuming bbox is (x0, y0, x1, y1) where y0 is the top coordinate.
    # This is crucial for correctly identifying sequential tables in the document.
    sorted_tables = sorted(tables, key=lambda t: (t["page"], t["bbox"][1]))

    if debug and sorted_tables:
        logger.debug(
            f"Tables sorted. First table on page: {sorted_tables[0]['page']}, Last table on page: {sorted_tables[-1]['page']}"
        )

    table_groups: List[List[Table]] = []
    # Initialize the first group with the first table.
    current_group: List[Table] = [sorted_tables[0]]

    # Determine if tqdm should be used: for a meaningful number of tables and not in debug mode.
    # tqdm is disabled in debug mode to prevent progress bar output from cluttering detailed logs.
    should_use_tqdm = (
        len(sorted_tables) > 10
    )  # Arbitrary threshold: use tqdm for more than 10 tables

    # The loop iterates from the second table.
    loop_iterator = range(1, len(sorted_tables))
    if should_use_tqdm:
        loop_iterator = tqdm(
            loop_iterator,
            desc="Processing tables for spanning",
            unit="table",
            ncols=100,
            disable=debug,
        )

    for i in loop_iterator:
        t_current = sorted_tables[i]
        # The decision to span is based on the last table added to the current_group.
        last_table_in_current_group = current_group[-1]

        if debug:
            logger.debug(
                f"--- Iteration for table index {i} (Page {t_current['page']}) ---"
            )
            logger.debug(
                f"T_current: Page {t_current['page']}, HasHeader: {t_current['is_has_header']}, "
                f"NewSectionCtx: {t_current['is_new_section_context']}, "
                f"CtxBefore: '{t_current['context_before'][:30].strip()}...'"
            )
            logger.debug(
                f"Comparing with T_prev_in_group: Page {last_table_in_current_group['page']}"
            )

        # Pre-condition for span: T_current must be on the page immediately following T_prev_in_group.
        is_on_next_page = t_current["page"] == last_table_in_current_group["page"] + 1

        if not is_on_next_page:
            if debug:
                logger.debug(
                    f"NOT A SPAN: T_current (Page {t_current['page']}) is not on the next page "
                    f"after T_prev_in_group (Page {last_table_in_current_group['page']}). Finalizing current group."
                )
            table_groups.append(list(current_group))  # Finalize current_group
            current_group = [t_current]  # Start a new group with t_current
            continue

        # At this point, is_on_next_page is True.

        # Case: New Table if T_current is marked by a new section context.
        # This condition has high precedence.
        if t_current["is_new_section_context"]:
            if debug:
                logger.debug(
                    f"NOT A SPAN: T_current (Page {t_current['page']}) has 'is_new_section_context' == True. "
                    "Finalizing current group."
                )
            table_groups.append(list(current_group))
            current_group = [t_current]
            continue

        # Case: New Table if T_current has its own header (and not a new section context).
        # A table with its own significant header is usually independent.
        if t_current["is_has_header"]:
            if debug:
                logger.debug(
                    f"NOT A SPAN: T_current (Page {t_current['page']}) has 'is_has_header' == True. "
                    "Finalizing current group."
                )
            table_groups.append(list(current_group))
            current_group = [t_current]
            continue

        # If we reach here, the conditions for a span are strong:
        # 1. T_current is on the next page.
        # 2. T_current.is_new_section_context is False.
        # 3. T_current.is_has_header is False.
        # The nature of 'context_before' differentiates clear vs. noisy span.

        # Case 1: Classic Span (context_before is empty).
        if not t_current["context_before"]:  # context_before is empty or None
            if debug:
                logger.debug(
                    f"IS A SPAN (Classic): T_current (Page {t_current['page']}) meets span criteria, "
                    "and 'context_before' is empty. Adding to current group."
                )
            current_group.append(t_current)
        else:  # Case 2: Span with "Noise" Context (context_before is not empty but not a new section marker).
            if debug:
                logger.debug(
                    f"IS A SPAN (Noisy Context): T_current (Page {t_current['page']}) meets span criteria, "
                    "and 'context_before' is present (considered noise). Adding to current group."
                )
            current_group.append(t_current)
            # Note: The problem description mentioned an optional further check for column compatibility here.
            # e.g., t_current['n_columns'] == last_table_in_current_group['n_columns'].
            # This is not strictly enforced by the provided flowchart logic for this case
            # but could be an enhancement for higher precision if required.

    # Add the last processed group to table_groups.
    # This ensures the group being built is added, especially if the loop finishes.
    if current_group:
        table_groups.append(list(current_group))

    if debug:
        logger.debug(
            f"Finished finding spanned table groups. Total groups found: {len(table_groups)}."
        )
        for i, group in enumerate(table_groups):
            group_pages = [t["page"] for t in group]
            logger.debug(
                f"Group {i + 1}: Contains {len(group)} table segment(s) on page(s): {group_pages}"
            )

    return table_groups

def merge_tables(tables: List[Table]) -> List[MergedTable]:
    # Convert each table's markdown text to a DataFrame and fix merged rows
    dfs = [fix_merged_row_df(convert_markdown_to_df(table["text"])) for table in tables]
    
    # Determine the maximum number of columns across all DataFrames
    max_cols = max(df.shape[1] for df in dfs)
    
    # Ensure all DataFrames have the same number of columns by adding blank columns
    for i, df in enumerate(dfs):
        if df.shape[1] < max_cols:
            # Add blank columns to match the maximum column count
            for _ in range(max_cols - df.shape[1]):
                df[df.shape[1]] = ""
    
    # Handle the case where dfs[0] has fewer columns than max_cols
    if dfs[0].shape[1] < max_cols:
        for _ in range(max_cols - dfs[0].shape[1]):
            dfs[0][dfs[0].shape[1]] = ""
    
    # Concatenate all DataFrames along the rows (vertically)
    headers = get_headers_from_markdown(tables[0]["text"])
    dfs = [dfs[0]] + [solve_non_header_table(df, headers) for df in dfs[1:]]
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Post-process the merged DataFrame
    # Fill NaN values with empty strings
    merged_df.fillna("", inplace=True)
    # Remove columns with names starting with 'Col' and replace their values with empty strings
    merged_df.columns = [re.sub(r"^Col\d+", "", col) for col in merged_df.columns]
    merged_df.replace(r"^Col\d+", "", regex=True, inplace=True)
    
    # Create a MergedTable with the combined data
    merged_table = MergedTable(
        text=merged_df.to_markdown(index=False),
        page=[table["page"] for table in tables],
        source=tables[0]["source"],
        bbox=[table["bbox"] for table in tables],
        headers=[get_headers_from_markdown(table["text"]) for table in tables],
        n_rows=merged_df.shape[0],
        n_columns=merged_df.shape[1],
        context_before=tables[0]["context_before"],
    )
    
    return merged_table


def full_pipeline(doc: Union[List[str], List[pymupdf.Document]], pages: List[int] = None) -> List[MergedTable]:
    merged_tables = []
    
    # Convert single string to list if needed
    if isinstance(doc, str):
        doc = [doc]
    
    for d in doc:
        try:
            if isinstance(d, str):
                # Validate file path
                if not os.path.exists(d):
                    logger.error(f"File not found: {d}")
                    continue
                logger.info(f"Processing document: {get_pdf_name(d)}")
            else:
                logger.info(f"Processing document: {d.name}")
            
            logger.info("   Extracting tables...")
            tables = get_tables_from_pdf_2(d, pages)
            logger.info("   Finding spanned table groups...")
            table_groups = find_spanned_table_groups(tables)
            logger.info("   Merging tables...")
            merged_tables.extend([merge_tables(group) for group in table_groups])
            logger.info("   Processed document: Done!")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            continue
            
    logger.info("Done!")
    return merged_tables


if __name__ == "__main__":
    # Example usage with proper file path
    source_path = "C:/Users/PC/CODE/WDM-AI-TEMIS/data/pdfs/0c92f65db928c431023f59603039aa1e.pdf"
    
    # Validate file exists before processing
    if not os.path.exists(source_path):
        logger.error(f"File not found: {source_path}")
    else:
        merged_tables = full_pipeline(source_path)
        for table in merged_tables:
            print("Tables:", table['context_before'])
            print("Page:", table['page'])
            print(table["text"])
            print("="*100)

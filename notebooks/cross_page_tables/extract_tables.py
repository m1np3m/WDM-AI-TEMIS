import logging
import os
import time
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Set,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import pandas as pd
import pymupdf
from notebooks.CrossPageTablePyMuPDF.check_using_api import generate

pd.set_option("future.no_silent_downcasting", True)


class ExtractedDataType(TypedDict):
    """Kiểu dữ liệu cho dữ liệu được trích xuất từ PDF"""
    dataframes: List[pd.DataFrame]
    page_numbers: List[int]


def get_pdf_name(source: str) -> str:
    """Lấy tên file PDF không có phần mở rộng"""
    pdf_name = os.path.basename(source)
    file_name_part, file_extension_part = os.path.splitext(pdf_name)
    return file_name_part


def preprocess_df(df):
    """Tiền xử lý DataFrame: xóa cột rỗng, hàng rỗng và chuẩn hóa tên cột"""
    if df.empty:
        return df

    df_processed = df.copy()
    # Xóa các cột chỉ chứa None với tên Col1, Col2, Col3,...
    col_names_with_prefix = [
        col for col in df_processed.columns if str(col).startswith("Col")
    ]
    for col in col_names_with_prefix:
        if df_processed[col].isnull().all():
            df_processed.drop(col, axis=1, inplace=True)

    # Xóa hàng chỉ chứa None, "", NaN
    df_processed = df_processed.replace([None, ""], np.nan)
    df_processed = df_processed.dropna(how="all")
    df_processed = df_processed.fillna("")

    # Đổi tên cột bắt đầu bằng Col thành rỗng
    df_processed_columns = df_processed.columns
    for col in df_processed_columns:
        if str(col).startswith("Col"):
            df_processed.rename(columns={col: ""}, inplace=True)

    return df_processed


def get_limited_text_before_table(
    page: pymupdf.Page,
    current_table_bbox: pymupdf.Rect,
    all_table_bboxes_on_page: List[pymupdf.Rect],
    n_tokens: int,
    search_height_multiplier: float = 2.0,
    min_search_height: int = 30,
) -> str:
    """Lấy văn bản giới hạn trước bảng, tránh vùng chồng lấn với bảng khác"""
    if isinstance(current_table_bbox, tuple):
        if len(current_table_bbox) == 4:
            try:
                current_table_bbox = pymupdf.Rect(current_table_bbox)
            except Exception:
                return ""
        else:
            return ""
    elif not isinstance(current_table_bbox, pymupdf.Rect):
        return ""

    if not current_table_bbox or current_table_bbox.is_empty:
        return ""

    page_width = page.rect.width

    try:
        table_height = current_table_bbox.height
        if table_height <= 0:
            table_height = 10
    except AttributeError:
        return ""

    search_area_height = max(table_height * search_height_multiplier, min_search_height)
    search_rect_y0 = max(0, current_table_bbox.y0 - 1)
    search_rect_y1 = max(0, search_rect_y0 - search_area_height)

    if search_rect_y1 >= search_rect_y0:
        if current_table_bbox.y0 <= min_search_height / 2:
            search_rect_y0 = max(0, current_table_bbox.y0 - 1)
            search_rect_y1 = max(0, search_rect_y0 - (min_search_height / 2))
            if search_rect_y1 >= search_rect_y0:
                return ""
        else:
            search_rect_y1 = max(0, current_table_bbox.y0 - min_search_height)
            search_rect_y0 = max(0, current_table_bbox.y0 - 1)
            if search_rect_y1 >= search_rect_y0:
                return ""

    search_rect = pymupdf.Rect(0, search_rect_y1, page_width, search_rect_y0)
    if search_rect.is_empty or search_rect.height <= 0:
        return ""

    blocks_in_search_area = page.get_text("blocks", clip=search_rect, sort=True)

    filtered_text_lines = []
    for block_data in blocks_in_search_area:
        if block_data[6] != 0:
            continue

        block_rect = pymupdf.Rect(
            block_data[0], block_data[1], block_data[2], block_data[3]
        )
        block_text_content = block_data[4]

        is_part_of_a_table = False
        for table_bbox_on_page in all_table_bboxes_on_page:
            if isinstance(table_bbox_on_page, tuple):
                test_table_rect = pymupdf.Rect(table_bbox_on_page)
            else:
                test_table_rect = table_bbox_on_page

            if not test_table_rect.is_empty:
                intersection = block_rect.irect & test_table_rect.irect
                if not intersection.is_empty:
                    is_part_of_a_table = True
                    break

        if not is_part_of_a_table:
            cleaned_block_text = block_text_content.replace("\n", " ").strip()
            if cleaned_block_text:
                filtered_text_lines.append(cleaned_block_text)

    text_above = " ".join(filtered_text_lines)
    text_above = " ".join(text_above.split())

    if not text_above:
        return ""

    tokens = text_above.split()
    if len(tokens) > n_tokens:
        limited_text = " ".join(tokens[-n_tokens:])
    else:
        limited_text = " ".join(tokens)

    return limited_text


def extract_tables_and_contexts(
    doc: pymupdf.Document,
    n_tokens_context: int = 15,
    page_numbers_to_process: Union[int, List[int], str] = "all",
    search_height_multiplier: float = 2.5,
    min_search_height: int = 40,
) -> Tuple[ExtractedDataType, List[str]]:
    """Trích xuất bảng và ngữ cảnh từ tài liệu PDF"""
    all_dataframes: List[pd.DataFrame] = []
    all_contexts: List[str] = []
    all_df_page_numbers: List[int] = []

    default_extracted_data: ExtractedDataType = {"dataframes": [], "page_numbers": []}

    if not isinstance(doc, pymupdf.Document):
        print("Lỗi: 'doc' phải là một đối tượng pymupdf.Document.")
        return default_extracted_data, []

    if isinstance(page_numbers_to_process, int):
        pages_to_process_indices = [page_numbers_to_process - 1]
    elif isinstance(page_numbers_to_process, list) and all(
        isinstance(pn, int) for pn in page_numbers_to_process
    ):
        pages_to_process_indices = sorted(
            list(set(p - 1 for p in page_numbers_to_process))
        )
    elif page_numbers_to_process == "all":
        pages_to_process_indices = list(range(len(doc)))
    else:
        print(
            "Lỗi: page_numbers_to_process phải là số nguyên hoặc danh sách số nguyên (bắt đầu từ 1)."
        )
        return default_extracted_data, []

    for page_index in pages_to_process_indices:
        if not (0 <= page_index < len(doc)):
            print(
                f"Cảnh báo: Số trang {page_index + 1} (index {page_index}) không hợp lệ. Bỏ qua."
            )
            continue

        page = doc[page_index]
        current_page_human_readable = page.number + 1

        page_table_finder = page.find_tables()
        all_found_table_bboxes_on_page: List[pymupdf.Rect] = []
        if page_table_finder.tables:
            all_found_table_bboxes_on_page = [
                pymupdf.Rect(tbl.bbox) for tbl in page_table_finder.tables if tbl.bbox
            ]

        if page_table_finder.tables:
            for table_obj in page_table_finder.tables:
                if not table_obj.bbox:
                    continue

                df_original = table_obj.to_pandas()
                df_original = preprocess_df(df_original)

                if not df_original.empty:
                    current_table_actual_bbox = pymupdf.Rect(table_obj.bbox)

                    context_text = get_limited_text_before_table(
                        page,
                        current_table_actual_bbox,
                        all_found_table_bboxes_on_page,
                        n_tokens_context,
                        search_height_multiplier=search_height_multiplier,
                        min_search_height=min_search_height,
                    )
                    all_contexts.append(context_text)
                    all_dataframes.append(df_original)
                    all_df_page_numbers.append(current_page_human_readable)

    extracted_data_output: ExtractedDataType = {
        "dataframes": all_dataframes,
        "page_numbers": all_df_page_numbers,
    }

    if not (len(all_dataframes) == len(all_contexts) == len(all_df_page_numbers)):
        print(
            f"Cảnh báo logic nghiêm trọng: Độ dài các list không khớp! DFs: {len(all_dataframes)}, Contexts: {len(all_contexts)}, PageNums: {len(all_df_page_numbers)}"
        )
        return default_extracted_data, []

    return extracted_data_output, all_contexts


def get_column_types(df: pd.DataFrame) -> List[str]:
    """
    Determine data types for each column

    Args:
        df: A pandas DataFrame to get column types from.

    Returns:
        A list of data types for each column.
    """
    types: List[str] = []
    if df.columns.empty:
        return types

    for col_name in df.columns:
        current_column_object = df[col_name]

        is_truly_all_na = False
        effective_dtype = None

        if isinstance(current_column_object, pd.DataFrame):
            is_truly_all_na = pd.isna(current_column_object).all().all()
            if not is_truly_all_na:
                effective_dtype = current_column_object.iloc[:, 0].dtype
        elif isinstance(current_column_object, pd.Series):
            is_truly_all_na = pd.isna(current_column_object).all()
            if not is_truly_all_na:
                effective_dtype = current_column_object.dtype
        else:
            types.append("error_unexpected_type")
            continue

        if is_truly_all_na:
            types.append("unknown")
        else:
            if effective_dtype is not None:
                types.append(
                    "numeric"
                    if pd.api.types.is_numeric_dtype(effective_dtype)
                    else str(effective_dtype)
                )
            else:
                types.append("unknown_error_in_logic")
    return types


def get_input_df(
    df: pd.DataFrame, n_rows: int = 10, sep: str = "=", max_tokens: int = 15
) -> str:
    """
    Get input DataFrame with sampling and formatting for LLM processing

    Args:
        df: A pandas DataFrame to get input DataFrame from.
        n_rows: The number of rows to sample from the DataFrame.
        sep: The separator to use for the CSV string.
        max_tokens: The maximum number of tokens to use for the CSV string.

    Returns:
        A string representation of the DataFrame in markdown format.
    """
    if df.empty:
        return ""

    df_copy = df.copy()
    
    column_data_types = get_column_types(df_copy)

    for dtype, col_name in zip(column_data_types, df_copy.columns):
        if dtype == "object":
            if isinstance(df_copy[col_name], pd.Series):
                current_series = df_copy[col_name]
                try:
                    df_copy.loc[:, col_name] = current_series.astype(str).apply(
                        lambda x: " ".join(x.split()[:max_tokens])
                    )
                except (
                    AttributeError
                ):  # Handle cases where elements might not be strings
                    df_copy.loc[:, col_name] = current_series.apply(
                        lambda x: " ".join(str(x).split()[:max_tokens])
                    )

    num_original_rows = len(df_copy)
    sampled_df_parts: List[pd.DataFrame] = []

    if num_original_rows <= n_rows:
        sampled_df_parts.append(df_copy)
    else:
        n_head = (n_rows + 1) // 2
        n_tail = n_rows - n_head

        sampled_df_parts.append(df_copy.head(n_head))
        if n_tail > 0:
            sampled_df_parts.append(df_copy.tail(n_tail))

    if not sampled_df_parts:
        return ""

    final_df_to_sample = pd.concat(sampled_df_parts)
    final_df_to_sample = final_df_to_sample.drop_duplicates().reset_index(drop=True)

    # Convert to markdown table format
    markdown_table = final_df_to_sample.to_markdown(index=False, tablefmt="pipe")
    return markdown_table


def build_llm_prompt(extracted_data: ExtractedDataType, contexts: List[str]) -> str:
    """
    Builds the full prompt string for the LLM by combining table data and contexts.
    """
    full_prompt = ""
    dataframes = extracted_data["dataframes"]

    for i, df in enumerate(dataframes):
        if df.empty:
            continue

        # These functions (get_input_df, get_column_types) are imported from extract_tables module
        df_string = get_input_df(df, n_rows=7)
        column_types = get_column_types(df=df)
        n_columns = len(df.columns)

        prompt_part = f"Table {i}:\n"
        # Ensure context index is valid and context is not empty/None
        if i < len(contexts) and contexts[i]:
            prompt_part += f"Context before table:\n{contexts[i]}\n\n"
        prompt_part += f"{df_string}\nNumber of columns: {n_columns}\nColumn types: {column_types}\n\n"

        full_prompt += prompt_part
    return full_prompt


class ExtractedDataType(TypedDict):
    dataframes: List[pd.DataFrame]
    page_numbers: List[int]


class ProcessedTableEntry(TypedDict):
    dataframe: pd.DataFrame
    page_numbers: List[int]
    source: str


def solve_non_header_table(df: pd.DataFrame, target_headers: List[str]) -> pd.DataFrame:
    """
    Chuyển đổi DataFrame có thể không có header bằng cách di chuyển tên cột hiện tại
    vào dòng dữ liệu đầu tiên, sau đó gán target_headers.
    Đã tối ưu để xử lý số lượng cột không khớp mà không làm mất dữ liệu dòng không cần thiết.
    """
    if not isinstance(target_headers, list):
        logging.warning(
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
            # Nếu header gốc (dưới dạng dữ liệu) dài hơn target, cắt bớt
            first_row_data_values = first_row_data_values[: len(target_headers)]
        else:
            # Nếu header gốc (dưới dạng dữ liệu) ngắn hơn target, thêm NaN
            first_row_data_values.extend(
                [np.nan] * (len(target_headers) - len(first_row_data_values))
            )

    new_first_row_df = pd.DataFrame([first_row_data_values], columns=target_headers)

    num_target_cols = len(target_headers)
    current_data_cols = df_copy.shape[1]

    if num_target_cols == 0:
        if current_data_cols > 0:
            df_copy = pd.DataFrame(index=df_copy.index)
    else:  # num_target_cols > 0
        if current_data_cols < num_target_cols:
            for i in range(num_target_cols - current_data_cols):
                df_copy[f"__temp_added_col_{i}"] = np.nan
        elif current_data_cols > num_target_cols:
            df_copy = df_copy.iloc[:, :num_target_cols]

    df_copy.columns = target_headers

    result_df = pd.concat([new_first_row_df, df_copy], ignore_index=True)

    return result_df.reset_index(drop=True)


def main_concatenation_logic(
    extracted_data: ExtractedDataType,
    concat_json: Dict[str, Any],
    source_name: str,
) -> List[ProcessedTableEntry]:
    """
    Ghép các DataFrame dựa trên chỉ mục được cung cấp trong concat_json.
    Các DataFrame trong nhóm ghép nối sẽ được xử lý bằng solve_non_header_table.
    Các DataFrame không được chỉ định để ghép sẽ được coi là các bảng độc lập.

    Args:
        extracted_data: Một dictionary chứa danh sách các DataFrame và
                        số trang tương ứng của chúng.
        concat_json: Một dictionary chứa khóa 'concatable_tables', là một danh sách
                     các nhóm chỉ mục DataFrame cần được ghép.
        source_name: Tên nguồn của dữ liệu (ví dụ: tên file, URL).

    Returns:
        Một danh sách các đối tượng ProcessedTableEntry.

    Raises:
        ValueError: Nếu số lượng dataframes và page_numbers không khớp.
    """
    all_original_dataframes = extracted_data["dataframes"]
    all_original_page_numbers = extracted_data["page_numbers"]

    if len(all_original_dataframes) != len(all_original_page_numbers):
        raise ValueError(
            "Số lượng DataFrame và số lượng page_numbers trong extracted_data không khớp."
        )

    processed_tables_result: List[ProcessedTableEntry] = []
    processed_df_indices: Set[int] = set()

    if "concatable_tables" in concat_json and concat_json["concatable_tables"]:
        for group_info in concat_json["concatable_tables"]:
            indices_in_group = group_info.get("table_index", [])

            if not indices_in_group:
                continue

            actual_indices_for_this_concat_group: List[int] = []
            for idx in indices_in_group:
                if 0 <= idx < len(all_original_dataframes):
                    actual_indices_for_this_concat_group.append(idx)
                    processed_df_indices.add(idx)
                else:
                    logging.warning(f"Index {idx} out of bounds, skipping for group.")

            if not actual_indices_for_this_concat_group:
                continue

            current_group_original_dfs = [
                all_original_dataframes[idx]
                for idx in actual_indices_for_this_concat_group
            ]

            pages_for_this_group: Set[int] = set()
            for valid_idx in actual_indices_for_this_concat_group:
                pages_for_this_group.add(all_original_page_numbers[valid_idx])

            max_cols_in_group = 0
            valid_dfs_for_shape_check = [
                df for df in current_group_original_dfs if isinstance(df, pd.DataFrame)
            ]
            if valid_dfs_for_shape_check:
                col_counts = [df.shape[1] for df in valid_dfs_for_shape_check]
                if col_counts:
                    max_cols_in_group = max(col_counts)

            group_target_headers = [str(i) for i in range(max_cols_in_group)]

            processed_dfs_for_group: List[pd.DataFrame] = []
            for df_original in current_group_original_dfs:
                if isinstance(df_original, pd.DataFrame):
                    processed_df = solve_non_header_table(
                        df_original, group_target_headers
                    )
                    processed_dfs_for_group.append(processed_df)
                else:
                    logging.warning(
                        f"Item at index used for group is not a DataFrame, skipping."
                    )

            if processed_dfs_for_group:
                final_concatenated_df = pd.concat(
                    processed_dfs_for_group, ignore_index=True
                )
                final_concatenated_df = preprocess_df(final_concatenated_df)

                new_entry: ProcessedTableEntry = {
                    "dataframe": final_concatenated_df,
                    "page_numbers": sorted(list(pages_for_this_group)),
                    "source": get_pdf_name(source_name),
                }
                processed_tables_result.append(new_entry)
            elif (
                actual_indices_for_this_concat_group
            ):  # Nếu có chỉ mục nhưng không có df nào được xử lý
                logging.warning(
                    f"Group with original indices {actual_indices_for_this_concat_group} resulted in no DataFrames to concatenate after processing."
                )

    total_num_dataframes = len(all_original_dataframes)
    for i in range(total_num_dataframes):
        if i not in processed_df_indices:
            independent_dataframe = all_original_dataframes[i]
            independent_dataframe = solve_non_header_table(
                independent_dataframe,
                [str(i) for i in range(len(independent_dataframe.columns))],
            )
            if isinstance(independent_dataframe, pd.DataFrame):  # Đảm bảo là DataFrame
                page_number_list = [all_original_page_numbers[i]]

                independent_entry: ProcessedTableEntry = {
                    "dataframe": independent_dataframe,  # Giữ nguyên DataFrame gốc
                    "page_numbers": page_number_list,
                    "source": source_name,
                }
                processed_tables_result.append(independent_entry)
            else:
                logging.warning(
                    f"Independent item at index {i} is not a DataFrame, skipping."
                )

    return processed_tables_result

## ---------------------- FOR TESTING ----------------------
def get_table_content_from_file(file_path: str) -> Tuple[ExtractedDataType, List[str]]:
    doc = pymupdf.open(file_path)
    extracted_data, contexts = extract_tables_and_contexts(doc)

    return extracted_data, contexts


def process_pdf_file_from_extracted_data(
    extracted_data: ExtractedDataType,
    contexts: List[str],
    return_type: Literal["dataframe", "markdown"] = "dataframe",
    verbose: bool = False,
    source_name: str = "testing",
) -> List[ProcessedTableEntry]:
    concat_json = generate(build_llm_prompt(extracted_data, contexts))
    processed_tables_result = main_concatenation_logic(
        extracted_data, concat_json, source_name
    )
    return processed_tables_result


## ---------------------- FULL PROCESSING FUNCTION ----------------------
def process_pdf_file(
    file_paths: Union[str, List[str]],
    return_type: Literal["dataframe", "markdown"] = "dataframe",
    verbose: bool = False,
) -> List[ProcessedTableEntry]:
    """
    Hàm này dùng để xử lý tất cả các file PDF được cung cấp trong file_paths, trích xuất ra các bảng từ PDF.
    
    Args:
        file_paths: Đường dẫn đến file PDF hoặc danh sách các đường dẫn đến file PDF.
        return_type: Kiểu trả về, có thể là "dataframe" hoặc "markdown".
        verbose: Nếu True, sẽ in ra thông tin chi tiết về quá trình xử lý.

    Returns:
        Một danh sách các đối tượng ProcessedTableEntry.
    """
    final_processed_tables_result: List[ProcessedTableEntry] = []
    final_concat_json: list = []
    final_input_tokens: list = []
    final_output_tokens: list = []
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    start_time = time.time()
    for file_path in file_paths:
        if verbose:
            print(f"Processing file: {get_pdf_name(file_path)}...")
        extracted_data, contexts = get_table_content_from_file(file_path)
        concat_json = generate(build_llm_prompt(extracted_data, contexts))
        logging.info(
            f"Model: {concat_json['model']}, Input tokens: {concat_json['input_tokens']}, Output tokens: {concat_json['output_tokens']}"
        )
        processed_tables_result = main_concatenation_logic(
            extracted_data, concat_json, file_path
        )
        final_processed_tables_result.extend(processed_tables_result)
        final_concat_json.append(concat_json)
        final_input_tokens.append(concat_json["input_tokens"])
        final_output_tokens.append(concat_json["output_tokens"])
        if verbose:
            print(
                f"File: {get_pdf_name(file_path)}, Input tokens: {concat_json['input_tokens']}, Output tokens: {concat_json['output_tokens']}"
            )

    if verbose:
        print(
            f"Model: {final_concat_json[0]['model']}, Total input tokens: {sum(final_input_tokens)}, Total output tokens: {sum(final_output_tokens)}"
        )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.3f} seconds")
    if return_type == "markdown":
        for table in final_processed_tables_result:
            table["dataframe"] = table["dataframe"].to_markdown(index=False)
    elif return_type == "dataframe":
        pass

    return final_processed_tables_result


def get_table_content(
    processed_tables_with_pages: List[ProcessedTableEntry],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Hàm này chỉ dùng để xuất kết quả dưới dạng markdown
    """
    def table_to_markdown(
        table: pd.DataFrame, name: str, page_numbers: List[int]
    ) -> str:
        markdown_str = f"# {name}\n"
        if len(page_numbers) > 1:
            # cross-page table
            markdown_str += f"This is a cross-page table. It spans multiple pages. Page numbers: {page_numbers}\n"
        else:
            # single-page table
            markdown_str += (
                f"This is a single-page table. Page number: {page_numbers[0]}\n\n"
            )
        markdown_str += table.to_markdown(index=False)

        table_shape = table.shape
        markdown_str += f"\n\nShape: {table_shape}\n"

        return markdown_str

    results: Dict[str, List[Dict[str, Any]]] = {}

    sources_ = set(table["source"] for table in processed_tables_with_pages)

    for source in sources_:
        file_name_part = get_pdf_name(source)

        # Get processed_tables_with_pages_i where 'source' key matches file_name_part
        processed_tables_source = [
            table
            for table in processed_tables_with_pages
            if table["source"] == file_name_part
        ]

        # logging
        print(f"source {file_name_part} has {len(processed_tables_source)} tables")

        for idx, table in enumerate(processed_tables_source):
            table_df = table["dataframe"]
            table_df = table_df.fillna("")
            page_numbers = table["page_numbers"]
            table_name = f"{file_name_part}_table_{idx}"
            markdown_str = table_to_markdown(table_df, table_name, page_numbers)

            result = {
                "table_content": markdown_str,
                "page_numbers": page_numbers,
                "source": file_name_part,
                "table_idx": idx,
            }
            if file_name_part not in results:
                results[file_name_part] = []
            results[file_name_part].append(result)
    return results

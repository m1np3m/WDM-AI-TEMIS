import logging
import os
import time
from typing import Any, Dict, List, Literal, Set, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import pymupdf
from check_using_api import generate

pd.set_option("future.no_silent_downcasting", True)


class ExtractedDataType(TypedDict):
    dataframes: List[pd.DataFrame]
    page_numbers: List[int]


def get_pdf_name(source: str) -> str:
    pdf_name = os.path.basename(source)
    file_name_part, file_extension_part = os.path.splitext(pdf_name)
    return file_name_part


def preprocess_df(df):
    if df.empty:
        return df

    df_processed = df.copy()
    col_names_with_prefix = [
        col for col in df_processed.columns if str(col).startswith("Col")
    ]
    for col in col_names_with_prefix:
        if df_processed[col].isnull().all():
            df_processed.drop(col, axis=1, inplace=True)

    df_processed = df_processed.replace([None, ""], np.nan)
    df_processed = df_processed.dropna(how="all")
    df_processed = df_processed.fillna("")

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
    main_content_max_x_ratio: float = 0.65,
) -> str:
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
        for table_bbox_on_pg in all_table_bboxes_on_page:
            test_table_rect = (
                pymupdf.Rect(table_bbox_on_pg)
                if isinstance(table_bbox_on_pg, tuple)
                else table_bbox_on_pg
            )
            if not test_table_rect.is_empty:
                if not (block_rect.irect & test_table_rect.irect).is_empty:
                    is_part_of_a_table = True
                    break
        if is_part_of_a_table:
            continue

        is_horizontally_relevant = False
        effective_table_x0 = current_table_bbox.x0 - (current_table_bbox.width * 0.1)
        effective_table_x1 = current_table_bbox.x1 + (current_table_bbox.width * 0.1)

        if max(block_rect.x0, effective_table_x0) < min(
            block_rect.x1, effective_table_x1
        ):
            is_horizontally_relevant = True

        if (
            is_horizontally_relevant
            and current_table_bbox.x1 < page_width * main_content_max_x_ratio
        ):
            if block_rect.x0 >= page_width * (main_content_max_x_ratio - 0.05):
                is_horizontally_relevant = False

        if is_horizontally_relevant:
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


def get_limited_text_from_bottom_of_page(
    page: pymupdf.Page,
    all_table_bboxes_on_page: List[pymupdf.Rect],
    n_tokens: int,
    search_height_ratio: float = 0.25,
    min_search_height_from_bottom: int = 50,
    main_content_max_x_ratio: float = 0.65,
) -> str:
    if n_tokens <= 0:
        return ""
    page_height = page.rect.height
    page_width = page.rect.width
    search_area_actual_height = max(
        page_height * search_height_ratio, min_search_height_from_bottom
    )
    search_rect_y1_for_bottom_area = max(0, page_height - search_area_actual_height)
    search_rect_y0_for_bottom_area = page_height
    if search_rect_y1_for_bottom_area >= search_rect_y0_for_bottom_area:
        return ""

    search_rect = pymupdf.Rect(
        0, search_rect_y1_for_bottom_area, page_width, search_rect_y0_for_bottom_area
    )
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
        for table_bbox_on_this_page in all_table_bboxes_on_page:
            test_table_rect = (
                pymupdf.Rect(table_bbox_on_this_page)
                if isinstance(table_bbox_on_this_page, tuple)
                else table_bbox_on_this_page
            )
            if not test_table_rect.is_empty:
                if not (block_rect.irect & test_table_rect.irect).is_empty:
                    is_part_of_a_table = True
                    break
        if is_part_of_a_table:
            continue

        if block_rect.x0 < page_width * main_content_max_x_ratio:
            cleaned_block_text = block_text_content.replace("\n", " ").strip()
            if cleaned_block_text:
                filtered_text_lines.append(cleaned_block_text)

    text_from_bottom = " ".join(filtered_text_lines)
    text_from_bottom = " ".join(text_from_bottom.split())
    if not text_from_bottom:
        return ""
    tokens = text_from_bottom.split()
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
    look_to_prev_page_threshold_y0: float = 50.0,
) -> Tuple[ExtractedDataType, List[str]]:
    all_dataframes: List[pd.DataFrame] = []
    all_contexts: List[str] = []
    all_df_page_numbers: List[int] = []
    default_extracted_data: ExtractedDataType = {"dataframes": [], "page_numbers": []}

    if not isinstance(doc, pymupdf.Document):
        logging.error("Lỗi: 'doc' phải là một đối tượng pymupdf.Document.")
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
        logging.error(
            "Lỗi: page_numbers_to_process phải là số nguyên hoặc danh sách số nguyên."
        )
        return default_extracted_data, []

    for page_index in pages_to_process_indices:
        if not (0 <= page_index < len(doc)):
            logging.warning(
                f"Cảnh báo: Số trang {page_index + 1} không hợp lệ. Bỏ qua."
            )
            continue

        page = doc[page_index]
        current_page_human_readable = page.number + 1
        page_table_finder = page.find_tables(strategy="lines_strict")
        all_found_table_bboxes_on_current_page: List[pymupdf.Rect] = []
        if page_table_finder.tables:
            all_found_table_bboxes_on_current_page = [
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
                    context_text_current_page = get_limited_text_before_table(
                        page,
                        current_table_actual_bbox,
                        all_found_table_bboxes_on_current_page,
                        n_tokens_context,
                        search_height_multiplier=search_height_multiplier,
                        min_search_height=min_search_height,
                    ).strip()
                    combined_context = context_text_current_page
                    if (
                        not context_text_current_page
                        and current_table_actual_bbox.y0
                        < look_to_prev_page_threshold_y0
                        and page_index > 0
                    ):
                        prev_page_index = page_index - 1
                        prev_page = doc[prev_page_index]
                        prev_page_table_finder = prev_page.find_tables()
                        all_found_table_bboxes_on_prev_page: List[pymupdf.Rect] = []
                        if prev_page_table_finder.tables:
                            all_found_table_bboxes_on_prev_page = [
                                pymupdf.Rect(tbl.bbox)
                                for tbl in prev_page_table_finder.tables
                                if tbl.bbox
                            ]
                        context_from_prev_page_bottom = (
                            get_limited_text_from_bottom_of_page(
                                prev_page,
                                all_found_table_bboxes_on_prev_page,
                                n_tokens_context,
                            ).strip()
                        )
                        if context_from_prev_page_bottom:
                            combined_context = context_from_prev_page_bottom
                    all_contexts.append(combined_context)
                    all_dataframes.append(df_original)
                    all_df_page_numbers.append(current_page_human_readable)
    extracted_data_output: ExtractedDataType = {
        "dataframes": all_dataframes,
        "page_numbers": all_df_page_numbers,
    }
    if not (len(all_dataframes) == len(all_contexts) == len(all_df_page_numbers)):
        logging.error(
            f"Cảnh báo logic nghiêm trọng: Độ dài các list không khớp! DFs: {len(all_dataframes)}, Contexts: {len(all_contexts)}, PageNums: {len(all_df_page_numbers)}"
        )
        return default_extracted_data, []
    return extracted_data_output, all_contexts


def get_column_types(df: pd.DataFrame) -> List[str]:
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
                except AttributeError:
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
    markdown_table = final_df_to_sample.to_markdown(index=False, tablefmt="pipe")
    return markdown_table


def build_llm_prompt(extracted_data: ExtractedDataType, contexts: List[str]) -> str:
    full_prompt = ""
    dataframes = extracted_data["dataframes"]
    for i, df in enumerate(dataframes):
        if df.empty:
            continue
        df_string = get_input_df(df, n_rows=7)
        column_types = get_column_types(df=df)
        n_columns = len(df.columns)
        prompt_part = f"Table {i}:\n"
        if i < len(contexts) and contexts[i]:
            prompt_part += f"Context before table:\n{contexts[i]}\n\n"
        prompt_part += f"{df_string}\nNumber of columns: {n_columns}\nColumn types: {column_types}\n\n"
        full_prompt += prompt_part
    return full_prompt


class ProcessedTableEntry(TypedDict):
    dataframe: pd.DataFrame
    page_numbers: List[int]
    source: str
    associated_contexts: List[str]


def solve_non_header_table(df: pd.DataFrame, target_headers: List[str]) -> pd.DataFrame:
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


def main_concatenation_logic(
    extracted_data: ExtractedDataType,
    contexts: List[str],
    concat_json: Dict[str, Any],
    source_name: str,
) -> List[ProcessedTableEntry]:
    all_original_dataframes = extracted_data["dataframes"]
    all_original_page_numbers = extracted_data["page_numbers"]

    if len(all_original_dataframes) != len(all_original_page_numbers):
        if len(all_original_dataframes) != len(contexts):
            logging.error(
                "Số lượng DataFrame, số lượng page_numbers và/hoặc contexts trong extracted_data không khớp."
            )
            raise ValueError(
                "Số lượng DataFrame, số lượng page_numbers và/hoặc contexts không khớp."
            )

    processed_tables_result: List[ProcessedTableEntry] = []
    processed_df_indices: Set[int] = set()

    if "concatable_tables" in concat_json and concat_json["concatable_tables"]:
        for group_info in concat_json["concatable_tables"]:
            indices_in_group = group_info.get("table_index", [])
            if not indices_in_group:
                continue

            actual_indices_for_this_concat_group: List[int] = []
            group_contexts: List[str] = []

            for idx in indices_in_group:
                if 0 <= idx < len(all_original_dataframes):
                    actual_indices_for_this_concat_group.append(idx)
                    processed_df_indices.add(idx)
                    if idx < len(contexts):
                        group_contexts.append(contexts[idx])
                    else:
                        group_contexts.append("")
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
                        "Item at index used for group is not a DataFrame, skipping."
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
                    "associated_contexts": group_contexts,
                }
                processed_tables_result.append(new_entry)
            elif actual_indices_for_this_concat_group:
                logging.warning(
                    f"Group with original indices {actual_indices_for_this_concat_group} resulted in no DataFrames to concatenate."
                )

    total_num_dataframes = len(all_original_dataframes)
    for i in range(total_num_dataframes):
        if i not in processed_df_indices:
            independent_dataframe = all_original_dataframes[i]
            current_cols_count = (
                len(independent_dataframe.columns)
                if isinstance(independent_dataframe, pd.DataFrame)
                else 0
            )
            target_headers_independent = [str(j) for j in range(current_cols_count)]

            if isinstance(independent_dataframe, pd.DataFrame):
                processed_independent_df = solve_non_header_table(
                    independent_dataframe, target_headers_independent
                )
            else:
                logging.warning(
                    f"Independent item at index {i} is not a DataFrame, creating empty DF."
                )
                processed_independent_df = pd.DataFrame()

            page_number_list = [all_original_page_numbers[i]]

            individual_context: List[str] = []
            if i < len(contexts):
                individual_context.append(contexts[i])
            else:
                individual_context.append("")

            independent_entry: ProcessedTableEntry = {
                "dataframe": processed_independent_df,
                "page_numbers": page_number_list,
                "source": get_pdf_name(source_name),
                "associated_contexts": individual_context,
            }
            processed_tables_result.append(independent_entry)

    return processed_tables_result

def table_to_markdown(
        table: pd.DataFrame,
        name: str,
        page_numbers: List[int],
        associated_contexts: List[str],
    ) -> str:
        markdown_str = f"# {name}\n"

        if associated_contexts:
            filtered_contexts = [
                ctx for ctx in associated_contexts if ctx and ctx.strip()
            ]
            if filtered_contexts:
                markdown_str += "## Associated Context(s) Before Table:\n"
                for ctx_item in filtered_contexts:
                    markdown_str += f"- {ctx_item}\n"
                markdown_str += "\n"

        if len(page_numbers) > 1:
            markdown_str += f"Cross-page table spanning pages: {page_numbers}\n"
        else:
            markdown_str += (
                f"This is a single-page table. Page number: {page_numbers[0]}\n\n"
            )

        markdown_str += table.to_markdown(index=False)
        table_shape = table.shape
        markdown_str += f"\n\nShape: {table_shape}\n"
        return markdown_str
    
    
def get_table_fragments_from_file(
    source_path: str, return_json: bool = False
) -> Union[Tuple[ExtractedDataType, List[str]], Dict[str, List[Dict[str, Any]]]]:
    doc = pymupdf.open(source_path)
    # Extract table fragments and their contexts from PDF
    # ExtractedDataType = {"dataframes": List[pd.DataFrame], "page_numbers": List[int]}
    extracted_data, contexts = extract_tables_and_contexts(doc)

    if return_json:
        source_pdf_name = get_pdf_name(source_path)
        
        json_output: Dict[str, List[Dict[str, Any]]] = {source_pdf_name: []}

        table_fragments_dfs = extracted_data["dataframes"]
        fragment_scalar_page_numbers = extracted_data["page_numbers"]

        for idx, table_df_fragment_original in enumerate(table_fragments_dfs):
            
            if not isinstance(table_df_fragment_original, pd.DataFrame):
                continue

            table_df_fragment = table_df_fragment_original.fillna("")

            current_fragment_page_num_scalar = -1 
            if idx < len(fragment_scalar_page_numbers):
                current_fragment_page_num_scalar = fragment_scalar_page_numbers[idx]
            
            current_fragment_page_numbers_as_list = [current_fragment_page_num_scalar]

            current_fragment_context_str = contexts[idx] if idx < len(contexts) else ""
            associated_contexts_for_markdown = [current_fragment_context_str]

            fragment_name = f"{source_pdf_name}_fragment_{idx}"

            markdown_str = table_to_markdown(
                table_df_fragment,
                fragment_name,
                current_fragment_page_numbers_as_list,
                associated_contexts_for_markdown,
            )

            result_item = {
                "table_content": markdown_str,
                "page_numbers": current_fragment_page_numbers_as_list,
                "source": source_pdf_name,
                "table_idx": idx,
                "associated_contexts": associated_contexts_for_markdown,
            }
            json_output[source_pdf_name].append(result_item)
        
        return json_output
    else:
        return extracted_data, contexts


def process_pdf_file_from_extracted_data(
    extracted_data: ExtractedDataType,
    contexts: List[str],
    return_type: Literal["dataframe", "markdown"] = "dataframe",
    source_name: str = "testing",
) -> List[ProcessedTableEntry]:
    llm_prompt = build_llm_prompt(extracted_data, contexts)
    concat_json = generate(llm_prompt)

    processed_tables_result = main_concatenation_logic(
        extracted_data,
        contexts,
        concat_json,
        source_name,
    )
    if return_type == "markdown":
        for table_entry in processed_tables_result:
            if isinstance(table_entry["dataframe"], pd.DataFrame):
                table_entry["dataframe"] = table_entry["dataframe"].to_markdown(
                    index=False
                )
            else:
                table_entry["dataframe"] = "Error: DataFrame not available"

    return processed_tables_result


def process_pdf_file(
    file_paths: Union[str, List[str]],
    return_type: Literal["dataframe", "markdown"] = "dataframe",
    verbose: Literal[0, 1, 2] = 0,
) -> List[ProcessedTableEntry]:
    final_processed_tables_result: List[ProcessedTableEntry] = []
    final_concat_json_list: list = []
    final_input_tokens: list = []
    final_output_tokens: list = []

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    start_time = time.time()
    for file_path in file_paths:
        if verbose >= 1:
            print(f"Processing file: {get_pdf_name(file_path)}...")

        extracted_data, contexts = get_table_fragments_from_file(file_path)

        llm_prompt = build_llm_prompt(extracted_data, contexts)
        if verbose >= 2:
            print(f"LLM prompt: {llm_prompt}")
        concat_json = generate(llm_prompt)
        if verbose >= 2:
            print(f"Concat JSON: {concat_json}")

        processed_tables_result_single_file = main_concatenation_logic(
            extracted_data,
            contexts,
            concat_json,
            file_path,
        )
        final_processed_tables_result.extend(processed_tables_result_single_file)
        final_concat_json_list.append(concat_json)
        final_input_tokens.append(concat_json.get("input_tokens", 0))
        final_output_tokens.append(concat_json.get("output_tokens", 0))

        if verbose >= 1:
            print(
                f"File: {get_pdf_name(file_path)}, Input tokens: {concat_json.get('input_tokens', 0)}, Output tokens: {concat_json.get('output_tokens', 0)}"
            )

    if verbose >= 1 and final_concat_json_list:
        print(
            f"Model: {final_concat_json_list[0].get('model', 'N/A')}, Total input tokens: {sum(final_input_tokens)}, Total output tokens: {sum(final_output_tokens)}"
        )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.3f} seconds")

    if return_type == "markdown":
        for table_entry in final_processed_tables_result:
            if isinstance(table_entry["dataframe"], pd.DataFrame):
                table_entry["dataframe"] = table_entry["dataframe"].to_markdown(
                    index=False
                )
            else:
                table_entry["dataframe"] = "Error: DataFrame not available"

    return final_processed_tables_result


def get_table_content(
    processed_tables_with_pages: List[ProcessedTableEntry],
) -> Dict[str, List[Dict[str, Any]]]:

    results: Dict[str, List[Dict[str, Any]]] = {}
    sources_ = set(table["source"] for table in processed_tables_with_pages)

    for source_pdf_name in sources_:
        processed_tables_source = [
            table_entry
            for table_entry in processed_tables_with_pages
            if table_entry["source"] == source_pdf_name
        ]

        logging.info(
            f"Source {source_pdf_name} has {len(processed_tables_source)} tables for markdown generation."
        )

        for idx, table_entry_item in enumerate(processed_tables_source):
            table_df = table_entry_item["dataframe"]
            if not isinstance(table_df, pd.DataFrame):
                logging.warning(
                    f"Item for {source_pdf_name}_table_{idx} is not a DataFrame. Skipping markdown generation for this item."
                )
                continue

            table_df = table_df.fillna("")
            page_numbers = table_entry_item["page_numbers"]
            associated_contexts = table_entry_item.get("associated_contexts", [])

            table_name = f"{source_pdf_name}_table_{idx}"

            markdown_str = table_to_markdown(
                table_df, table_name, page_numbers, associated_contexts
            )

            result_item = {
                "table_content": markdown_str,
                "page_numbers": page_numbers,
                "source": source_pdf_name,
                "table_idx": idx,
                "associated_contexts": associated_contexts,
            }
            if source_pdf_name not in results:
                results[source_pdf_name] = []
            results[source_pdf_name].append(result_item)

    return results


def full_pipeline(
    source_path: Union[str, List[str]], verbose: Literal[0, 1, 2] = 0
) -> List[ProcessedTableEntry]:
    if isinstance(source_path, str):
        source_path = [source_path]

    final_processed_tables_result = process_pdf_file(
        file_paths=source_path, verbose=verbose
    )
    table_content = get_table_content(final_processed_tables_result)
    return table_content


if __name__ == "__main__":
    #### Định nghĩa source path, có thể là folder hoặc file pdf
    source_path = "C:/Users/PC/CODE/WDM-AI-TEMIS/notebooks/cross_page_tables/ccc3348504535e22aa44231e57052869 (1).pdf"  # hoặc là [os.listdir(source_path)]

    ### CODE NÀY CHỈ LẤY RA CÁC TABLE FRAGMENTS CÓ CẤU TRÚC NHƯ Logic 666
    json_result = get_table_fragments_from_file(source_path=source_path, return_json=True) # NHỚ RẰNG `return_json=True`
    print(json_result)
    ### CODE NÀY LÀ CHẠY FULL PIPELINE, RA LUÔN TABLE FULL -> logic 666
    # json_result = full_pipeline(source_path=source_path, verbose=1)
    # print(json_result)

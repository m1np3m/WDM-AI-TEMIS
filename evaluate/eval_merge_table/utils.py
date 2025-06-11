# Cần cài đặt các thư viện nếu chưa có:
# pip install python-Levenshtein scikit-learn numpy

import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re
from langchain_google_vertexai import ChatVertexAI
from prompts import EVAL_WDM_PARSER_PROMPT

# --- Các hàm tính độ tương đồng (Giữ nguyên) ---

def normalize_markdown_for_comparison(md_string):
    if not isinstance(md_string, str):
        return ""
    # 1. Thay thế <br> hoặc <br /> bằng một khoảng trắng (hoặc newline nếu bạn muốn giữ cấu trúc nhiều dòng trong ô)
    #    Ở đây, tôi thay bằng khoảng trắng để các ô thành một dòng dài hơn.
    processed_string = re.sub(r'<br\s*/?>', ' ', md_string)

    # 2. (Tùy chọn) Xử lý các ký tự đặc biệt. Ví dụ, nếu PyMuPDF loại bỏ ☑:
    #    Nếu bạn muốn so sánh có phân biệt ☑, bạn cần tìm cách giữ nó trong PyMuPDF output (khó)
    #    Hoặc, loại bỏ nó khỏi cả hai chuỗi để so sánh công bằng hơn (mất thông tin)
    #    processed_string = processed_string.replace("☑", "") # Ví dụ loại bỏ

    # 3. Chuẩn hóa nhiều khoảng trắng thành một khoảng trắng
    processed_string = re.sub(r'\s+', ' ', processed_string)

    # 4. Strip từng dòng và kết hợp lại (quan trọng sau các thay đổi)
    lines = [line.strip() for line in processed_string.split('\n') if line.strip()] # Lọc dòng trống
    return "\n".join(lines)

def normalize_markdown_table_for_comparison(table_str):
    """
    Enhanced normalization specifically for markdown tables.
    """
    if not isinstance(table_str, str):
        return ""
    
    # Apply basic markdown normalization first
    normalized = normalize_markdown_for_comparison(table_str)
    
    lines = normalized.split('\n')
    normalized_lines = []
    
    for line in lines:
        if '|' in line:
            # Normalize table row structure
            # Remove leading/trailing pipes if present
            line = line.strip()
            if line.startswith('|'):
                line = line[1:]
            if line.endswith('|'):
                line = line[:-1]
            
            # Split by pipes and normalize each cell
            cells = [cell.strip() for cell in line.split('|')]
            
            # Skip separator lines (lines with only -, :, and spaces)
            if re.match(r'^[\s\-:]+$', '|'.join(cells)):
                continue
            
            # Reconstruct the line with normalized spacing
            normalized_line = '| ' + ' | '.join(cells) + ' |'
            normalized_lines.append(normalized_line)
        else:
            normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)

def extract_table_content_only(table_str):
    """
    Extract only the data content from markdown table, removing separators and structure.
    """
    if not isinstance(table_str, str):
        return ""
    
    lines = table_str.strip().split('\n')
    content_lines = []
    
    for line in lines:
        if '|' in line:
            line = line.strip()
            # Skip separator lines
            if re.match(r'^[\|\s\-:]+$', line):
                continue
            
            # Extract cell content
            if line.startswith('|'):
                line = line[1:]
            if line.endswith('|'):
                line = line[:-1]
            
            cells = [cell.strip() for cell in line.split('|')]
            content_lines.append(' '.join(cells))
    
    return '\n'.join(content_lines)

def levenshtein_similarity(str1, str2):
    distance = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)

def jaccard_similarity_lines(table1_str, table2_str):
    """
    Tính độ tương đồng Jaccard dựa trên các dòng của hai bảng.
    Đã cải thiện để robust hơn với khoảng trắng ở đầu/cuối dòng và các dòng trống.
    """
    lines1 = {line.strip() for line in table1_str.strip().split('\n') if line.strip()}
    lines2 = {line.strip() for line in table2_str.strip().split('\n') if line.strip()}

    if not lines1 and not lines2:
        return 1.0
    if not lines1 or not lines2:
        return 0.0

    intersection = len(lines1.intersection(lines2))
    union = len(lines1.union(lines2))

    if union == 0: # Về lý thuyết không đạt tới đây nếu đã qua các check trên
        return 1.0 
    return intersection / union

def jaccard_similarity_cells(table1_str, table2_str):
    """
    Jaccard similarity based on table cells content, ignoring structure.
    """
    def extract_cells(table_str):
        cells = set()
        lines = table_str.strip().split('\n')
        for line in lines:
            if '|' in line and not re.match(r'^[\|\s\-:]+$', line):
                line = line.strip()
                if line.startswith('|'):
                    line = line[1:]
                if line.endswith('|'):
                    line = line[:-1]
                for cell in line.split('|'):
                    cell_content = cell.strip()
                    if cell_content:
                        cells.add(cell_content)
        return cells
    
    cells1 = extract_cells(table1_str)
    cells2 = extract_cells(table2_str)
    
    if not cells1 and not cells2:
        return 1.0
    if not cells1 or not cells2:
        return 0.0
    
    intersection = len(cells1.intersection(cells2))
    union = len(cells1.union(cells2))
    
    return intersection / union if union > 0 else 1.0

def calculate_cosine_similarity_for_list(table_list_to_compare, ground_truth_table):
    if not table_list_to_compare:
        return np.array([])

    documents = [ground_truth_table] + table_list_to_compare

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        similarities = []
        for doc_str_in_list in table_list_to_compare:
            gt_is_empty = not ground_truth_table.strip()
            doc_is_empty = not doc_str_in_list.strip()
            if gt_is_empty and doc_is_empty:
                 similarities.append(1.0)
            elif (gt_is_empty and not doc_is_empty) or (not gt_is_empty and doc_is_empty):
                 similarities.append(0.0)
            else:
                 similarities.append(0.0)
        return np.array(similarities)

    if tfidf_matrix.shape[0] < 2 :
        return np.zeros(len(table_list_to_compare))

    ground_truth_vector = tfidf_matrix[0]
    other_vectors = tfidf_matrix[1:]

    if other_vectors.shape[0] == 0:
        return np.zeros(len(table_list_to_compare))

    similarities = cosine_similarity(ground_truth_vector, other_vectors)
    return similarities.flatten()

# --- Hàm Ensemble Chính (Cập nhật với Ngưỡng) ---

def find_most_similar_table_ensemble(label_table_str, list_of_table_strs, verbose=True, min_similarity_threshold=None, use_table_normalization=True):
    """
    Tìm chỉ số của bảng trong list_of_table_strs giống nhất với label_table_str
    sử dụng phương pháp ensemble (trung bình) các điểm tương đồng.
    Nếu không có bảng nào đạt ngưỡng tương đồng tối thiểu, trả về -1.

    Args:
        label_table_str (str): Chuỗi Markdown của bảng ground truth.
        list_of_table_strs (list): Danh sách các chuỗi Markdown của các bảng cần so sánh.
        verbose (bool): Nếu True, in ra các điểm tương đồng chi tiết.
        min_similarity_threshold (float, optional): Ngưỡng tương đồng trung bình tối thiểu
                                                     để một bảng được coi là khớp.
                                                     Nếu None, không áp dụng ngưỡng (trả về chỉ số tốt nhất).
                                                     Mặc định là None.
        use_table_normalization (bool): Whether to use table-specific normalization.

    Returns:
        int: Chỉ số của bảng giống nhất nếu đạt ngưỡng.
             Trả về -1 nếu không có bảng nào đạt ngưỡng, danh sách rỗng, hoặc có lỗi.
    """
    if not all(isinstance(s, str) for s in [label_table_str] + list_of_table_strs):
        if verbose:
            print("Lỗi: Không phải tất cả đầu vào đều là chuỗi. Hãy đảm bảo label và tất cả các bảng trong danh sách là chuỗi.")
        return -1

    if not list_of_table_strs:
        if verbose:
            print("Danh sách bảng để so sánh trống.")
        return -1

    # Apply normalization if requested
    if use_table_normalization:
        normalized_label = normalize_markdown_table_for_comparison(label_table_str)
        normalized_tables = [normalize_markdown_table_for_comparison(table) for table in list_of_table_strs]
    else:
        normalized_label = label_table_str
        normalized_tables = list_of_table_strs

    num_tables = len(normalized_tables)
    levenshtein_scores = np.zeros(num_tables)
    jaccard_scores = np.zeros(num_tables)
    jaccard_cell_scores = np.zeros(num_tables)

    if verbose:
        print("--- Levenshtein Similarity ---")
    for i, table_content in enumerate(normalized_tables):
        sim = levenshtein_similarity(normalized_label, table_content)
        levenshtein_scores[i] = sim
        if verbose:
            print(f"Bảng {i} vs ground truth: {sim:.4f}")
    if num_tables > 0:
        best_levenshtein_idx = np.argmax(levenshtein_scores)
        if verbose:
            print(f"Chỉ số bảng giống nhất (Levenshtein): {best_levenshtein_idx} với điểm {levenshtein_scores[best_levenshtein_idx]:.4f}\n")

    if verbose:
        print("--- Jaccard Similarity (dựa trên các dòng) ---")
    for i, table_content in enumerate(normalized_tables):
        sim = jaccard_similarity_lines(normalized_label, table_content)
        jaccard_scores[i] = sim
        if verbose:
            print(f"Bảng {i} vs ground truth: {sim:.4f}")
    if num_tables > 0:
        best_jaccard_idx = np.argmax(jaccard_scores)
        score_to_display_jaccard = jaccard_scores[best_jaccard_idx] if jaccard_scores.size > 0 else 0.0
        if verbose:
            print(f"Chỉ số bảng giống nhất (Jaccard - dòng): {best_jaccard_idx} với điểm {score_to_display_jaccard:.4f}\n")

    if verbose:
        print("--- Jaccard Similarity (dựa trên nội dung ô) ---")
    for i, table_content in enumerate(list_of_table_strs):  # Use original strings for cell comparison
        sim = jaccard_similarity_cells(label_table_str, table_content)
        jaccard_cell_scores[i] = sim
        if verbose:
            print(f"Bảng {i} vs ground truth: {sim:.4f}")
    if num_tables > 0:
        best_jaccard_cell_idx = np.argmax(jaccard_cell_scores)
        score_to_display_jaccard_cell = jaccard_cell_scores[best_jaccard_cell_idx] if jaccard_cell_scores.size > 0 else 0.0
        if verbose:
            print(f"Chỉ số bảng giống nhất (Jaccard - ô): {best_jaccard_cell_idx} với điểm {score_to_display_jaccard_cell:.4f}\n")

    if verbose:
        print("--- Cosine Similarity (TF-IDF) ---")
    cosine_scores = calculate_cosine_similarity_for_list(normalized_tables, normalized_label)
    if cosine_scores.size == num_tables:
        for i, score in enumerate(cosine_scores):
            if verbose:
                print(f"Bảng {i} vs ground truth: {score:.4f}")
        best_cosine_idx = np.argmax(cosine_scores) if cosine_scores.size > 0 else 0
        score_to_display_cosine = cosine_scores[best_cosine_idx] if cosine_scores.size > 0 else 0.0
        if verbose:
             print(f"Chỉ số bảng giống nhất (Cosine TF-IDF): {best_cosine_idx} với điểm {score_to_display_cosine:.4f}\n")
    else:
        if verbose:
            print("Không thể tính toán Cosine scores một cách chính xác, sử dụng điểm 0 cho ensemble.\n")
        cosine_scores = np.zeros(num_tables)

    if not (len(levenshtein_scores) == num_tables and \
            len(jaccard_scores) == num_tables and \
            len(jaccard_cell_scores) == num_tables and \
            len(cosine_scores) == num_tables):
        if verbose:
            print("Lỗi: Kích thước của các mảng điểm không khớp để tính trung bình.")
        return -1

    # Use 4 metrics now instead of 3
    average_scores = (levenshtein_scores + jaccard_scores + jaccard_cell_scores + cosine_scores) / 4.0
    
    if verbose:
        print("--- Điểm Trung Bình ---")
        for i, avg_score in enumerate(average_scores):
            print(f"Bảng {i} - Điểm trung bình: {avg_score:.4f}")

    if average_scores.size > 0:
        best_average_idx = np.argmax(average_scores)
        highest_avg_score = average_scores[best_average_idx]

        if verbose:
            print(f"\nChỉ số bảng có điểm trung bình cao nhất: {best_average_idx} với điểm trung bình {highest_avg_score:.4f}")
            if min_similarity_threshold is not None:
                 print(f"Ngưỡng tương đồng tối thiểu yêu cầu: {min_similarity_threshold:.4f}")

        if min_similarity_threshold is not None:
            if highest_avg_score >= min_similarity_threshold:
                if verbose:
                    print(f"Điểm trung bình cao nhất ({highest_avg_score:.4f}) >= ngưỡng ({min_similarity_threshold:.4f}). Trả về chỉ số.")
                return int(best_average_idx)
            else:
                if verbose:
                    print(f"Điểm trung bình cao nhất ({highest_avg_score:.4f}) < ngưỡng ({min_similarity_threshold:.4f}). Không có bảng nào đủ giống. Trả về -1.")
                return -1
        else: # Không có ngưỡng, trả về chỉ số tốt nhất
            if verbose:
                print("Không có ngưỡng tương đồng tối thiểu nào được áp dụng. Trả về chỉ số tốt nhất.")
            return int(best_average_idx)
    else:
        if verbose:
            print("\nKhông thể tính điểm trung bình tổng hợp hoặc không có bảng nào để đánh giá.")
        return -1


# --- LLM-based Table Evaluation ---

def evaluate_table_similarity(ground_truth_table: str, table_to_evaluate: str) -> float:
    """
    Evaluate table similarity using LLM and return a float score between 0.0 and 1.0.
    
    Args:
        ground_truth_table (str): The ground truth markdown table
        table_to_evaluate (str): The extracted table to evaluate
        
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    try:
        prompt = EVAL_WDM_PARSER_PROMPT.format(
            ground_truth_table=ground_truth_table, 
            extracted_table=table_to_evaluate
        )
        
        model = ChatVertexAI(model="gemini-2.0-flash-exp", temperature=0.0)
        response = model.invoke(prompt)
        
        # Extract score from response
        response_text = response.content
        
        # Look for SIMILARITY_SCORE: pattern
        import re
        score_match = re.search(r'SIMILARITY_SCORE:\s*([0-9]*\.?[0-9]+)', response_text)
        
        if score_match:
            score = float(score_match.group(1))
            # Ensure score is within valid range
            return max(0.0, min(1.0, score))
        else:
            # Fallback: try to find any number between 0 and 1
            numbers = re.findall(r'\b0\.[0-9]+\b|\b1\.0\b|\b0\b|\b1\b', response_text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            else:
                print(f"Warning: Could not extract score from response: {response_text[:200]}...")
                return 0.0
                
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        # Fallback to ensemble method
        ensemble_score = (
            levenshtein_similarity(ground_truth_table, table_to_evaluate) +
            jaccard_similarity_lines(ground_truth_table, table_to_evaluate) +
            jaccard_similarity_cells(ground_truth_table, table_to_evaluate)
        ) / 3.0
        return ensemble_score
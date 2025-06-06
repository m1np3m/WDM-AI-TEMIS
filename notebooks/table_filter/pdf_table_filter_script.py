import os
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import numpy as np
from transformers import pipeline
import torch
import cv2
import camelot
import tabula
from typing import List, Tuple, Optional
import logging
import argparse # Để nhận tham số từ dòng lệnh

# --- THIẾT LẬP LOGGING BAN ĐẦU ---
# Sẽ được cấu hình lại dựa trên tham số dòng lệnh
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)


class PDFTableDetector:
    def __init__(self, method='hybrid', vision_threshold=0.85, dpi=288,
                 vision_high_confidence_threshold=0.95,
                 vision_medium_confidence_lower_threshold=0.75):
        """
        Khởi tạo detector với các phương pháp khác nhau

        Args:
            method: 'vision_model', 'traditional', 'hybrid'
            vision_threshold: Ngưỡng tin cậy chung cho vision model (quyết định có phát hiện hay không trong detect_tables_vision_model)
            dpi: Độ phân giải ảnh khi chuyển từ PDF sang image
            vision_high_confidence_threshold: Ngưỡng cao để tin tưởng tuyệt đối Vision Model
            vision_medium_confidence_lower_threshold: Ngưỡng dưới cho mức tin cậy trung bình của Vision Model
        """
        self.method = method
        self.vision_threshold = vision_threshold
        self.dpi = dpi
        self.matrix = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        self.table_detector = None
        self.VISION_HIGH_CONFIDENCE_THRESHOLD = vision_high_confidence_threshold
        self.VISION_MEDIUM_CONFIDENCE_LOWER_THRESHOLD = vision_medium_confidence_lower_threshold

        if method in ['vision_model', 'hybrid']:
            try:
                pipeline_device = 0 if torch.cuda.is_available() else -1
                self.table_detector = pipeline(
                    "object-detection",
                    model="microsoft/table-transformer-detection",
                    device=pipeline_device
                )
                device_name = "GPU" if pipeline_device == 0 else "CPU"
                logger.info(f"✅ Vision model 'microsoft/table-transformer-detection' đã tải thành công trên {device_name}.")
            except Exception as e:
                logger.warning(f"⚠️ Không thể tải vision model: {e}", exc_info=False)
                if self.method == 'vision_model':
                    raise RuntimeError(f"Không thể tải vision model và phương pháp là 'vision_model'. Lỗi: {e}")
                if self.method == 'hybrid':
                    logger.info("Do vision model lỗi, phương pháp hybrid sẽ cố gắng hoạt động như traditional.")

    def pdf_to_images(self, pdf_path: str) -> List[Tuple[int, Image.Image]]:
        """Chuyển đổi PDF thành images"""
        doc = fitz.open(pdf_path)
        images = []
        logger.info(f"Chuyển đổi {len(doc)} trang từ PDF '{os.path.basename(pdf_path)}' sang ảnh với DPI={self.dpi}...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=self.matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((page_num, img))
        doc.close()
        logger.info(f"Đã chuyển đổi xong {len(images)} ảnh.")
        return images

    def detect_tables_vision_model(self, image: Image.Image, page_num_for_log: int) -> bool:
        """
        Sử dụng vision model để detect bảng.
        Hàm này trả về True/False dựa trên self.vision_threshold chung.
        Logic kết hợp trong has_table_on_page sẽ xem xét score chi tiết hơn.
        """
        if not self.table_detector:
            logger.warning(f"P{page_num_for_log+1}: Vision model không khả dụng để detect.")
            return False
        try:
            results = self.table_detector(image)
            tables = [r for r in results if r['label'] in ['table', 'table rotated'] and r['score'] > self.vision_threshold]
            return len(tables) > 0
        except Exception as e:
            logger.error(f"P{page_num_for_log+1} Vision (Base Check): Lỗi khi detect: {e}", exc_info=False)
            return False

    def detect_tables_traditional(self, pdf_path: str, page_num: int) -> bool:
        """Sử dụng phương pháp traditional (Camelot & Tabula) để detect bảng"""
        page_str_for_libs = str(page_num + 1)
        try:
            tables_lattice = camelot.read_pdf(
                pdf_path, pages=page_str_for_libs, flavor='lattice',
                suppress_stdout=True, line_scale=40
            )
            if tables_lattice.n > 0:
                for table in tables_lattice:
                    if table.parsing_report['accuracy'] > 70 and table.df.shape[0] >= 2 and table.df.shape[1] >=2:
                        logger.debug(f"P{page_num+1} Camelot(lattice): OK, accuracy {table.parsing_report['accuracy']:.2f}, shape {table.df.shape}")
                        return True

            tables_stream = camelot.read_pdf(
                pdf_path, pages=page_str_for_libs, flavor='stream',
                suppress_stdout=True, edge_tol=500
            )
            if tables_stream.n > 0:
                for table in tables_stream:
                    if table.parsing_report['accuracy'] > 60 and table.df.shape[0] >= 2 and table.df.shape[1] >=2:
                        logger.debug(f"P{page_num+1} Camelot(stream): OK, accuracy {table.parsing_report['accuracy']:.2f}, shape {table.df.shape}")
                        return True
        except Exception as e:
            logger.warning(f"P{page_num+1} Camelot: Lỗi - {e}", exc_info=False)

        try:
            tables_tabula = tabula.read_pdf(
                pdf_path, pages=page_str_for_libs, multiple_tables=True, silent=True,
            )
            if tables_tabula and len(tables_tabula) > 0:
                for table_df in tables_tabula:
                    if isinstance(table_df, pd.DataFrame) and not table_df.empty:
                        if table_df.shape[0] >= 2 and table_df.shape[1] >= 2 and table_df.notna().sum().sum() > max(table_df.shape[0], table_df.shape[1]):
                            logger.debug(f"P{page_num+1} Tabula: OK, shape {table_df.shape}, non-NA cells {table_df.notna().sum().sum()}")
                            return True
        except Exception as e:
            logger.warning(f"P{page_num+1} Tabula: Lỗi - {e}", exc_info=False)
        return False

    def detect_table_opencv(self, image: Image.Image, page_num_for_log: int) -> bool:
        """Sử dụng OpenCV để detect bảng dựa trên đường kẻ. Chỉ hiệu quả với bảng có đường kẻ rõ."""
        try:
            img_cv = np.array(image.convert('L'))
            thresh = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            horizontal_kernel_size = max(15, int(img_cv.shape[1] / 50))
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
            detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            cnts_h, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            vertical_kernel_size = max(15, int(img_cv.shape[0] / 50))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
            detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            cnts_v, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(cnts_h) >= 2 and len(cnts_v) >= 2:
                 logger.debug(f"P{page_num_for_log+1} OpenCV: Tìm thấy {len(cnts_h)} ngang, {len(cnts_v)} dọc.")
                 return True
            return False
        except Exception as e:
            logger.error(f"P{page_num_for_log+1} OpenCV: Lỗi - {e}", exc_info=False)
            return False

    def _get_image_for_page(self, pdf_path: str, page_num: int, images_cache: List[Tuple[int, Image.Image]]) -> Optional[Image.Image]:
        """Lấy ảnh từ cache hoặc load on-demand nếu cần."""
        img_tuple = next((item for item in images_cache if item[0] == page_num), None)
        if img_tuple:
            return img_tuple[1]

        logger.info(f"P{page_num+1}: Không có ảnh pre-load, thử load on-demand...")
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=self.matrix)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            images_cache.append((page_num, image)) # Thêm vào cache để lần sau dùng
            logger.info(f"P{page_num+1}: Đã load ảnh on-demand.")
            return image
        except Exception as e:
            logger.error(f"P{page_num+1}: Không thể load ảnh on-demand: {e}", exc_info=False)
            return None

    def has_table_on_page(self, pdf_path: str, page_num: int, images_cache: List[Tuple[int, Image.Image]]) -> bool:
        """
        Kiểm tra trang có bảng hay không, sử dụng chiến lược kết hợp nâng cao.
        """
        page_image: Optional[Image.Image] = None
        detection_results = {"vision": None, "traditional": None, "opencv": None}
        method_executed = {"vision": False, "traditional": False, "opencv": False}
        log_prefix = f"P{page_num+1} CombinedLogic:"

        # --- Bước 1: Ưu tiên Vision Model ---
        if self.method in ['vision_model', 'hybrid'] and self.table_detector:
            page_image = self._get_image_for_page(pdf_path, page_num, images_cache)
            method_executed["vision"] = True
            if page_image:
                try:
                    raw_vision_results = self.table_detector(page_image)
                    relevant_tables = [r for r in raw_vision_results if r['label'] in ['table', 'table rotated']]
                    if relevant_tables:
                        max_score = max(r['score'] for r in relevant_tables)
                        detection_results["vision"] = max_score
                        score_strings = [f"{r['score']:.2f}" for r in relevant_tables]
                        scores_str_representation = ", ".join(score_strings)
                        logger.info(f"{log_prefix} Vision raw scores: [{scores_str_representation}], Max: {max_score:.2f}")
                    else:
                        detection_results["vision"] = 0.0
                        logger.info(f"{log_prefix} Vision: No 'table' or 'table rotated' objects found.")
                except Exception as e:
                    logger.error(f"{log_prefix} Vision: Error during detection - {e}", exc_info=False)
                    detection_results["vision"] = -1 # Lỗi
            else:
                logger.warning(f"{log_prefix} Vision: No image available.")
                detection_results["vision"] = -1 # Không có ảnh
        else: # Vision không được kích hoạt hoặc không có detector
            method_executed["vision"] = False
            detection_results["vision"] = -1 # -1 để chỉ trạng thái không chạy/lỗi

        # --- Logic kết hợp dựa trên kết quả Vision ---
        if detection_results["vision"] is not None and detection_results["vision"] >= self.VISION_HIGH_CONFIDENCE_THRESHOLD:
            logger.info(f"{log_prefix} Decision: Vision model HIGH confidence ({detection_results['vision']:.2f}). ✅ Has table.")
            return True

        needs_confirmation = False
        if detection_results["vision"] is not None and \
           self.VISION_MEDIUM_CONFIDENCE_LOWER_THRESHOLD <= detection_results["vision"] < self.VISION_HIGH_CONFIDENCE_THRESHOLD:
            logger.info(f"{log_prefix} Vision MEDIUM confidence ({detection_results['vision']:.2f}). Needs confirmation.")
            needs_confirmation = True
        elif method_executed["vision"] and detection_results["vision"] != -1 : # Vision đã chạy và không lỗi, nhưng score thấp
             logger.info(f"{log_prefix} Vision LOW confidence or NO detection ({detection_results['vision']:.2f}). Relying on other methods.")


        run_traditional = False
        if self.method == 'traditional':
            run_traditional = True
        elif self.method == 'hybrid':
            # Chạy traditional nếu vision lỗi, hoặc vision có score thấp hơn high_conf
            if detection_results["vision"] == -1: # Vision lỗi hoặc không chạy
                run_traditional = True
            elif detection_results["vision"] < self.VISION_HIGH_CONFIDENCE_THRESHOLD :
                run_traditional = True

        if run_traditional:
            detection_results["traditional"] = self.detect_tables_traditional(pdf_path, page_num)
            method_executed["traditional"] = True
            logger.info(f"{log_prefix} Traditional result: {detection_results['traditional']}")
            if needs_confirmation and detection_results["traditional"]:
                logger.info(f"{log_prefix} Decision: Vision MEDIUM + Traditional CONFIRMED. ✅ Has table.")
                return True
            if not needs_confirmation and \
               detection_results["vision"] is not None and \
               detection_results["vision"] < self.VISION_MEDIUM_CONFIDENCE_LOWER_THRESHOLD and \
               detection_results["traditional"]:
                logger.info(f"{log_prefix} Decision: Vision LOW/NO + Traditional POSITIVE. ✅ Has table.")
                return True

        run_opencv = False
        if self.method == 'hybrid':
            # Chạy OpenCV nếu:
            # 1. Vision ở mức medium và traditional không tìm thấy
            # 2. Vision ở mức low/no (hoặc lỗi) và traditional cũng không tìm thấy
            if needs_confirmation and not detection_results.get("traditional", False):
                run_opencv = True
            elif not needs_confirmation and \
                  (detection_results["vision"] == -1 or (detection_results["vision"] is not None and detection_results["vision"] < self.VISION_MEDIUM_CONFIDENCE_LOWER_THRESHOLD)) and \
                  not detection_results.get("traditional", False):
                run_opencv = True

        if run_opencv:
            if not page_image: # Load ảnh nếu chưa có (ví dụ vision không chạy)
                  page_image = self._get_image_for_page(pdf_path, page_num, images_cache)

            if page_image:
                detection_results["opencv"] = self.detect_table_opencv(page_image, page_num)
                method_executed["opencv"] = True
                logger.info(f"{log_prefix} OpenCV result: {detection_results['opencv']}")

                if needs_confirmation and not detection_results.get("traditional", False) and detection_results["opencv"]:
                    logger.info(f"{log_prefix} Decision: Vision MEDIUM + Traditional NEGATIVE + OpenCV CONFIRMED. ✅ Has table.")
                    return True
                if not needs_confirmation and \
                    (detection_results["vision"] == -1 or (detection_results["vision"] is not None and detection_results["vision"] < self.VISION_MEDIUM_CONFIDENCE_LOWER_THRESHOLD)) and \
                    not detection_results.get("traditional", False) and \
                    detection_results["opencv"]:
                    logger.info(f"{log_prefix} Decision: Vision LOW/NO + Traditional NEGATIVE + OpenCV POSITIVE. ✅ Has table.")
                    return True
            else:
                logger.warning(f"{log_prefix} OpenCV: No image available.")
                method_executed["opencv"] = False # Đánh dấu là không thực thi được


        # --- Bước cuối: Voting nếu là hybrid và các quyết định trên chưa return True ---
        # Logic voting này có thể cần xem xét lại nếu các bước trên đã xử lý hầu hết các trường hợp.
        # Hiện tại, nó có thể không cần thiết nếu logic if/else ở trên đã đủ bao quát.
        # Hoặc, chúng ta có thể đơn giản hóa nó.
        if self.method == 'hybrid':
            # Chỉ tính vote nếu phương pháp đó đã được thực thi và cho kết quả dương tính
            # Vision vote: Nếu score >= medium lower (hoặc nếu bạn muốn ngưỡng khác cho vote)
            vision_vote = method_executed["vision"] and detection_results["vision"] is not None and \
                          detection_results["vision"] >= self.VISION_MEDIUM_CONFIDENCE_LOWER_THRESHOLD
            traditional_vote = method_executed["traditional"] and detection_results.get("traditional", False)
            opencv_vote = method_executed["opencv"] and detection_results.get("opencv", False)
            
            true_votes = sum([vision_vote, traditional_vote, opencv_vote])
            
            # Đếm số phương pháp đã thực sự chạy (không bị lỗi, có ảnh, v.v.)
            # Điều này quan trọng để tránh trường hợp 1/1 vote cũng pass
            methods_actually_run = sum([
                method_executed["vision"] and detection_results["vision"] != -1,
                method_executed["traditional"], # Giả sử traditional luôn chạy nếu được gọi
                method_executed["opencv"] and page_image is not None # OpenCV chỉ chạy nếu có ảnh
            ])

            MIN_VOTES_REQUIRED = 2 
            # Yêu cầu ít nhất 2 phương pháp chạy để áp dụng voting, hoặc nếu chỉ 1 phương pháp chạy thì nó phải dương tính
            # (Logic này đã được xử lý ở các bước trên nếu chỉ 1 phương pháp chạy và dương tính)

            if methods_actually_run >= 2 and true_votes >= MIN_VOTES_REQUIRED :
                logger.info(f"{log_prefix} Decision: Voting passed ({true_votes}/{methods_actually_run} votes). ✅ Has table.")
                return True
            # Trường hợp đặc biệt: Nếu chỉ có 1 phương pháp được thực thi thành công (ví dụ chỉ traditional chạy)
            # và nó cho kết quả dương tính, thì các nhánh if ở trên đã return True rồi.
            # Nếu chỉ vision chạy và high_conf -> return True.
            # Nếu chỉ vision chạy và medium_conf -> needs_confirmation -> các phương pháp khác chạy (hoặc không)
            # Nếu chỉ vision chạy và low_conf -> dựa vào pp khác -> các phương pháp khác chạy (hoặc không)

        logger.info(f"{log_prefix} Decision: No method or combination confirmed a table. ❌ No table.")
        return False


    def filter_pdf_pages_with_tables(self, input_pdf: str, output_pdf: str) -> List[int]:
        """Lọc và tạo PDF mới chỉ chứa các trang có bảng"""
        logger.info(f"🔍 Bắt đầu phân tích file: '{os.path.basename(input_pdf)}'")
        logger.info(f"📄 Phương pháp: {self.method}, Vision Threshold (base): {self.vision_threshold}, DPI: {self.dpi}")
        logger.info(f"📄 Vision High Conf: {self.VISION_HIGH_CONFIDENCE_THRESHOLD}, Vision Medium Lower Conf: {self.VISION_MEDIUM_CONFIDENCE_LOWER_THRESHOLD}")

        images_cache: List[Tuple[int, Image.Image]] = []

        if self.method in ['vision_model', 'hybrid'] and self.table_detector:
            try:
                images_cache = self.pdf_to_images(input_pdf)
            except Exception as e:
                logger.error(f"Lỗi nghiêm trọng khi chuyển PDF sang ảnh hàng loạt: {e}", exc_info=False)
                if self.method == 'vision_model':
                     logger.error("Vision model mode: không thể xử lý nếu không có ảnh.")
                     return [] # Trả về list rỗng nếu không thể tạo ảnh cho vision model

        doc = fitz.open(input_pdf)
        total_pages = len(doc)
        pages_with_tables_indices: List[int] = []

        logger.info(f"📋 Tổng số trang: {total_pages}. Kiểm tra từng trang...")

        for page_num in range(total_pages):
            log_prefix_page = f"   Trang {page_num + 1}/{total_pages}"
            logger.info(f"{log_prefix_page} - Đang xử lý...")

            try:
                has_table = self.has_table_on_page(input_pdf, page_num, images_cache)
                if has_table:
                    pages_with_tables_indices.append(page_num)
                    logger.info(f"{log_prefix_page} - ✅ Có bảng")
                else:
                    logger.info(f"{log_prefix_page} - ❌ Không có bảng")
            except Exception as e: # Bắt lỗi chung cho việc xử lý một trang
                logger.error(f"{log_prefix_page} - Lỗi khi xử lý trang: {e}", exc_info=True)
                # Quyết định xem có nên bỏ qua trang này hay dừng cả file

        if pages_with_tables_indices:
            new_doc = fitz.open()
            for page_idx in pages_with_tables_indices:
                new_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)

            try:
                new_doc.save(output_pdf)
                logger.info(f"\\n🎉 Hoàn thành file '{os.path.basename(input_pdf)}'!")
                logger.info(f"📊 Tìm thấy {len(pages_with_tables_indices)} trang có bảng: {[p+1 for p in pages_with_tables_indices]}")
                logger.info(f"💾 Đã lưu file mới vào: '{output_pdf}'")
            except Exception as e:
                logger.error(f"Lỗi khi lưu file PDF output '{output_pdf}': {e}", exc_info=True)
                # Nếu không lưu được, thì coi như file này không xử lý thành công
                pages_with_tables_indices = [] # Reset để không báo cáo là đã xử lý thành công
            finally:
                new_doc.close()
        else:
            logger.info(f"\\n🤷 Không tìm thấy trang nào có bảng trong tài liệu '{os.path.basename(input_pdf)}'!")

        doc.close()
        return pages_with_tables_indices


def main():
    parser = argparse.ArgumentParser(description="Lọc các trang PDF chứa bảng.")
    parser.add_argument("input_dir", help="Thư mục chứa các file PDF đầu vào.")
    parser.add_argument("output_dir", help="Thư mục để lưu các file PDF đã được lọc.")
    parser.add_argument(
        "--method",
        choices=['vision_model', 'traditional', 'hybrid'],
        default='hybrid',
        help="Phương pháp phát hiện bảng (mặc định: hybrid)."
    )
    parser.add_argument(
        "--vision_threshold",
        type=float,
        default=0.85, # Giá trị này được dùng trong detect_tables_vision_model cho quyết định ban đầu
        help="Ngưỡng tin cậy chung cho vision model (0.0-1.0, mặc định: 0.85)."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=216,
        help="Độ phân giải ảnh (DPI) khi chuyển PDF sang ảnh (mặc định: 216)."
    )
    parser.add_argument(
        "--vision_high_conf",
        type=float,
        default=0.95,
        help="Ngưỡng tin cậy CAO cho vision model trong logic kết hợp (mặc định: 0.95)."
    )
    parser.add_argument(
        "--vision_medium_lower_conf",
        type=float,
        default=0.75,
        help="Ngưỡng tin cậy TRUNG BÌNH (cận dưới) cho vision model trong logic kết hợp (mặc định: 0.75)."
    )
    parser.add_argument(
        "--log_level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Mức độ log (mặc định: INFO)."
    )

    args = parser.parse_args()

    # --- CẤU HÌNH LẠI LOGGING DỰA TRÊN THAM SỐ ---
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        logging.warning(f"Mức log '{args.log_level}' không hợp lệ. Đặt về INFO.")
        numeric_log_level = logging.INFO
    
    # Cập nhật root logger
    logging.getLogger().setLevel(numeric_log_level)
    # Nếu muốn log ra file, có thể thêm FileHandler ở đây
    # Ví dụ:
    # file_handler = logging.FileHandler("pdf_filter_script.log")
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'))
    # logging.getLogger().addHandler(file_handler)

    logger.info(f"ℹ️ Mức log được đặt thành: {args.log_level}")


    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- KHỞI TẠO DETECTOR ---
    try:
        detector = PDFTableDetector(
            method=args.method,
            vision_threshold=args.vision_threshold,
            dpi=args.dpi,
            vision_high_confidence_threshold=args.vision_high_conf,
            vision_medium_confidence_lower_threshold=args.vision_medium_lower_conf
        )
        logger.info(f"✅ Khởi tạo PDFTableDetector thành công với phương pháp '{args.method}'.")
    except RuntimeError as e:
        logger.error(f"LỖI RUNTIME NGHIÊM TRỌNG KHI KHỞI TẠO DETECTOR: {e}", exc_info=False)
        print(f"🚫 Đã xảy ra lỗi nghiêm trọng khi khởi tạo detector. Vui lòng kiểm tra log.")
        return # Thoát sớm
    except Exception as e:
        logger.error(f"LỖI KHÔNG XÁC ĐỊNH KHI KHỞI TẠO DETECTOR: {e}", exc_info=True)
        print(f"🚫 Đã có lỗi xảy ra khi khởi tạo detector. Vui lòng kiểm tra log.")
        return # Thoát sớm

    # --- XỬ LÝ HÀNG LOẠT FILE PDF ---
    pdf_files_to_process = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files_to_process:
        logger.info(f"Không tìm thấy file PDF nào trong thư mục: '{args.input_dir}'. Vui lòng thêm file PDF và chạy lại.")
        print(f"ℹ️ Không tìm thấy file PDF nào trong thư mục '{args.input_dir}'.")
    else:
        logger.info(f"Tìm thấy {len(pdf_files_to_process)} file PDF trong '{args.input_dir}'. Bắt đầu xử lý...")
        print(f"🚀 Tìm thấy {len(pdf_files_to_process)} file PDF. Bắt đầu xử lý...")

        total_files = len(pdf_files_to_process)
        successful_files = 0
        failed_files = 0

        for i, pdf_filename in enumerate(pdf_files_to_process):
            current_file_num = i + 1
            logger.info(f"--- [{current_file_num}/{total_files}] Đang xử lý file: {pdf_filename} ---")
            print(f"\n--- [{current_file_num}/{total_files}] Đang xử lý file: {pdf_filename} ---")

            input_pdf_path = os.path.join(args.input_dir, pdf_filename)
            
            base, ext = os.path.splitext(pdf_filename)
            output_pdf_filename = f"{base}_tables_only{ext}"
            output_pdf_path = os.path.join(args.output_dir, output_pdf_filename)

            try:
                pages_found_indices = detector.filter_pdf_pages_with_tables(input_pdf_path, output_pdf_path)

                if pages_found_indices and os.path.exists(output_pdf_path):
                    logger.info(f"[{current_file_num}/{total_files}] ✅ Xử lý thành công '{pdf_filename}'. {len(pages_found_indices)} trang có bảng được lưu vào '{output_pdf_path}'")
                    print(f"[{current_file_num}/{total_files}] ✅ '{pdf_filename}' -> '{output_pdf_filename}' ({len(pages_found_indices)} trang)")
                    successful_files +=1
                elif not pages_found_indices:
                    logger.info(f"[{current_file_num}/{total_files}] ℹ️ Không tìm thấy trang nào chứa bảng trong '{pdf_filename}' theo các tiêu chí.")
                    print(f"[{current_file_num}/{total_files}] ℹ️ Không có bảng trong '{pdf_filename}'.")
                    # Vẫn tính là thành công vì script đã chạy qua file mà không lỗi
                    successful_files +=1 
                else:
                     logger.warning(f"[{current_file_num}/{total_files}] ⚠️  Có vẻ đã xảy ra lỗi với file '{pdf_filename}', file output không được tạo ra dù có thể đã tìm thấy trang.")
                     print(f"[{current_file_num}/{total_files}] ⚠️ Lỗi không rõ với '{pdf_filename}'.")
                     failed_files +=1

            except RuntimeError as e:
                logger.error(f"[{current_file_num}/{total_files}] LỖI RUNTIME khi xử lý '{pdf_filename}': {e}", exc_info=False)
                print(f"[{current_file_num}/{total_files}] 🚫 Lỗi Runtime khi xử lý '{pdf_filename}'. Bỏ qua file này.")
                failed_files +=1
            except Exception as e:
                logger.error(f"[{current_file_num}/{total_files}] LỖI KHÔNG XÁC ĐỊNH khi xử lý '{pdf_filename}': {e}", exc_info=True)
                print(f"[{current_file_num}/{total_files}] 🚫 Lỗi không xác định khi xử lý '{pdf_filename}'. Bỏ qua file này.")
                failed_files +=1
            
            logger.info(f"--- [{current_file_num}/{total_files}] Kết thúc xử lý file: {pdf_filename} ---\n")

        logger.info("🎉🎉🎉 Hoàn tất xử lý tất cả các file! 🎉🎉🎉")
        logger.info(f"Tổng kết: {successful_files} file xử lý thành công (hoặc không có bảng), {failed_files} file lỗi.")
        print(f"\n🎉 Hoàn tất xử lý tất cả các file!")
        print(f"Tổng kết: {successful_files} file xử lý thành công (hoặc không có bảng), {failed_files} file lỗi.")

if __name__ == "__main__":
    main()
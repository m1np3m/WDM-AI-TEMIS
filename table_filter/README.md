# PDF Table Page Filter

Công cụ này được thiết kế để xử lý hàng loạt các tệp PDF, tự động xác định các trang chứa bảng và tạo ra các tệp PDF mới chỉ bao gồm những trang đó. Nó sử dụng kết hợp các mô hình học sâu (Vision Model), các thư viện xử lý PDF truyền thống (Camelot, Tabula) và kỹ thuật xử lý ảnh (OpenCV) để đạt được độ chính xác cao.

## Tính năng

*   **Phát hiện bảng đa phương pháp:**
    *   **Vision Model:** Sử dụng mô hình `microsoft/table-transformer-detection` để nhận diện bảng dựa trên hình ảnh.
    *   **Traditional Methods:** Tận dụng Camelot (cả `lattice` và `stream`) và Tabula để trích xuất bảng dựa trên cấu trúc PDF.
    *   **OpenCV:** Phát hiện các đường kẻ ngang và dọc để xác định bảng có cấu trúc rõ ràng.
    *   **Hybrid Mode:** Kết hợp thông minh kết quả từ tất cả các phương pháp trên để đưa ra quyết định cuối cùng, có tính đến độ tin cậy của từng phương pháp.
*   **Xử lý hàng loạt:** Có khả năng xử lý tất cả các tệp PDF trong một thư mục đầu vào được chỉ định.
*   **Đầu ra có tổ chức:** Lưu các tệp PDF đã được lọc vào một thư mục đầu ra riêng biệt, giữ nguyên tên tệp gốc với hậu tố `_tables_only`.
*   **Cấu hình linh hoạt:** Cho phép tùy chỉnh các tham số quan trọng thông qua dòng lệnh, bao gồm:
    *   Phương pháp phát hiện.
    *   Ngưỡng tin cậy cho mô hình vision.
    *   Độ phân giải DPI cho việc chuyển đổi PDF sang ảnh.
    *   Mức độ chi tiết của log.
*   **Logging chi tiết:** Cung cấp thông tin log về quá trình xử lý từng tệp và từng trang.

## Yêu cầu hệ thống

*   Python 3.8+
*   Ghostscript (thường cần thiết cho Camelot, đặc biệt trên Windows)
*   Nếu sử dụng Vision Model với GPU:
    *   NVIDIA GPU với CUDA Toolkit được cài đặt.
    *   cuDNN (nếu được yêu cầu bởi phiên bản PyTorch của bạn).

## Cài đặt

1.  **Clone repository (nếu có):**
    ```bash
    # git clone <your-repository-url>
    # cd <repository-name>
    ```

2.  **Tạo và kích hoạt môi trường ảo (khuyến nghị):**
    ```bash
    python -m venv venv
    ```
    *   Trên Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
    *   Trên Windows:
        ```bash
        venv\Scripts\activate
        ```

3.  **Cài đặt các thư viện cần thiết:**
    Đảm bảo bạn có file `requirements.txt` trong thư mục dự án.
    ```bash
    pip install -r requirements.txt
    ```
    *Lưu ý:* Lần đầu tiên chạy script với phương pháp `vision_model` hoặc `hybrid`, mô hình `microsoft/table-transformer-detection` sẽ được tải xuống từ Hugging Face Hub. Quá trình này có thể mất một chút thời gian và yêu cầu kết nối internet.

4.  **(Tùy chọn - Windows) Cài đặt Ghostscript:**
    Nếu bạn gặp lỗi liên quan đến Ghostscript khi sử dụng Camelot, hãy tải và cài đặt Ghostscript từ [trang web chính thức](https://www.ghostscript.com/releases/gsdnld.html) và đảm bảo nó được thêm vào PATH hệ thống của bạn.

## Cách sử dụng

Script được chạy từ dòng lệnh.

**Cú pháp cơ bản:**

```bash
python pdf_table_filter_script.py <input_directory> <output_directory> [options]
```

**Trong đó:**

*   `<input_directory>`: Đường dẫn đến thư mục chứa các tệp PDF bạn muốn xử lý.
*   `<output_directory>`: Đường dẫn đến thư mục nơi các tệp PDF đã lọc sẽ được lưu. Thư mục này sẽ được tạo nếu chưa tồn tại.

**Ví dụ:**

1.  **Chạy với các cài đặt mặc định (phương pháp hybrid, DPI 216, log INFO):**
    ```bash
    python pdf_table_filter_script.py ./input_pdfs ./output_filtered_pdfs
    ```
    (Hãy tạo thư mục `input_pdfs` và đặt các tệp PDF của bạn vào đó trước khi chạy.)

2.  **Chỉ sử dụng Vision Model, DPI cao hơn, log DEBUG:**
    ```bash
    python pdf_table_filter_script.py ./input_pdfs ./output_filtered_pdfs --method vision_model --dpi 288 --log_level DEBUG
    ```

3.  **Chỉ sử dụng phương pháp truyền thống, tùy chỉnh ngưỡng tin cậy cho Vision (nếu hybrid được dùng sau này):**
    ```bash
    python pdf_table_filter_script.py ./input_pdfs ./output_filtered_pdfs --method traditional --vision_high_conf 0.98
    ```

**Các tùy chọn dòng lệnh (`[options]`):**

*   `--method {vision_model,traditional,hybrid}`: Phương pháp phát hiện bảng (mặc định: `hybrid`).
*   `--vision_threshold VISION_THRESHOLD`: Ngưỡng tin cậy chung cho vision model (0.0-1.0, mặc định: `0.85`). Dùng trong bước kiểm tra ban đầu của vision model.
*   `--dpi DPI`: Độ phân giải ảnh (DPI) khi chuyển PDF sang ảnh (mặc định: `216`). DPI cao hơn cho ảnh rõ hơn nhưng xử lý chậm hơn.
*   `--vision_high_conf VISION_HIGH_CONF`: Ngưỡng tin cậy CAO cho vision model trong logic kết hợp (mặc định: `0.95`). Nếu score vision >= ngưỡng này, trang được coi là có bảng.
*   `--vision_medium_lower_conf VISION_MEDIUM_LOWER_CONF`: Ngưỡng tin cậy TRUNG BÌNH (cận dưới) cho vision model trong logic kết hợp (mặc định: `0.75`). Nếu score vision nằm giữa ngưỡng này và `vision_high_conf`, cần xác nhận thêm từ các phương pháp khác.
*   `--log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Mức độ chi tiết của log (mặc định: `INFO`).

Để xem tất cả các tùy chọn và mô tả của chúng:
```bash
python pdf_table_filter_script.py --help
```

## Cấu trúc thư mục dự kiến

```
your_project_directory/
├── pdf_table_filter_script.py  # Script chính
├── requirements.txt            # Danh sách các thư viện Python cần thiết
├── README.md                   # File hướng dẫn này
├── input_pdfs/                 # Thư mục chứa các PDF đầu vào
│   ├── document1.pdf
│   └── document2.pdf
└── output_filtered_pdfs/       # Thư mục chứa các PDF đã được lọc (được tạo bởi script)
    ├── document1_tables_only.pdf
    └── document2_tables_only.pdf
```

## Xử lý sự cố

*   **Lỗi `ghostscript` (thường với Camelot):** Đảm bảo Ghostscript đã được cài đặt và thêm vào PATH hệ thống.
*   **Lỗi tải model:** Kiểm tra kết nối internet. Nếu bạn ở sau proxy, có thể cần cấu hình biến môi trường `HTTP_PROXY` và `HTTPS_PROXY`.
*   **Hiệu suất chậm:**
    *   Giảm giá trị DPI (ví dụ: `144` hoặc `150`) có thể tăng tốc độ xử lý nhưng có thể làm giảm độ chính xác của Vision Model.
    *   Nếu không cần Vision Model, sử dụng phương pháp `traditional`.
    *   Xử lý các tệp PDF lớn với nhiều trang có thể tốn thời gian.
*   **Phát hiện không chính xác:**
    *   Thử nghiệm với các giá trị `vision_threshold`, `vision_high_conf`, `vision_medium_lower_conf` khác nhau.
    *   Kiểm tra log ở mức `DEBUG` để hiểu rõ hơn quyết định của từng phương pháp trên mỗi trang.
    *   Chất lượng của tệp PDF gốc (ví dụ: PDF dạng ảnh quét so với PDF văn bản) ảnh hưởng lớn đến kết quả.

## Đóng góp

Nếu bạn muốn đóng góp cho dự án này, vui lòng fork repository và tạo một pull request với các thay đổi của bạn.

## Giấy phép

[MIT License](LICENSE.md) (Bạn cần tạo file LICENSE.md nếu muốn)

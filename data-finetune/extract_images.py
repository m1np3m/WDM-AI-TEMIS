import json
import os
import shutil
from pathlib import Path

# --- Cấu hình ---
JSON_FILE_PATH = '/home/tiamo/WDM-AI-TEMIS/data-finetune/final.json'
SOURCE_FOLDER = '/home/tiamo/WDM-AI-TEMIS/data-finetune/cropped_tables_output'
DESTINATION_FOLDER = '/home/tiamo/WDM-AI-TEMIS/data-finetune/extracted_images'

def copy_images_from_json(json_file_path, source_folder, destination_folder):
    """
    Đọc file JSON, trích xuất danh sách ảnh và copy từ thư mục nguồn sang thư mục đích.
    
    Args:
        json_file_path (str): Đường dẫn đến file JSON.
        source_folder (str): Thư mục chứa ảnh nguồn.
        destination_folder (str): Thư mục đích để copy ảnh.
    """
    # Kiểm tra thư mục nguồn
    if not os.path.isdir(source_folder):
        print(f"Lỗi: Thư mục nguồn '{source_folder}' không tồn tại.")
        return
    
    # Tạo thư mục đích nếu chưa tồn tại
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    print(f"Thư mục đích: {destination_folder}")
    
    # Đọc file JSON
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file JSON tại '{json_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: File JSON tại '{json_file_path}' không hợp lệ.")
        return
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file JSON: {e}")
        return

    if not isinstance(data, list):
        print("Lỗi: Dữ liệu JSON không phải là một danh sách (list).")
        return

    print(f"Bắt đầu copy ảnh từ '{source_folder}' sang '{destination_folder}'...\n")

    copied_count = 0
    skipped_count = 0
    error_count = 0
    total_images = 0
    image_set = set()  # Để tránh copy trùng lặp

    # Trích xuất danh sách ảnh từ JSON
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Cảnh báo: Mục thứ {index+1} không phải là dictionary, bỏ qua.")
            continue

        image_filename = item.get('image_path')

        if not image_filename:
            print(f"Cảnh báo: Mục thứ {index+1} không có trường 'image_path'.")
            continue
        
        if not isinstance(image_filename, str):
            print(f"Cảnh báo: Mục thứ {index+1} có 'image_path' không phải là chuỗi: {image_filename}.")
            continue

        # Thêm vào set để tránh trùng lặp
        image_set.add(image_filename)

    total_images = len(image_set)
    print(f"Tổng số ảnh duy nhất cần copy: {total_images}\n")

    # Copy từng ảnh
    for image_filename in image_set:
        source_path = os.path.join(source_folder, image_filename)
        destination_path = os.path.join(destination_folder, image_filename)
        
        # Kiểm tra file nguồn có tồn tại không
        if not os.path.exists(source_path):
            print(f"[KHÔNG TÌM THẤY]: {image_filename}")
            error_count += 1
            continue
        
        if not os.path.isfile(source_path):
            print(f"[LỖI]: {image_filename} (không phải file)")
            error_count += 1
            continue
        
        # Kiểm tra file đích đã tồn tại chưa
        if os.path.exists(destination_path):
            print(f"[ĐÃ TỒN TẠI]: {image_filename} (bỏ qua)")
            skipped_count += 1
            continue
        
        # Copy file
        try:
            shutil.copy2(source_path, destination_path)
            print(f"[COPY THÀNH CÔNG]: {image_filename}")
            copied_count += 1
        except Exception as e:
            print(f"[LỖI COPY]: {image_filename} - {e}")
            error_count += 1

    # In thống kê kết quả
    print("\n" + "="*50)
    print("THỐNG KÊ KẾT QUẢ")
    print("="*50)
    print(f"Tổng số ảnh cần copy     : {total_images}")
    print(f"Số ảnh copy thành công   : {copied_count}")
    print(f"Số ảnh đã tồn tại (bỏ qua): {skipped_count}")
    print(f"Số ảnh lỗi               : {error_count}")
    print(f"Thư mục đích            : {destination_folder}")
    
    if copied_count > 0:
        print(f"\n✅ Copy thành công {copied_count} ảnh!")
    if error_count > 0:
        print(f"\n⚠️  Có {error_count} ảnh gặp lỗi, vui lòng kiểm tra!")
        

if __name__ == '__main__':
    copy_images_from_json(JSON_FILE_PATH, SOURCE_FOLDER, DESTINATION_FOLDER)
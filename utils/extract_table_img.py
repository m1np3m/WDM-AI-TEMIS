import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from pdf2image import convert_from_path
from PIL import Image
import os
import uuid
import os
import json


def detect_tables_and_save_images(pdf_path: str, output_dir: str = "table_snapshots", table_idx_map=None):
    import os
    os.makedirs(output_dir, exist_ok=True)

    processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    model.eval()

    pages = convert_from_path(pdf_path, dpi=200)

    global_table_idx = 0  # chỉ số duy nhất của bảng trên toàn bộ tài liệu

    for page_idx, page_image in enumerate(pages):
        image_rgb = page_image.convert("RGB")
        inputs = processor(images=image_rgb, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_object_detection(
            outputs, threshold=0.8, target_sizes=[image_rgb.size[::-1]]
        )[0]

        table_count_on_page = 0
        for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            if label.item() == 0:  # table
                xmin, ymin, xmax, ymax = map(int, box.tolist())
                cropped_image = image_rgb.crop((xmin, ymin, xmax, ymax))

                # File name pattern
                filename = f"{os.path.basename(pdf_path).replace('.pdf', '')}_page{page_idx+1}_table{global_table_idx}_part{table_count_on_page+1}.png"
                save_path = os.path.join(output_dir, filename)
                cropped_image.save(save_path)
                print(f"✅ Saved: {save_path}")

                # Cập nhật danh sách ảnh của bảng
                if table_idx_map is not None:
                    table_idx_map.setdefault(global_table_idx, []).append(filename)

                table_count_on_page += 1

        global_table_idx += 1




def save_tables_to_markdown(json_path: str, image_dir: str, markdown_dir: str):
    os.makedirs(markdown_dir, exist_ok=True)

    # Load data
    with open(json_path, "r", encoding="utf-8") as f:
        tables_sources = json.load(f)

    sources = list(tables_sources.keys())

    for source in sources:
        pdf_name = os.path.splitext(os.path.basename(source))[0]

        print(f"📄 Source {source} has {len(tables_sources[source])} tables")
        print("-" * 100)

        for table in tables_sources[source]:
            table_md = table['table_content']
            page_numbers = table['page_numbers']
            source_name = table['source']
            table_idx = table['table_idx']
            page_number = page_numbers[0] if page_numbers else 1

            # Tạo tên ảnh snapshot tương ứng
            image_filename = f"{pdf_name}_page{page_number}_table{table_idx}.png"
            image_path = os.path.join(image_dir, image_filename)

            # Kiểm tra file ảnh có tồn tại
            if not os.path.exists(image_path):
                print(f"⚠️  Image not found: {image_path}")
                continue

            # Ghi file markdown
            markdown_filename = f"{pdf_name}_table_{table_idx}.md"
            markdown_path = os.path.join(markdown_dir, markdown_filename)

            markdown_content = f"""---
source_pdf: {source_name}
table_idx: {table_idx}
page_numbers: {page_numbers}
---

![Table image](../{image_dir}/{image_filename})

{table_md}
"""
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            print(f"✅ Saved: {markdown_path}")
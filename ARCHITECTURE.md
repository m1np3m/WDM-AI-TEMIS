# Cấu trúc Package WDM-AI-TEMIS

Dự án đã được tổ chức lại theo các nguyên tắc best practices trong Python để đảm bảo code chuyên nghiệp, dễ bảo trì và có thể mở rộng.

## 🏗️ Cấu trúc thư mục

```
WDM-AI-TEMIS/
├── src/                          # Main package
│   ├── __init__.py              # Package entry point - expose các components chính
│   └── WDMParser/               # Core module (FULLY INDEPENDENT)
│       ├── __init__.py          # Module entry point
│       ├── WDMParser.py         # PDF parser chính
│       ├── extract_tables.py    # Logic trích xuất bảng
│       ├── enrich.py            # Enrich table bằng AI (local)
│       └── credential_helper.py # Helper xử lý credentials
├── utils/                       # Utility functions (separate from WDMParser)
│   ├── __init__.py             # Utils entry point
│   ├── enrich.py               # Legacy enrichment utilities
│   └── ...                     # Các utility khác
├── demo.py                     # Entry point chính cho demo
└── main.py                     # Entry point chính cho production
```

## ✅ Tuân thủ nguyên tắc Best Practices

### 1. **Module hoá và Import từ bên ngoài**
- ✅ Mỗi file chỉ chứa logic cụ thể, không chạy code trực tiếp
- ✅ Sử dụng relative imports trong package (`from .module import ...`)
- ✅ Không có khối `if __name__ == "__main__":` trong các module

### 2. **Entry Point rõ ràng**
- ✅ `demo.py` - Entry point cho demo và testing
- ✅ `main.py` - Entry point chính cho production use
- ✅ Tất cả logic demo/test được tách ra khỏi modules

### 3. **Expose Components đúng cách**
- ✅ `src/__init__.py` expose tất cả components cần thiết
- ✅ `src/WDMParser/__init__.py` expose các classes và functions
- ✅ `utils/__init__.py` expose utility functions

### 4. **🆕 Module Independence**
- ✅ **WDMParser hoàn toàn độc lập** - không phụ thuộc vào utils
- ✅ Tất cả enrich functionality được internal hoá trong WDMParser
- ✅ Utils và WDMParser có thể được sử dụng riêng biệt

## 📦 Cách sử dụng Package

### Import cơ bản:
```python
from src import (
    WDMPDFParser,           # Main parser class
    WDMText, WDMImage,      # Data types
    WDMTable, WDMMergedTable, # Table types
    full_pipeline,          # Table extraction pipeline
    get_tables_from_pdf,    # Basic table extraction
    Enrich_VertexAI,        # AI enrichment (from WDMParser)
    Enrich_Openrouter,      # OpenRouter enrichment (from WDMParser)
    setup_default_credentials # Credential helper
)
```

### Import trực tiếp từ WDMParser (độc lập):
```python
from src.WDMParser import (
    WDMPDFParser,
    Enrich_VertexAI,        # Internal enrich class
    Enrich_Openrouter,      # Internal enrich class
    validate_credentials_path
)
```

### Sử dụng parser:
```python
# Tạo parser instance
parser = WDMPDFParser(
    file_path="path/to/your.pdf",
    debug=True
)

# Extract tables cơ bản
tables = parser.extract_tables(pages=[1, 2])

# Extract tables với merging (cần credentials)  
merged_tables = parser.extract_tables(
    pages=[1, 2], 
    merge_span_tables=True,
    enrich=True
)

# Extract text và images
texts = parser.extract_text(pages=[1])
images = parser.extract_images(pages=[1])
```

## 🎯 Lợi ích của cấu trúc mới

1. **Professional**: Code được tổ chức theo chuẩn Python package
2. **Maintainable**: Dễ bảo trì và debug khi có vấn đề
3. **Scalable**: Dễ thêm tính năng mới mà không ảnh hưởng code cũ
4. **Testable**: Dễ viết unit tests cho từng module
5. **Reusable**: Có thể import và sử dụng ở nhiều nơi khác nhau
6. **🆕 Independent**: WDMParser hoàn toàn độc lập, có thể sử dụng riêng biệt

## 🚀 Cách chạy

### Demo:
```bash
python demo.py
```

### Production:
```bash
python main.py
```

### Import trong project khác:
```python
# Có thể import như một package thông thường
from src import WDMPDFParser
parser = WDMPDFParser("document.pdf")

# Hoặc chỉ sử dụng WDMParser độc lập
from src.WDMParser import WDMPDFParser, Enrich_VertexAI
parser = WDMPDFParser("document.pdf")
enricher = Enrich_VertexAI(credentials_path="path/to/key.json")
```

## 🔒 Module Independence

**WDMParser** giờ đây hoàn toàn độc lập:
- ✅ Không phụ thuộc vào `utils` package
- ✅ Có enrich functionality riêng trong `src/WDMParser/enrich.py`
- ✅ Có thể được sử dụng như một package riêng biệt
- ✅ Dễ dàng extract thành standalone package nếu cần

---

Cấu trúc này tuân thủ hoàn toàn các nguyên tắc best practices và đảm bảo independence giữa các modules. 
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WDMParser import WDMPDFParser

if __name__ == "__main__":
    parser = WDMPDFParser(
        file_path="C:/Users/PC/CODE/WDM-AI-TEMIS/data/pdfs/b014b8ca3c8ee543b655c29747cc6090.pdf",
        debug=True,
        debug_level=1,
    )
    print("Extracting images...")
    images = parser.extract_images(pages=[1])
    print(f"Found {len(images)} images")

    # Extract text
    print("Extracting text...")
    texts = parser.extract_text()
    print(f"Extracted text from {len(texts)} pages")

    # Extract tables
    print("Extracting tables ... (without merging)")
    tables = parser.extract_tables()
    print(f"Found {len(tables)} tables")

    print("Extracting tables ... (with merging)")
    tables = parser.extract_tables(merge_span_tables=True)
    print(f"Found {len(tables)} tables")

    print("Extracting tables ... (with merging and enriching)")
    tables = parser.extract_tables(merge_span_tables=True, enrich=True)
    print(f"Found {len(tables)} tables")

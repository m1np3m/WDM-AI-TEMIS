"""Example usage of WDM-AI-TEMIS package with proper imports."""

# Option 1: Import from src package
from src import WDMPDFParser

# Option 2: Import directly from src.extraction  
# from src.extraction import WDMPDFParser, Text, WDMImage

# Option 3: Import from root package (after installing as package)
# from wdm_ai_temis import WDMPDFParser

def main():
    # Initialize parser
    parser = WDMPDFParser(
        file_path="data/pdfs/b014b8ca3c8ee543b655c29747cc6090.pdf",
        debug=True,
        debug_level=1,
    )
    
    # Extract images (without saving to disk)
    images = parser.extract_images(pages=[1])
    print(f"Found {len(images)} images")
    
    # Extract images (with saving to disk)
    images_with_files = parser.extract_images(
        pages=[1], 
        stored_path="test_images"
    )
    print(f"Saved {len(images_with_files)} images to disk")
    
    # Extract text
    texts = parser.extract_text()
    print(f"Extracted text from {len(texts)} pages")
    
    # Extract tables
    tables = parser.extract_tables()
    print(f"Found {len(tables)} tables")
    
    # Extract tables with advance features
    tables = parser.extract_tables(merge_span_tables=True)
    print(f"Found {len(tables)} tables")

if __name__ == "__main__":
    main() 
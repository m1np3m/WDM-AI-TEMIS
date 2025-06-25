import sys
import os
import time

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WDMParser import WDMPDFParser



if __name__ == "__main__":
    parser = WDMPDFParser(
        file_path="data/pdfs/b014b8ca3c8ee543b655c29747cc6090.pdf",
        # credential_path="C:/Users/Admin/Data/WDM-AI-TEMIS/gdsc2025-74596a254ab4.json",  # Explicit credential path
        debug=True,
        debug_level=1,
    )
    
    start_time = time.time()
    tables = parser.extract_tables(merge_span_tables=True, enrich=True, pages=[2, 3])
    end_time = time.time()
    for table in tables:
        print(table['text'])



    print(f"Time taken: {end_time - start_time} seconds")
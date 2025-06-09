import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WDMParser import WDMPDFParser



if __name__ == "__main__":
    parser = WDMPDFParser(
        file_path="data/experiment_data/b014b8ca3c8ee543b655c29747cc6090.pdf",
        credential_path="key_vertex.json",  # Explicit credential path
        debug=True,
        debug_level=1,
    )
    
    
    tables = parser.extract_tables(merge_span_tables=True, enrich=True, pages=[2, 3])
    for table in tables:
        print(table['text'])
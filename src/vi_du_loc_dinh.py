import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WDMParser import WDMPDFParser



if __name__ == "__main__":
    parser = WDMPDFParser(
        file_path="C:/Users/Admin/Data/WDM-AI-TEMIS/data/experiment_data/c935e2902adf7040a6ffe0db0f7c11e6.pdf",
        debug=True,
        debug_level=1,
    )
    
    
    tables = parser.extract_tables(merge_span_tables=False, enrich=False, pages=[2, 3])
    for table in tables:
        print(table['text'])
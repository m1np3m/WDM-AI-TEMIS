import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WDMParser import WDMPDFParser

def test_credentials():
    """Test credential validation"""
    try:
        from WDMParser.credential_helper import validate_credentials_path
        cred_path = validate_credentials_path()
        print(f"✅ Credentials found at: {cred_path}")
        return True
    except Exception as e:
        print(f"❌ Credentials validation failed: {e}")
        return False

def test_extraction():
    """Test table extraction with debug info"""
    print("\n" + "="*60)
    print("TESTING TABLE EXTRACTION")
    print("="*60)
    
    # Test credentials first
    has_credentials = test_credentials()
    
    parser = WDMPDFParser(
        file_path="C:/Users/Admin/Data/WDM-AI-TEMIS/data/experiment_data/b014b8ca3c8ee543b655c29747cc6090.pdf",
        debug=True,
        debug_level=1,
    )
    
    print(f"\nTesting with credentials: {has_credentials}")
    print("Extracting tables with merge_span_tables=True, enrich=True...")
    
    tables = parser.extract_tables(merge_span_tables=True, enrich=True, pages=None)
    for table in tables:
        print(table['text'])
if __name__ == "__main__":
    test_extraction()
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)
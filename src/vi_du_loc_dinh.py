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
        file_path="data/experiment_data/b014b8ca3c8ee543b655c29747cc6090.pdf",
        debug=True,
        debug_level=1,
    )
    
    print(f"\nTesting with credentials: {has_credentials}")
    print("Extracting tables with merge_span_tables=True, enrich=True...")
    
    try:
        tables = parser.extract_tables(merge_span_tables=True, enrich=True, pages=[2, 3])
        print(f"\n🎉 SUCCESS: Found {len(tables)} merged tables")
        
        for i, table in enumerate(tables):
            print(f"\nTable {i+1}:")
            print(f"  Pages: {table['page']}")
            print(f"  Rows: {table['n_rows']}")
            print(f"  Columns: {table['n_columns']}")
            print(f"  Text preview: {table['text'][:200]}...")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        
        # Try without advanced features as fallback
        print("\nTrying basic extraction without merge_span_tables...")
        try:
            basic_tables = parser.extract_tables(pages=[2, 3])
            print(f"Basic extraction found {len(basic_tables)} tables")
        except Exception as e2:
            print(f"Even basic extraction failed: {e2}")

if __name__ == "__main__":
    test_extraction() 
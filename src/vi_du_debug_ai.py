import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WDMParser import WDMPDFParser

def debug_ai_analysis():
    """Debug AI analysis step by step"""
    
    print("="*60)
    print("DEBUG AI ANALYSIS")
    print("="*60)
    
    # Test credentials first
    try:
        from WDMParser.credential_helper import validate_credentials_path
        cred_path = validate_credentials_path()
        print(f"✅ Credentials found at: {cred_path}")
    except Exception as e:
        print(f"❌ Credentials validation failed: {e}")
        return
    
    # Test AI functions
    try:
        from WDMParser.llm_feat import get_is_new_section_context, get_is_has_header
        
        # Test with dummy data
        contexts = ["This is a test context"]
        headers = [["Year", "Title", "Role"]]
        first_3_rows = ["| Year | Title | Role |\n|------|-------|------|\n| 2002 | Test | Actor |"]
        
        print("\n🧪 Testing get_is_new_section_context...")
        try:
            res1, prompt1 = get_is_new_section_context(contexts, return_prompt=True)
            print(f"✅ get_is_new_section_context success: {res1}")
        except Exception as e:
            print(f"❌ get_is_new_section_context failed: {e}")
        
        print("\n🧪 Testing get_is_has_header...")
        try:
            res2, prompt2 = get_is_has_header(headers, first_3_rows, return_prompt=True)
            print(f"✅ get_is_has_header success: {res2}")
        except Exception as e:
            print(f"❌ get_is_has_header failed: {e}")
            
    except Exception as e:
        print(f"❌ Failed to import AI functions: {e}")
    
    # Test full extraction
    print("\n🧪 Testing full table extraction...")
    try:
        parser = WDMPDFParser(
            file_path="data/experiment_data/b014b8ca3c8ee543b655c29747cc6090.pdf",
            debug=True,
            debug_level=1,
        )
        
        # Test without merge first
        basic_tables = parser.extract_tables(pages=[2, 3])
        print(f"Basic extraction: {len(basic_tables)} tables")
        
        # Test with merge 
        merged_tables = parser.extract_tables(merge_span_tables=True, enrich=False, pages=[2, 3])
        print(f"Merge extraction (no enrich): {len(merged_tables)} tables")
        
    except Exception as e:
        print(f"❌ Table extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ai_analysis() 
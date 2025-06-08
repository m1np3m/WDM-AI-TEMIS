#!/usr/bin/env python3
"""
Demo script for WDM-AI-TEMIS PDF extraction package.
This demonstrates the proper way to use the package with module imports.
"""

import os
from src import (
    WDMPDFParser, 
    setup_default_credentials, 
    print_credentials_help
)


def main():
    """Main demo function showing package usage."""
    
    # Try to setup credentials automatically
    if not setup_default_credentials():
        print("WARNING: No Google Cloud credentials found.")
        print("If you plan to use enrichment features, please set up credentials.")
        print("For now, running without enrichment...\n")
        print("For setup instructions, run: python -c 'from src import print_credentials_help; print_credentials_help()'")
        print()

    # Example PDF file path - update this to your actual PDF
    source_path = "C:/Users/PC/CODE/WDM-AI-TEMIS/data/pdfs/b014b8ca3c8ee543b655c29747cc6090.pdf"
    
    # Check if file exists
    if not os.path.exists(source_path):
        print(f"PDF file not found: {source_path}")
        print("Please update the source_path variable in this script to point to an actual PDF file.")
        return

    # Create parser instance
    parser = WDMPDFParser(
        file_path=source_path,
        debug=True,
        debug_level=1,
    )

    # Demo 1: Extract text
    print("=== DEMO 1: Extracting text ===")
    try:
        texts = parser.extract_text(pages=[1])  # Extract from first page only
        print(f"✅ Extracted text from {len(texts)} pages")
        if texts:
            print(f"Preview: {texts[0]['text'][:200]}...")
    except Exception as e:
        print(f"❌ Error extracting text: {e}")

    print("\n")

    # Demo 2: Extract tables (basic)
    print("=== DEMO 2: Extracting tables (basic) ===")
    try:
        tables = parser.extract_tables(pages=[1])  # Extract from first page only
        print(f"✅ Found {len(tables)} tables")
        for i, table in enumerate(tables):
            print(f"Table {i+1}: {table['n_rows']} rows, {table['n_columns']} columns")
    except Exception as e:
        print(f"❌ Error extracting tables: {e}")

    print("\n")

    # Demo 3: Extract tables with merging (requires credentials)
    print("=== DEMO 3: Extracting tables with merging ===")
    try:
        merged_tables = parser.extract_tables(pages=[1], merge_span_tables=True)
        print(f"✅ Found {len(merged_tables)} merged tables")
        for i, table in enumerate(merged_tables):
            print(f"Merged Table {i+1}: {table['n_rows']} rows, {table['n_columns']} columns")
    except Exception as e:
        print(f"❌ Error extracting merged tables: {e}")

    print("\n")
    
    # Demo 4: Extract tables with enrichment
    print("=== DEMO 4: Extracting tables with enrichment ===")
    try:
        merged_tables = parser.extract_tables(pages=[1], merge_span_tables=True, enrich=True)
        print(f"✅ Found {len(merged_tables)} merged tables")
        for i, table in enumerate(merged_tables):
            print(f"Merged Table {i+1}: {table['n_rows']} rows, {table['n_columns']} columns")
    except Exception as e:  
        print(f"❌ Error extracting merged tables with enrichment: {e}")
        
    print("\n")

    # Demo 5: Extract images 
    print("=== DEMO 5: Extracting images ===")
    try:
        images = parser.extract_images(pages=[1])  # Extract from first page only
        print(f"✅ Found {len(images)} images")
        for i, image in enumerate(images):
            print(f"Image {i+1}: Page {image['page']}, Size: {len(image['base64_image'])} chars")
    except Exception as e:
        print(f"❌ Error extracting images: {e}")

    print("\n=== Demo completed! ===")


if __name__ == "__main__":
    main() 
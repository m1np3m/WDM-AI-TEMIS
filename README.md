# WDM-AI-TEMIS

WDM-AI-TEMIS is an advanced RAG (Retrieval-Augmented Generation) system that processes PDF documents to extract text, tables, and images for intelligent document analysis.

## 🆕 New Features

### Image Extraction with AI-Generated Summaries

The system now supports intelligent image extraction with AI-powered summaries:

- **Mode=summary**: Uses AI to generate descriptive summaries of images
- **Mode=metadata**: Provides basic image information (location, source, path)
- **Automatic image detection**: Finds and extracts images from PDF documents
- **Smart filtering**: Skips small images (likely icons or decorations)

## Features

- **Multi-format Processing**: Extract text, tables, and images from PDF documents
- **AI-Enhanced Table Processing**: Merge spanning tables and enrich with AI analysis
- **Intelligent Image Analysis**: Generate summaries and descriptions of images
- **Async Processing**: Handle multiple documents concurrently
- **Memory Management**: Efficient processing with memory optimization
- **Conversation Memory**: Maintain context across multiple queries
- **Reranking**: Improve search results with advanced reranking algorithms

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.WDMParser import WDMPDFParser

# Initialize parser
parser = WDMPDFParser(
    file_path="path/to/your/document.pdf",
    credential_path="path/to/credentials.json",  # Optional, for AI features
    debug=True
)

# Extract images with AI summaries
images = parser.extract_images(pages=[1], mode="summary")
print(f"Found {len(images)} images with summaries")

# Extract text
texts = parser.extract_text()
print(f"Extracted text from {len(texts)} pages")

# Extract tables
tables = parser.extract_tables(merge_span_tables=True)
print(f"Found {len(tables)} tables")
```

### Advanced Async Processing

```python
import asyncio
from src.WDMParser import WDMPDFParser

async def process_documents():
    settings = WDMPDFParser.create_settings(
        credential_path="path/to/credentials.json",
        debug=True,
        max_concurrent_files=3
    )
    
    parser = WDMPDFParser(settings=settings)
    
    # Process with all features enabled
    documents = await parser.process_pdf(
        pdf_data="path/to/document.pdf",
        extract_text=True,
        extract_images=True,
        image_mode="summary",
        merge_span_tables=True,
        enrich=True
    )
    
    # Separate by document type
    text_docs = [d for d in documents if d.metadata.get('type') == 'text']
    table_docs = [d for d in documents if d.metadata.get('type') == 'table']
    image_docs = [d for d in documents if d.metadata.get('type') == 'image']
    
    print(f"Results: {len(text_docs)} text, {len(table_docs)} tables, {len(image_docs)} images")

asyncio.run(process_documents())
```

### RAG System with Image Support

```python
from src.rag import RAG

# Initialize RAG system
rag = RAG(
    embedding_type="huggingface",
    embedding_model="BAAI/bge-base-en-v1.5",
    enable_hybrid_search=True,
    chunk_type="character",
    use_memory=True,
    collection_name="my_collection",
    persist_dir="./qdrant_db",
    use_reranker=True,
    enable_conversation_memory=True
)

# Process PDFs with image extraction
documents, results, stats = await rag.process_pdfs_bytes(
    pdf_data_list=["path/to/doc1.pdf", "path/to/doc2.pdf"],
    extract_images=True,
    image_mode="summary",
    debug_mode=True
)

# Add documents to vector store
rag.add_documents(documents)

# Query with image support
result = rag.chat("Show me the charts and diagrams about sales performance")
print(result["response"])
```

## Image Processing Modes

### Summary Mode (Recommended)

```python
# Extract images with AI-generated summaries
images = parser.extract_images(pages=[1, 2, 3], mode="summary")

# Example output:
# "Image from page 1 of document.pdf
# Location: (0, 0, 800, 600)
# Summary: A bar chart showing quarterly sales data with three categories: 
# Q1 showing $50k, Q2 showing $75k, Q3 showing $100k, and Q4 showing $125k.
# The chart uses blue bars and includes a legend."
```

### Metadata Mode

```python
# Extract images with basic metadata
images = parser.extract_images(pages=[1], mode="metadata")

# Example output:
# "Image from page 1 of document.pdf
# Location: (0, 0, 800, 600)
# Image path: extracted_images/document_page1_0.png"
```

## Configuration

### Credentials Setup

For AI-powered features (image summaries, table enrichment), set up Google Cloud credentials:

1. Create a Google Cloud project
2. Enable Vertex AI API
3. Create a service account and download the JSON key file
4. Set the credential path in your code

### Memory Management

```python
settings = WDMPDFParser.create_settings(
    max_concurrent_files=3,  # Limit concurrent processing
    max_memory_mb=4096,      # Set memory limit (4GB)
    batch_size=5,            # Process in batches
    cleanup_interval=10      # Clean up every 10 batches
)
```

## Query Types for Image Search

The system can intelligently detect when users are asking about visual content:

```python
# These queries will automatically search for images:
rag.chat("Show me the charts in the financial report")
rag.chat("What diagrams are in the technical documentation?")
rag.chat("Are there any graphs showing the trends?")
rag.chat("Display the flowchart from the process document")
```

## File Structure

```
WDM-AI-TEMIS/
├── src/
│   ├── WDMParser/
│   │   ├── WDMParser.py          # Main parser with image support
│   │   ├── extract_tables.py     # Table extraction
│   │   ├── enrich.py            # AI enhancement
│   │   └── ...
│   ├── rag.py                   # RAG system with image support
│   ├── vectorstore.py           # Vector storage
│   └── ...
├── extracted_images/            # Saved extracted images
├── test_images/                 # Test image outputs
└── README.md
```

## API Reference

### WDMPDFParser Methods

- `extract_images(pages=None, mode="summary")` - Extract images from PDF
- `extract_text(pages=None)` - Extract text content
- `extract_tables(pages=None, merge_span_tables=True, enrich=True)` - Extract tables
- `process_pdf(pdf_data, extract_images=True, image_mode="summary", ...)` - Async processing

### RAG Methods

- `process_pdfs_bytes(pdf_data_list, extract_images=True, image_mode="summary", ...)` - Process multiple PDFs
- `chat(query)` - Chat interface with image support
- `query_analysis(query, available_types=["text", "table", "image"])` - Analyze query intent

## Examples

See `src/example_use_parser.py` for complete usage examples.

## Troubleshooting

### Common Issues

1. **Image summarization fails**: Ensure valid Google Cloud credentials are set
2. **Memory errors**: Reduce `max_concurrent_files` and `batch_size`
3. **No images found**: Check if PDF contains extractable images (not scanned)
4. **Poor image quality**: Increase DPI in extraction settings

### Debug Mode

Enable debug mode for detailed logging:

```python
parser = WDMPDFParser(debug=True, debug_level=2)
```

## Performance Tips

- Use `mode="metadata"` for faster processing when summaries aren't needed
- Set appropriate memory limits based on your system
- Process large documents in smaller page ranges
- Use async processing for multiple documents

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details




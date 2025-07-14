# WDM-AI-TEMIS

WDM-AI-TEMIS is an advanced RAG (Retrieval-Augmented Generation) system that processes PDF documents to extract text, tables, and images for intelligent document analysis through an intuitive Streamlit web interface.

## Features

- **Multi-format Document Processing**: Extract text, tables, and images from PDF documents
- **AI-Enhanced Content Analysis**: Generate intelligent summaries and descriptions of images and tables
- **Intelligent Conversation Management**: Maintain context across multiple queries with conversation optimization
- **Advanced Search Capabilities**: Hybrid search combining dense and sparse vectors for superior retrieval
- **Reranking Technology**: Improve search results precision with advanced reranking algorithms
- **Real-time Processing**: Handle multiple documents concurrently with memory optimization
- **Flexible Configuration**: Customizable embedding models, chunk strategies, and optimization settings
- **Interactive Web Interface**: User-friendly Streamlit application for seamless document interaction

## Quick Start with Streamlit Application

### Prerequisites

```bash
git clone https://github.com/m1np3m/WDM-AI-TEMIS.git

cd WDM-AI-TEMIS

uv venv --python 3.10
uv sync
```

### Environment Setup

Create a `.env` file in the root directory with your credentials:

```env
# Langfuse (Optional - for conversation tracking)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# Google Cloud Credentials (Optional - for AI-enhanced features)
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
```

### Running the Application

```bash
source .venv/bin/activate # Linux / Macos

#.venv\Scripts\activate # Windows
```

```bash
cd app
streamlit run main.py
```

The application will be available at `http://localhost:8501`

## Application Guide

### 1. Initial Configuration

When you first open the application, configure the following settings in the sidebar:

#### Vector Database Settings
- **Embedding Type**: Choose between `huggingface` (local) or `vertexai` (Google Cloud)
- **Embedding Model**: Select from available models based on your embedding type
- **Hybrid Search**: Enable for improved retrieval accuracy (recommended)
- **Text Chunking Strategy**: Choose between `character` or `recursive` chunking
- **Use Reranker**: Enable to improve search result precision

#### Conversation Optimization
- **Optimization Preset**: Select from predefined optimization levels:
  - `performance`: Maximum speed, minimal context
  - `default`: Balanced performance and context retention
  - `quality`: Maximum context retention, slower processing
- **Token Management**: Configure maximum conversation tokens and buffer settings
- **Summarization**: Set when to trigger conversation summarization

### 2. Document Upload and Processing

#### Uploading Documents
1. Navigate to the "📄 Upload PDF" section in the sidebar
2. Enable "Debug Mode" for detailed processing information (optional)
3. Provide Google Service Account credentials path if available (for enhanced AI features)
4. Select one or multiple PDF files using the file uploader
5. Click "🚀 Process PDFs" to begin processing

#### Processing Features
The system automatically extracts:
- **Text Content**: All readable text from PDF pages
- **Tables**: Structured data with intelligent table merging
- **Images**: Visual content with AI-generated descriptions (when credentials are provided)

### 3. Conversational Interface

#### Starting a Conversation
- Once documents are processed, use the chat input at the bottom to ask questions
- The system maintains conversation context automatically
- Previous conversations are preserved and can be accessed

#### Query Types
The system intelligently handles various query types:

**Text-based Queries**:
```
"What is the main conclusion of the research paper?"
"Summarize the financial results from Q3"
```

**Table-specific Queries**:
```
"Show me the sales data from the quarterly report"
"What are the key metrics in the performance table?"
```

**Image-related Queries**:
```
"Describe the charts in the presentation"
"What diagrams show the system architecture?"
"Are there any graphs showing trends?"
```

#### Conversation Management
- **🆕 New Chat**: Start a fresh conversation
- **📜 Show History**: View previous messages in current conversation
- **Clear History**: Reset conversation while keeping documents
- **🗑️ Clear Database**: Remove all documents and conversations

### 4. Understanding Results

#### Retrieved Context Panel
The right panel shows:
- **Source Documents**: List of documents used to answer your query
- **Document Types**: Icons indicating text (📝), tables (🔢), or images (🖼️)
- **Full Context**: Expandable view of the complete context sent to the AI model
- **Available Sources**: List of all documents in your knowledge base

#### Response Quality Indicators
- **Conversation Context**: Shows how previous messages influence current responses
- **Token Usage**: Displays conversation optimization status
- **Source Attribution**: Each response references specific document pages

### 5. Advanced Configuration

#### Performance Optimization
- **Max Concurrent Files**: Adjust based on system memory (default: 2-3)
- **Memory Management**: Set limits for large document processing
- **Batch Processing**: Configure batch sizes for optimal performance

#### Search Optimization
- **Hybrid Search**: Combines semantic and keyword search for better results
- **Reranking**: Uses advanced models to improve result relevance
- **Chunk Strategy**: Optimize text splitting for your document types

#### Conversation Optimization
- **Token Limits**: Prevent context window overflow
- **Summarization**: Automatic conversation summarization for long chats
- **Message Prioritization**: Keep most relevant recent messages

## Troubleshooting

### Common Issues

**No documents found for query**:
- Ensure documents are successfully processed (check for success message)
- Try rephrasing your question
- Verify the content exists in your uploaded documents

**Memory errors during PDF processing**:
- Reduce the number of concurrent files in settings
- Process large documents in smaller batches
- Ensure sufficient system memory

**AI features not working**:
- Verify Google Cloud credentials are correctly configured
- Check that the service account has necessary permissions
- Ensure Vertex AI API is enabled in your Google Cloud project

**Slow response times**:
- Disable reranking for faster responses
- Use performance optimization preset
- Reduce conversation token limits

### Performance Tips

1. **Document Preparation**: Ensure PDFs are text-searchable rather than scanned images
2. **Query Optimization**: Be specific in your questions for better results
3. **Memory Management**: Clear database periodically when working with many documents
4. **Model Selection**: Choose appropriate embedding models for your use case

## API Integration

For programmatic access, the underlying RAG system can be used directly:

```python
from src.rag import RAG

# Initialize RAG system
rag = RAG(
    embedding_type="huggingface",
    embedding_model="BAAI/bge-base-en-v1.5",
    enable_hybrid_search=True,
    use_reranker=True,
    enable_conversation_memory=True
)

# Process documents
documents = await process_pdfs_with_settings(pdf_files)
rag.add_documents(documents)

# Start conversation
conversation_id = rag.start_conversation(user_id="user1")
rag.use_conversation(conversation_id)

# Query
result = rag.chat("Your question here")
print(result["response"])
```

## File Structure

```
WDM-AI-TEMIS/
├── app/
│   ├── main.py                 # Streamlit application
│   └── fastapi_app.py         # FastAPI backend (optional)
├── src/
│   ├── WDMParser/             # PDF processing engine
│   ├── rag.py                 # RAG system core
│   ├── vectorstore.py         # Vector database management
│   └── setting.py             # Configuration management
├── qdrant_db/                 # Vector database storage
├── extracted_images/          # Processed image outputs
└── configs/                   # Configuration files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the application logs in debug mode




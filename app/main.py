import asyncio
import os
import sys
import time
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langfuse import Langfuse
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import RAG
from src.WDMParser.WDMParser import WDMPDFParser

st.set_page_config(
    page_title="WDM-AI-TEMIS - RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

load_dotenv()

# ============================== CACHED FUNCTIONS ==============================
@st.cache_resource
def initialize_langfuse():
    """Cache Langfuse client."""
    try:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            logger.warning("Langfuse keys not found in .env file. Tracing will be disabled.")
            return None
            
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        logger.info("Langfuse client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Langfuse initialization error: {e}")
        st.error(f"❌ Error initializing Langfuse: {e}")
        return None

@st.cache_resource
def initialize_rag(
    embedding_type,
    embedding_model,
    enable_hybrid_search,
    chunk_type,
    collection_name,
    persist_dir,
    use_reranker,
    _langfuse_client,
):
    """Cache RAG instance để tránh khởi tạo lại mỗi lần refresh"""
    try:
        rag = RAG(
            embedding_type=embedding_type,
            embedding_model=embedding_model,
            enable_hybrid_search=enable_hybrid_search,
            chunk_type=chunk_type,
            use_memory=False,
            collection_name=collection_name,
            persist_dir=persist_dir,
            use_reranker=use_reranker,
            langfuse_client=_langfuse_client,
        )
        logger.info(f"RAG initialized with cache")
        return rag
    except Exception as e:
        logger.error(f"RAG initialization error: {e}")
        st.error(f"❌ Error initializing RAG: {e}")
        st.stop()

# ============================== USEFUL FUNCTIONS ==============================

def clear_history():
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.success("History cleared!")

def clear_rag_cache():
    """Clear RAG cache và reinitialize"""
    initialize_rag.clear()
    # Xóa RAG khỏi session state để force reinitialize
    if "rag" in st.session_state:
        del st.session_state.rag
    if "rag_config_key" in st.session_state:
        del st.session_state.rag_config_key
    st.success("RAG cache cleared! Page will refresh to reinitialize.")
    st.rerun()

async def process_pdfs_with_streamlit(pdf_files: List, credential_path: Optional[str] = None) -> List[Document]:
    """
    Process PDF files using the new WDMParser with bytes support
    
    Args:
        pdf_files: List of Streamlit UploadedFile objects
        credential_path: Path to Google Cloud credentials (can be None)
        
    Returns:
        List of processed documents
    """
    if not pdf_files:
        return []
    
    # Create parser settings
    settings = WDMPDFParser.create_settings(
        credential_path=credential_path if credential_path else "",
        debug=True,
        debug_level=1,
        max_concurrent_files=2,  # Conservative for web apps
        max_memory_mb=2048,      # 2GB limit for web apps
        batch_size=3,
        cleanup_interval=2
    )
    
    parser = WDMPDFParser(settings=settings)
    
    # Convert uploaded files to bytes
    pdf_bytes_list = []
    file_names = []
    
    for pdf_file in pdf_files:
        try:
            pdf_bytes = pdf_file.getvalue()
            pdf_bytes_list.append(pdf_bytes)
            file_names.append(pdf_file.name)
        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {e}")
            continue
    
    if not pdf_bytes_list:
        st.error("No valid PDF files to process")
        return []
    
    try:
        # Process all PDFs asynchronously with bytes
        results = await parser.process_documents(
            pdf_documents=pdf_bytes_list,
            merge_span_tables=True,
            enrich=False,  # Disable for faster web processing
            extract_text=True,
            return_failed=False
        )
        
        # Handle potential tuple return from process_documents
        if isinstance(results, tuple):
            results_dict, failed_files = results
        else:
            results_dict = results
        
        # Combine all documents from all files
        all_documents = []
        total_tables = 0
        total_text = 0
        
        for identifier, documents in results_dict.items():
            # Update source metadata to use original filename
            file_index = int(identifier.replace("<in-memory-", "").replace(">", ""))
            original_filename = file_names[file_index] if file_index < len(file_names) else f"file_{file_index}"
            
            for doc in documents:
                doc.metadata['source'] = original_filename
                all_documents.append(doc)
            
            table_docs = [d for d in documents if d.metadata.get('type') == 'table']
            text_docs = [d for d in documents if d.metadata.get('type') == 'text']
            
            total_tables += len(table_docs)
            total_text += len(text_docs)
        
        # Show processing summary
        st.success(f"✅ Successfully processed {len(pdf_files)} PDF files!")
        st.info(f"📊 Extracted: {total_tables} tables, {total_text} text blocks ({len(all_documents)} total documents)")
        
        # Show memory usage
        memory_info = parser.get_memory_info()
        if 'error' not in memory_info:
            st.info(f"💾 Memory usage: {memory_info['rss_mb']:.1f}MB")
        
        return all_documents
        
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {e}")
        logger.error(f"PDF processing error: {e}")
        return []

# ============================== MAIN FUNCTION ================================

def main():
    st.title("🤖 WDM-AI-TEMIS - RAG Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    langfuse_client = initialize_langfuse()

    # Side bar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            st.button("Clear history", on_click=clear_history)
        with col2:
            st.button("Clear RAG cache", on_click=clear_rag_cache)

        st.subheader("🗂️ Vector Database Settings")

        embedding_type = st.selectbox(
            "Embedding Type",
            options=["huggingface", "vertexai"],
            index=0,
            help="Choose between Google VertexAI or local HuggingFace embeddings",
        )
        if embedding_type == "huggingface":
            embedding_model = st.selectbox(
                "HuggingFace Model",
                options=[
                    "BAAI/bge-base-en",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                ],
                index=0,
                help="Select HuggingFace embedding model",
            )
        else:
            embedding_model = st.selectbox(
                "VertexAI Model",
                options=["text-embedding-004", "textembedding-gecko@003"],
                index=0,
                help="Select Google VertexAI embedding model",
            )

        # Vector store settings
        enable_hybrid_search = st.checkbox(
            "Enable Hybrid Search",
            value=True,
            help="Combine dense and sparse vectors for better retrieval",
        )
        
        use_reranker = st.checkbox(
            "Use Reranker",
            value=True,
            help="Use reranker to get the top K documents",
        )

        chunk_type = st.selectbox(
            "Text Chunking Strategy",
            options=["character", "recursive"],
            index=0,
            help="Choose text splitting strategy: character (simple) or recursive (smart)",
        )

        # Initialize RAG với cache - chỉ khi cần thiết
        persist_dir = "./qdrant_db"

        # Tạo key để kiểm tra xem có cần khởi tạo lại không
        rag_config_key = f"{embedding_type}_{embedding_model}_{enable_hybrid_search}_{chunk_type}_{use_reranker}"
        
        # Tự động tạo collection name dựa trên config để tránh xung đột
        collection_name = f"wdm_{rag_config_key}".replace("-", "_").replace(".", "_").lower()

        # Chỉ khởi tạo RAG khi chưa có hoặc config thay đổi
        if (
            "rag" not in st.session_state
            or st.session_state.rag is None
            or "rag_config_key" not in st.session_state
            or st.session_state.rag_config_key != rag_config_key
        ):
            with st.spinner("Initializing Vector Database..."):
                rag = initialize_rag(
                    embedding_type=embedding_type,
                    embedding_model=embedding_model,
                    enable_hybrid_search=enable_hybrid_search,
                    chunk_type=chunk_type,
                    collection_name=collection_name,
                    persist_dir=persist_dir,
                    use_reranker=use_reranker,
                    _langfuse_client=langfuse_client,
                )

            # Lưu vào session state
            st.session_state.rag = rag
            st.session_state.rag_config_key = rag_config_key
            logger.info(f"RAG initialized with config key: {rag_config_key}")
        else:
            # RAG đã có sẵn, không cần khởi tạo lại
            logger.debug("Using existing RAG from session state")

        # Display current configuration
        with st.expander("📊 Current Vector Database Config", expanded=False):
            st.write(f"**Collection:** {collection_name}")
            st.write(f"**Embedding:** {embedding_type} - {embedding_model}")
            st.write(
                f"**Hybrid Search:** {'Enabled' if enable_hybrid_search else 'Disabled'}"
            )
            st.write(f"**Chunk Type:** {chunk_type}")
            st.write(f"**Use Reranker:** {st.session_state.rag.reranker_name if st.session_state.rag.use_reranker else 'Disabled'}")
            if st.session_state.rag:
                sources = st.session_state.rag.get_unique_sources()
                if sources and sources != ["No sources available"]:
                    st.write(f"**Sources:** {len(sources)} document(s)")
                else:
                    st.write("**Sources:** No documents loaded")

        # Vector Database Actions
        if st.button("🗑️ Clear Database"):
            if st.session_state.rag:
                with st.spinner("Clearing database..."):
                    try:
                        st.session_state.rag.clear_vectorstore()
                        clear_history()
                        st.success("✅ Database cleared!")
                    except Exception as e:
                        st.error(f"❌ Error clearing database: {e}")
            else:
                st.warning("No database to clear")

        st.markdown("---")
        st.subheader("📄 Upload PDF")

        # Add debug mode option
        debug_mode = st.checkbox(
            "Enable Debug Mode",
            value=False,
            help="Show detailed logging information during PDF processing",
        )

        # Add credential path input - simplified
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            credential_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            st.info(f"✅ Using credentials from environment")
            cred_path = credential_path
        else:
            credential_path = st.text_input(
                "Google Service Account Credentials Path (Optional)",
                placeholder="key_vertex.json",
                help="Required for advanced table extraction features. Leave empty for basic table extraction.",
            )
            cred_path = credential_path.strip() if credential_path and credential_path.strip() else None

        pdf_files = st.file_uploader(
            "Upload PDF", type="pdf", accept_multiple_files=True
        )

        if pdf_files:
            if st.button("🚀 Process PDFs", type="primary"):
                if not st.session_state.rag:
                    st.error("❌ Vector database not initialized. Please check settings above.")
                    st.stop()

                # Validate credential path if provided
                if cred_path and not os.path.exists(cred_path):
                    st.error(f"❌ Credentials file not found: {cred_path}")
                    st.stop()

                # Process PDFs using new WDMParser with bytes
                with st.spinner(f"🔄 Processing {len(pdf_files)} PDF files with enhanced parser..."):
                    try:
                        # Run async processing
                        all_documents = asyncio.run(
                            process_pdfs_with_streamlit(pdf_files, cred_path)
                        )

                        if all_documents:
                            # Add documents to vectorstore
                            try:
                                with st.spinner("📥 Adding documents to vector database..."):
                                    st.session_state.rag.add_documents(documents=all_documents)

                                st.success(
                                    f"🎉 Successfully processed and added {len(all_documents)} documents to the knowledge base!\n\n"
                                    f"📚 **Ready for questions!** You can now ask about the content of your PDF files."
                                )

                            except Exception as e:
                                st.error(f"❌ Error adding documents to vector database: {e}")
                                logger.error(f"Vectorstore add_documents error: {e}")
                        else:
                            st.error("❌ No documents were extracted from the PDF files.")

                    except Exception as e:
                        st.error(f"❌ Error processing PDFs: {e}")
                        logger.error(f"PDF processing error: {e}")

    # Main chat interface with 2-column layout
    st.markdown("---")

    # Create two columns: chat (left) and context (right)
    chat_col, context_col = st.columns([2, 1])  # 2:1 ratio

    with chat_col:
        st.subheader("🗨️ Conversation")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    with context_col:
        st.subheader("📋 Retrieved Context")

        # Always show initial state when not processing
        if not st.session_state.messages:
            st.info("💡 Start a conversation to see relevant documents here!")
        else:
            st.info("💡 Context will appear here when asking questions!")

        # Show current database status
        if st.session_state.rag:
            sources = st.session_state.rag.get_unique_sources()
            if sources and sources != ["No sources available"]:
                st.markdown("### 📚 Available Sources:")
                for i, source in enumerate(sources[:10], 1):
                    st.write(f"{i}. {source}")
                if len(sources) > 10:
                    st.write(f"... and {len(sources) - 10} more documents")
            else:
                st.write("📭 No documents in database yet")

    # Chat input at the bottom (outside columns)
    if prompt := st.chat_input("Ask me anything about your documents!"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message immediately
        with chat_col:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Searching knowledge base..."):
                try:
                    if st.session_state.rag:
                        # Sử dụng method __call__ của RAG để lấy đầy đủ thông tin
                        rag_result = st.session_state.rag(prompt)

                        # Extract thông tin từ dictionary result
                        docs = rag_result["docs"]
                        context = rag_result["context"]
                        response = rag_result["response"]
                        query = rag_result["query"]
                        analysis = rag_result["analysis"]

                        print(analysis)

                        # Extract text content if response is a message object
                        if hasattr(response, "content"):
                            response = response.content
                        elif not isinstance(response, str):
                            response = str(response)

                        # Show context in right panel during processing
                        with context_col:
                            st.markdown("### 🔍 Retrieved for Current Query:")

                            if docs:
                                with st.expander("📄 Source Documents", expanded=True):
                                    st.write(f"**Total Retrieved: {len(docs)} documents**")
                                    st.markdown("---")

                                    for i, doc in enumerate(docs, 1):
                                        source = doc.metadata.get("source", "Unknown source")
                                        page = doc.metadata.get("page", "Unknown page")
                                        doc_type = doc.metadata.get("type", "text")

                                        # Show source info with page
                                        st.write(f"**{i}. {source}** (Page {page})")

                                        # Type badge
                                        if doc_type == "table":
                                            st.markdown("🔢 `TABLE`")
                                        else:
                                            st.markdown("📝 `TEXT`")

                                        # Content preview
                                        preview = (
                                            doc.page_content[:300] + "..."
                                            if len(doc.page_content) > 300
                                            else doc.page_content
                                        )
                                        st.markdown(f"*{preview}*")

                                        if i < len(docs):
                                            st.markdown("---")

                                # Show context used for generation (optional debug info)
                                if st.checkbox(
                                    "🔍 Show Full Context",
                                    key=f"show_context_{len(st.session_state.messages)}",
                                ):
                                    with st.expander("📋 Full Context Sent to LLM", expanded=False):
                                        st.text(
                                            context[:2000] + "..."
                                            if len(context) > 2000
                                            else context
                                        )
                            else:
                                st.info("No relevant documents found for this query.")

                    else:
                        with context_col:
                            st.info("📤 Upload PDF documents to start searching!")
                        response = "Please upload PDF documents first to start using the knowledge base."

                except Exception as e:
                    logger.error(f"RAG processing error: {e}")
                    with context_col:
                        st.error(f"Search error: {str(e)}")
                    response = f"Error processing query: {str(e)}"

        # Add assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display assistant response
        with chat_col:
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()

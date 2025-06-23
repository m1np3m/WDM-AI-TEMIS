import atexit
import os
import sys
import time
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

import streamlit as st
from loguru import logger

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.vectorstore import QdrantClientManager, VectorStore
from src.file_loader import PDFLoader
from src.prompts import SYSTEM_MESSAGE, GENERATE_PROMPT

def cleanup_qdrant_clients():
    """Cleanup function to close all Qdrant clients on app shutdown."""
    try:
        QdrantClientManager.close_all_clients()
        logger.info("All Qdrant clients closed successfully")
    except Exception as e:
        logger.warning(f"Error during Qdrant cleanup: {e}")


atexit.register(cleanup_qdrant_clients)

st.set_page_config(
    page_title="WDM-AI-TEMIS - RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ============================== USEFUL FUNCTIONS ==============================

def clear_history():
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.success("History cleared!")


def process_pdf_sync(pdf_file, loader, debug_mode=False):
    """Synchronous wrapper for PDF processing - no Streamlit calls to avoid warnings"""
    start_time = time.time()
    try:
        # Only log to console, avoid Streamlit calls in thread
        logger.info(f"Starting processing {pdf_file.name}")
        splits = loader.load(pdf_file=pdf_file, original_filename=pdf_file.name)
        processing_time = time.time() - start_time
        
        logger.info(f"Completed {pdf_file.name} in {processing_time:.1f}s - {len(splits)} documents")
        
        return {
            'file_name': pdf_file.name,
            'success': True,
            'splits': splits,
            'count': len(splits),
            'processing_time': processing_time,
            'file_size_mb': len(pdf_file.getvalue()) / (1024 * 1024)
        }
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing {pdf_file.name} after {processing_time:.1f}s: {str(e)}")
        return {
            'file_name': pdf_file.name,
            'success': False,
            'error': str(e),
            'splits': [],
            'processing_time': processing_time,
            'file_size_mb': len(pdf_file.getvalue()) / (1024 * 1024)
        }


def format_time(seconds):
    """Format seconds to human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def estimate_remaining_time(completed_files, total_files, elapsed_time):
    """Estimate remaining processing time"""
    if completed_files == 0:
        return "Calculating..."
    
    avg_time_per_file = elapsed_time / completed_files
    remaining_files = total_files - completed_files
    estimated_remaining = avg_time_per_file * remaining_files
    
    return format_time(estimated_remaining)


def process_pdfs_concurrent(pdf_files, credential_path=None, debug_mode=False, temp_dir=None, max_workers=3):
    """Process multiple PDF files concurrently with simple UI feedback"""
    
    # Create loader once
    loader = PDFLoader(
        credential_path=credential_path,
        debug=debug_mode,
        temp_dir=temp_dir,
        enrich=False
    )
    
    all_splits = []
    results = []
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_pdf_sync, pdf_file, loader, debug_mode): pdf_file 
            for pdf_file in pdf_files
        }
        
        # Wait for completion and collect results
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            results.append(result)
            
            if result['success']:
                all_splits.extend(result['splits'])
    
    # Calculate summary stats
    total_time = time.time() - start_time
    successful_files = len([r for r in results if r['success']])
    total_docs = sum(r['count'] for r in results if r['success'])
    text_docs = len([doc for doc in all_splits if doc.metadata.get("type") == "text"])
    table_docs = len([doc for doc in all_splits if doc.metadata.get("type") == "table"])
    
    # Log summary for debugging
    logger.info(f"Processing completed: {successful_files}/{len(pdf_files)} files, {total_docs} documents, {format_time(total_time)}")
    
    return all_splits, results, {
        'successful_files': successful_files,
        'total_files': len(pdf_files),
        'total_docs': total_docs,
        'text_docs': text_docs,
        'table_docs': table_docs,
        'total_time': total_time
    }

def generate_response(source_list: str, prompt: str, context: str) -> str:
    # combined_prompt = SYSTEM_MESSAGE.format(source_list=source_list) + "\n\n" + GENERATE_PROMPT
    combined_prompt = GENERATE_PROMPT
    template = PromptTemplate(
        input_variables=["context", "question"],
        template=combined_prompt
    )

    llm = init_chat_model(
        model_provider="google_vertexai",
        model="gemini-2.0-flash",
        temperature=0.0,
    )    
    
    chain = template | llm
    response = chain.invoke({
        "context": context,
        "question": prompt
    })
    return response

# =================================================================================


# ============================== MAIN FUNCTION ==============================

def main():
    st.title("🤖 WDM-AI-TEMIS - RAG Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize VectorStore in session state 
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        
    with st.sidebar:
        st.header("⚙️ Settings")
        st.button("Clear history", on_click=clear_history)
        
        # VectorStore Configuration
        st.subheader("🗂️ Vector Database Settings")
        
        # Embedding settings
        embedding_type = st.selectbox(
            "Embedding Type",
            options=["huggingface", "vertexai"],
            index=0,
            help="Choose between Google VertexAI or local HuggingFace embeddings"
        )
        
        if embedding_type == "huggingface":
            embedding_model = st.selectbox(
                "HuggingFace Model",
                options=[
                    "BAAI/bge-base-en",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                ],
                index=0,
                help="Select HuggingFace embedding model"
            )
        else:
            embedding_model = st.selectbox(
                "VertexAI Model", 
                options=["text-embedding-004", "textembedding-gecko@003"],
                index=0,
                help="Select Google VertexAI embedding model"
            )
        
        # Vector store settings
        enable_hybrid_search = st.checkbox(
            "Enable Hybrid Search",
            value=True,
            help="Combine dense and sparse vectors for better retrieval"
        )
        
        chunk_type = st.selectbox(
            "Text Chunking Strategy",
            options=["character", "recursive"],
            index=0,
            help="Choose text splitting strategy: character (simple) or recursive (smart)"
        )
        
        use_memory = st.checkbox(
            "Use Memory Storage",
            value=False,
            help="⚠️ Memory storage: Data will be LOST when you refresh (F5) the page. Uncheck for persistent storage that survives page refreshes."
        )
        
        collection_name = st.text_input(
            "Collection Name",
            value="wdm_ai_temis",
            help="Name for the vector database collection"
        )
        
        # Initialize or update VectorStore based on settings
        vectorstore_key = f"vs_{embedding_type}_{embedding_model}_{enable_hybrid_search}_{chunk_type}_{use_memory}_{collection_name}"
        
        # Check if settings changed and need to reinitialize
        if (st.session_state.vectorstore is None or 
            "current_vectorstore_key" not in st.session_state or 
            st.session_state.current_vectorstore_key != vectorstore_key):
            
            with st.spinner("Initializing Vector Database..."):
                try:
                    # Create persist directory - handle memory vs persistent storage
                    if use_memory:
                        # For memory storage, use a temp directory but set use_memory=True
                        persist_dir = "./temp_qdrant_db"
                    else:
                        persist_dir = "./qdrant_db"
                    
                    # Initialize VectorStore
                    st.session_state.vectorstore = VectorStore(
                        persist_directory=persist_dir,
                        collection_name=collection_name,
                        embedding_type=embedding_type,
                        embedding_model=embedding_model,
                        enable_hybrid_search=enable_hybrid_search,
                        chunk_type=chunk_type,
                        use_memory=use_memory
                    )
                    
                    st.session_state.current_vectorstore_key = vectorstore_key
                    logger.info(f"VectorStore initialized with key: {vectorstore_key}")
                    
                except Exception as e:
                    st.error(f"❌ Error initializing VectorStore: {e}")
                    logger.error(f"VectorStore initialization error: {e}")
                    st.stop()
        
        # Display current configuration
        with st.expander("📊 Current Vector Database Config", expanded=False):
            st.write(f"**Storage:** {'Memory (temporary)' if use_memory else 'Persistent'}")
            st.write(f"**Collection:** {collection_name}")
            st.write(f"**Embedding:** {embedding_type} - {embedding_model}")
            st.write(f"**Hybrid Search:** {'Enabled' if enable_hybrid_search else 'Disabled'}")
            st.write(f"**Chunk Type:** {chunk_type}")
            
            if st.session_state.vectorstore:
                sources = st.session_state.vectorstore.get_unique_sources()
                if sources and sources != ["No sources available"]:
                    st.write(f"**Sources:** {len(sources)} document(s)")
                else:
                    st.write("**Sources:** No documents loaded")
        
        # Vector Database Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Database"):
                if st.session_state.vectorstore:
                    with st.spinner("Clearing database..."):
                        try:
                            st.session_state.vectorstore.clear_vectorstore()
                            clear_history()
                            st.success("✅ Database cleared!")
                        except Exception as e:
                            st.error(f"❌ Error clearing database: {e}")
                else:
                    st.warning("No database to clear")
        
        with col2:
            if st.button("🔄 Reinitialize"):
                # Force reinitialize by clearing session state
                if "vectorstore" in st.session_state:
                    del st.session_state.vectorstore
                if "current_vectorstore_key" in st.session_state:
                    del st.session_state.current_vectorstore_key
                st.rerun()
        
        st.markdown("---")
        st.subheader("📄 Upload PDF")
        
        # Add debug mode option
        debug_mode = st.checkbox(
            "Enable Debug Mode",
            value=False,
            help="Show detailed logging information during PDF processing"
        )
        
        # Add credential path input
        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            credential_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            st.info(f"✅ Using credentials from environment: {credential_path}")
            cred_path = credential_path
        else:   
            credential_path = st.text_input(
                "Google Service Account Credentials Path (Optional)",
                placeholder="/path/to/service-account-key.json",
                help="Required for advanced table extraction features (merge_span_tables, enrich). Leave empty for basic table extraction."
            )
            cred_path = credential_path.strip() if credential_path and credential_path.strip() else None
        
        # Add custom temp directory option
        temp_dir = st.text_input(
            "Custom Temporary Directory (Optional)",
            placeholder="/tmp/pdf_processing",
            help="Specify a custom directory for temporary files. Leave empty to use system default."
        )
        
        # Add max workers option for concurrent processing
        max_workers = st.slider(
            "Max Concurrent Workers",
            min_value=1,
            max_value=8,
            value=3,
            help="Number of PDFs to process simultaneously. Higher values use more resources."
        )
        
        pdf_files = st.file_uploader(
            "Upload PDF", type="pdf", accept_multiple_files=True
        )
        if pdf_files:

            
            # Process PDFs button
            if st.button("🚀 Process PDFs", type="primary") and pdf_files:
                # Check if vectorstore is ready
                if not st.session_state.vectorstore:
                    st.error("❌ Vector database not initialized. Please check settings above.")
                    st.stop()
                    
                # Validate credential path if provided
                if cred_path and not os.path.exists(cred_path):
                    st.error(f"❌ Credentials file not found: {cred_path}")
                    st.stop()
                
                # Validate temp directory if provided
                temp_dir_path = temp_dir.strip() if temp_dir and temp_dir.strip() else None
                if temp_dir_path and not os.path.exists(temp_dir_path):
                    try:
                        os.makedirs(temp_dir_path, exist_ok=True)
                        st.info(f"📁 Created temporary directory: {temp_dir_path}")
                    except Exception as e:
                        st.error(f"❌ Failed to create temp directory {temp_dir_path}: {e}")
                        st.stop()
                
                # Process PDFs concurrently with spinner
                with st.spinner(f"Processing {len(pdf_files)} PDF files..."):
                    all_splits, results, stats = process_pdfs_concurrent(
                        pdf_files=pdf_files,
                        credential_path=cred_path,
                        debug_mode=debug_mode,
                        temp_dir=temp_dir_path,
                        max_workers=max_workers
                    )
                
                if all_splits:
                    # Add documents to vectorstore
                    try:
                        with st.spinner("Adding documents to vector database..."):
                            st.session_state.vectorstore.add_documents(documents=all_splits)
                        
                        st.success(
                            f"🎉 Successfully processed {stats['successful_files']}/{stats['total_files']} PDF(s)!\n\n"
                            f"📄 **Total documents:** {stats['total_docs']} "
                            f"({stats['text_docs']} text, {stats['table_docs']} tables) | "
                            f"⏱️ **Time:** {format_time(stats['total_time'])}\n\n"
                            f"✅ **Added to vector database:** {len(all_splits)} documents"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error adding documents to vector database: {e}")
                        logger.error(f"Vectorstore add_documents error: {e}")
                    
                else:
                    st.error("❌ No documents were extracted from the PDF files.")
            
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
        if st.session_state.vectorstore:
            sources = st.session_state.vectorstore.get_unique_sources()
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
        
        # Generate response and show context during processing
        if st.session_state.vectorstore and st.session_state.vectorstore.retriever:
            with st.spinner("Searching knowledge base..."):
                try:
                    # Get relevant documents
                    docs = st.session_state.vectorstore.retriever.invoke(prompt)
                    
                    # Show context in right panel during processing
                    with context_col:
                        st.markdown("### 🔍 Retrieved for Current Query:")
                        
                        if docs:
                            with st.expander("📄 Source Documents", expanded=True):
                                st.write(f"**Total Retrieved: {len(docs)} documents**")
                                st.markdown("---")
                                
                                for i, doc in enumerate(docs, 1):
                                    source = doc.metadata.get('source', 'Unknown source')
                                    page = doc.metadata.get('page', 'Unknown page')
                                    doc_type = doc.metadata.get('type', 'text')
                                    
                                    # Show source info with page
                                    st.write(f"**{i}. {source}** (Page {page})")
                                    
                                    # Type badge
                                    if doc_type == 'table':
                                        st.markdown("🔢 `TABLE`")
                                    else:
                                        st.markdown("📝 `TEXT`")
                                    
                                    # Content preview
                                    preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                    st.markdown(f"*{preview}*")
                                    
                                    if i < len(docs):
                                        st.markdown("---")
                            
                            # Generate response using LLM with ALL retrieved documents
                            # Format context with clear separators for each document
                            context_parts = []
                            source_list = []
                            
                            for i, doc in enumerate(docs, 1):  # Use ALL documents, not just top 5
                                source = doc.metadata.get('source', 'Unknown source')
                                page = doc.metadata.get('page', 'Unknown page')
                                doc_type = doc.metadata.get('type', 'text')
                                
                                # Create source list entry with basic info
                                source_list.append(f"{i}. {source} (Page {page}, Type: {doc_type})")
                                
                                # Format all metadata for LLM context
                                metadata_lines = []
                                for key, value in doc.metadata.items():
                                    metadata_lines.append(f"{key.title()}: {value}")
                                metadata_str = "\n".join(metadata_lines)
                                
                                # Format document with clear header including all metadata
                                doc_header = f"=== DOCUMENT {i} ===\nMETADATA:\n{metadata_str}\n" + "="*50
                                doc_content = doc.page_content.strip()
                                doc_footer = f"{'='*50}\nEND OF DOCUMENT {i}\n{'='*50}"
                                
                                context_parts.append(f"{doc_header}\n\n{doc_content}\n\n{doc_footer}")
                            
                            # Join all documents with clear separators
                            context = "\n\n".join(context_parts)
                            source_list_str = "\n".join(source_list)
                        
                            # Generate response using the LLM
                            with st.spinner("Generating response..."):
                                try:
                                    response = generate_response(
                                        source_list=source_list_str,
                                        prompt=prompt,
                                        context=context
                                    )
                                    # Extract text content if response is a message object
                                    if hasattr(response, 'content'):
                                        response = response.content
                                    elif not isinstance(response, str):
                                        response = str(response)
                                        
                                except Exception as e:
                                    logger.error(f"LLM generation error: {e}")
                                    response = f"❌ Error generating response: {str(e)}"
                        else:
                            st.info("No relevant documents found for this query.")
                            # Still generate a response using LLM even without context
                            with st.spinner("Generating response..."):
                                try:
                                    response = generate_response(
                                        source_list="No specific documents found",
                                        prompt=prompt,
                                        context="No relevant context found in the knowledge base."
                                    )
                                    # Extract text content if response is a message object
                                    if hasattr(response, 'content'):
                                        response = response.content
                                    elif not isinstance(response, str):
                                        response = str(response)
                                except Exception as e:
                                    logger.error(f"LLM generation error: {e}")
                                    response = f"❌ Error generating response: {str(e)}"
                        
                except Exception as e:
                    logger.error(f"Retrieval error: {e}")
                    with context_col:
                        st.error(f"Search error: {str(e)}")
                    response = f"Error searching knowledge base: {str(e)}"
        else:
            with context_col:
                st.info("📤 Upload PDF documents to start searching!")
            # Generate response using LLM without context
            with st.spinner("Generating response..."):
                try:
                    response = generate_response(
                        source_list="No documents uploaded yet",
                        prompt=prompt,
                        context="No documents have been uploaded to the knowledge base yet."
                    )
                    # Extract text content if response is a message object
                    if hasattr(response, 'content'):
                        response = response.content
                    elif not isinstance(response, str):
                        response = str(response)
                except Exception as e:
                    logger.error(f"LLM generation error: {e}")
                    response = f"❌ Error generating response: {str(e)}"
        
        # Add assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with chat_col:
            with st.chat_message("assistant"):
                st.markdown(response)



if __name__ == "__main__":
    main()

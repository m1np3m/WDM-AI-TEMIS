import asyncio
import os
from src import RAG
from src.WDMParser.WDMParser import WDMPDFParser
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList, Filter, FieldCondition, MatchValue
    
def check_collection_exists_and_has_data(collection_name: str, persist_dir: str) -> bool:
    """
    Check if collection exists and has documents
    
    Returns:
        bool: True if collection exists and has data, False otherwise
    """
    try:
        client = QdrantClient(path=persist_dir)
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            print(f"Collection '{collection_name}' does not exist")
            return False
        
        # Check if collection has data
        collection_info = client.get_collection(collection_name)
        point_count = collection_info.points_count
        
        if point_count == 0:
            print(f"Collection '{collection_name}' exists but is empty")
            return False
        
        print(f"Collection '{collection_name}' exists with {point_count} documents")
        return True
        
    except Exception as e:
        print(f"Error checking collection: {e}")
        return False
    finally:
        try:
            client.close()
        except:
            pass

async def process_pdfs_with_new_parser(pdf_files: list, credential_path: str = "key_vertex.json") -> list:
    """
    Process PDF files using the new WDMParser with bytes support
    
    Args:
        pdf_files: List of PDF file paths
        credential_path: Path to Google Cloud credentials
        
    Returns:
        List of processed documents
    """
    print("\n=== Processing PDFs with New WDMParser ===")
    
    # Create parser settings
    settings = WDMPDFParser.create_settings(
        credential_path=credential_path,
        debug=True,
        debug_level=1,
        max_concurrent_files=2,  # Conservative for stability
        max_memory_mb=4096,      # 4GB limit
        batch_size=3,
        cleanup_interval=2
    )
    
    parser = WDMPDFParser(settings=settings)
    
    # Filter existing files
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    if not existing_files:
        print("No existing PDF files found!")
        return []
    
    print(f"Found {len(existing_files)} PDF files to process")
    
    # Process all files asynchronously
    try:
        results = await parser.process_documents(
            pdf_documents=existing_files,
            merge_span_tables=True,
            enrich=False,  # Disable for faster processing
            extract_text=True,
            return_failed=False
        )
        
        # Combine all documents from all files
        all_documents = []
        total_tables = 0
        total_text = 0
        
        # Handle potential tuple return from process_documents
        if isinstance(results, tuple):
            results_dict, failed_files = results
        else:
            results_dict = results
        
        for file_path, documents in results_dict.items():
            table_docs = [d for d in documents if d.metadata.get('type') == 'table']
            text_docs = [d for d in documents if d.metadata.get('type') == 'text']
            
            total_tables += len(table_docs)
            total_text += len(text_docs)
            all_documents.extend(documents)
            
            print(f"  {os.path.basename(file_path)}: {len(table_docs)} tables, {len(text_docs)} text blocks")
        
        print(f"\nTotal processed: {total_tables} tables, {total_text} text blocks, {len(all_documents)} documents")
        
        # Show memory usage
        memory_info = parser.get_memory_info()
        if 'error' not in memory_info:
            print(f"Memory usage: {memory_info['rss_mb']:.1f}MB (within limit: {memory_info['within_limit']})")
        
        return all_documents
        
    except Exception as e:
        print(f"Error processing PDFs: {e}")
        return []

async def main():
    # Configuration
    collection_name = "wdm-ai-temis"
    persist_dir = "./main_test_db"
    credential_path = "key_vertex.json"
    
    # PDF files to process
    pdf_files = [
        "data/experiment_data/CA Warn Report.pdf",
        "data/experiment_data/b014b8ca3c8ee543b655c29747cc6090.pdf",
        "data/experiment_data/national-capitals.pdf"
    ]
    
    print("=== WDM AI TEMIS - Enhanced PDF Processing ===")
    print(f"Collection: {collection_name}")
    print(f"Persist Directory: {persist_dir}")
    
    # Check if collection already exists and has data
    collection_has_data = check_collection_exists_and_has_data(collection_name, persist_dir)
    
    # Create RAG instance
    my_rag = RAG(
        embedding_type="huggingface",
        embedding_model="BAAI/bge-base-en", 
        enable_hybrid_search=True,
        chunk_type="character",
        use_memory=False,
        collection_name=collection_name,
        persist_dir=persist_dir,
        use_reranker=True
    )
    
    # Process PDFs only if collection doesn't have data
    if not collection_has_data:
        print("\n🔄 Collection is empty or doesn't exist. Processing PDFs...")
        
        # Process PDFs with new parser
        all_documents = await process_pdfs_with_new_parser(pdf_files, credential_path)
        
        if all_documents:
            print(f"\n📥 Adding {len(all_documents)} documents to vectorstore...")
            my_rag.add_documents(all_documents)
            print("✅ Documents added successfully!")
        else:
            print("❌ No documents were processed. Exiting.")
            return
    else:
        print("\n✅ Collection already has data. Skipping PDF processing.")
        print("💡 To reprocess PDFs, delete the collection or use a different collection name.")
    
    # Show collection statistics
    try:
        unique_sources = my_rag.get_unique_sources()
        print(f"\n📊 Available sources: {unique_sources}")
    except Exception as e:
        print(f"Could not get sources: {e}")
    
    # Start a conversation
    conversation_id = my_rag.start_conversation(user_id="cli_user")
    print(f"\n💬 Started conversation: {conversation_id}")
    
    # Interactive query loop with conversation support
    print("\n🎯 Ready for queries! Available commands:")
    print("  - 'exit' : Quit the application")
    print("  - 'stats' : Show collection statistics")
    print("  - 'history' : Show conversation history")
    print("  - 'clear' : Clear current conversation") 
    print("  - 'new' : Start a new conversation")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n💬 Enter your query: ").strip()
            
            if query.lower() == "exit":
                print("👋 Goodbye!")
                break
            elif query.lower() == "stats":
                try:
                    # Show collection statistics
                    client = QdrantClient(path=persist_dir)
                    collection_info = client.get_collection(collection_name)
                    print(f"\n📈 Collection Statistics:")
                    print(f"  - Documents: {collection_info.points_count}")
                    print(f"  - Collection: {collection_name}")
                    print(f"  - Sources: {my_rag.get_unique_sources()}")
                    client.close()
                except Exception as e:
                    print(f"Error getting stats: {e}")
                continue
            elif query.lower() == "history":
                try:
                    history = my_rag.get_conversation_history()
                    print(f"\n📜 Conversation History ({len(history)} messages):")
                    print("-" * 40)
                    for i, msg in enumerate(history, 1):
                        role_emoji = "👤" if msg["role"] == "user" else "🤖"
                        content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        print(f"{i}. {role_emoji} {msg['role'].title()}: {content_preview}")
                    print("-" * 40)
                except Exception as e:
                    print(f"Error getting history: {e}")
                continue
            elif query.lower() == "clear":
                try:
                    my_rag.clear_conversation()
                    print("🗑️ Conversation cleared!")
                except Exception as e:
                    print(f"Error clearing conversation: {e}")
                continue
            elif query.lower() == "new":
                try:
                    conversation_id = my_rag.start_conversation(user_id="cli_user")
                    print(f"🆕 Started new conversation: {conversation_id}")
                except Exception as e:
                    print(f"Error starting new conversation: {e}")
                continue
            elif not query:
                print("❓ Please enter a valid query")
                continue
            
            # Process query with conversation
            print("🔍 Processing query...")
            response = my_rag.chat(query)  # Use chat method which includes conversation
            
            print("\n📝 Response:")
            print("-" * 40)
            print(response['response'])
            print("-" * 40)
            
            # Show conversation context if available
            if response.get('conversation_context'):
                print(f"💭 Conversation context: {len(response['conversation_context'])} chars")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")

if __name__ == "__main__":
    asyncio.run(main())
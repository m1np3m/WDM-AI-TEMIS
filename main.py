import asyncio
from src import RAG
    
async def main():
    # Tạo RAG instance
    my_rag = RAG(
        embedding_type="huggingface",
        embedding_model="BAAI/bge-base-en",
        enable_hybrid_search=True,
        chunk_type="character",
        use_memory=False,
        collection_name="wdm-ai-temis",
        persist_dir="vs_main_script_huggingface_BAAI/bge-base-en_True_character_False_wdm-ai-temis"
    )
    
    # Load PDFs và add vào vectorstore
    pdf_files = [
        "data/experiment_data/CA Warn Report.pdf",
        "data/experiment_data/b014b8ca3c8ee543b655c29747cc6090.pdf",
        "data/experiment_data/national-capitals.pdf"
    ]
    
    splits, results, stats = await my_rag.load_pdfs(pdf_files)
    my_rag.add_documents(splits)
    
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break
        response = my_rag(query)
        print(response.content)
        print("-"*100)

if __name__ == "__main__":
    asyncio.run(main())
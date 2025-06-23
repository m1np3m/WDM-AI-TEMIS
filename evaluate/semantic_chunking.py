from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


def semanticlangchain(all_page_documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",   # split when distance exceeds this percentile
        breakpoint_threshold_amount=95.0,           # default 95th percentile
        min_chunk_size=50, # split on newlines for table rows
    )

    all_text_chunk_documents = text_splitter.split_documents(all_page_documents)
    print(f"Generated {len(all_text_chunk_documents)} semantic chunks from table data")
    
    return all_text_chunk_documents

# Cấu hình của vectorstore
VECTORSTORE_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 128,
    "chunk_type": "character",
    "k": 7,
    "separators": ["\n\n", "\n", ". ", "! ", "? ", ":", ";", " "],
    "embedding_type": "huggingface",
    "embedding_model": "BAAI/bge-base-en",
    "enable_hybrid_search": False,
    "separator": "\n\n",
}

# Cấu hình của WDMParser
IGNORE_TABLES = True
ENRICH_TABLES = False

# Cấu hình của Reranker
USE_REANKER = True
REANKER_MODEL_NAME = "bce"


# Cấu hình của LLM
QUERY_ANALYSIS_MODEL = {
    "model_provider": "google_vertexai",
    "model_name": "gemini-2.0-flash",
}

QUERY_ANALYSIS_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 1024,
}
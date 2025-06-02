from langchain.text_splitter import RecursiveCharacterTextSplitter
import json


def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """
    Chia văn bản thành các chunk nhỏ.

    Args:
        text (str): Văn bản cần chia.
        chunk_size (int): Kích thước mỗi chunk.
        overlap (int): Độ chồng lấn giữa các chunk.

    Returns:
        list: Danh sách các chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_embedding(embedding_model, text: str) -> list[float]:
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)

    return embedding.tolist()



def vector_search(embedding_model, user_query, collection, limit=2):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """
    query_embedding = get_embedding(embedding_model ,user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."
    
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 400,
            "limit": limit,
        }
    }

    unset_stage = {
        "$unset": "embedding"
    }

    project_stage = {
        "$project": {
            "_id": 0,
            "table": 1,
            "chunk": 1,
            "score": {
                "$meta": "vectorSearchScore"
            }
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]
    results = collection.aggregate(pipeline)

    return list(results)


def get_search_result(embedding_model, query, collection):
    get_knowledge = vector_search(embedding_model, query, collection, 2)
    search_result = ""
    i = 0
    for result in get_knowledge:
        print(result)
        if 'chunk' in result:
            i += 1
            summary = result['chunk']  # Có thể rút gọn hơn nếu cần
            table = result.get('table', '')  # Nếu có trường 'table'

            search_result += f"\n {i}) Summary: {summary}\n Table: {table}\n\n"

        
    return search_result
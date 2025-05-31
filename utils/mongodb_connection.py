import pymongo

def get_mongo_client(mongo_uri):
  """Establish connection to the MongoDB."""
  try:
    client = pymongo.MongoClient(mongo_uri, appname="devrel.content.python")
    print("Connection to MongoDB successful")
    return client
  except pymongo.errors.ConnectionFailure as e:
    print(f"Connection failed: {e}")
    return None

def save_chunks_to_db(chunks, mongo_client, db_name="Kyanon", collection_name="RAG"):
    """
    Lưu các chunk và embedding vào MongoDB.

    Args:
        chunks (list): Danh sách các chunk và embedding (dạng [(chunk, embedding)]).
        mongo_client (MongoClient): Đối tượng MongoDB client đã kết nối.
        db_name (str): Tên database MongoDB.
        collection_name (str): Tên collection trong MongoDB.
    """
    try:
        db = mongo_client[db_name]
        collection = db[collection_name]

        documents = [{"embedding": embedding, "chunk": chunk } for chunk, embedding in chunks]

        result = collection.insert_many(documents)
        print(f"Đã lưu {len(result.inserted_ids)} chunk và embedding vào MongoDB collection '{collection_name}'.")
    except Exception as e:
        print(f"Lỗi khi lưu vào MongoDB: {e}")
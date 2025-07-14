import hashlib
import os
from typing import List, Union, Optional, Dict

from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from loguru import logger

from .setting import VECTORSTORE_CONFIG


class QdrantClientManager:
    # Manages Qdrant client instances to avoid conflicts
    _clients: Dict[str, QdrantClient] = {}
    
    @classmethod
    def get_client(cls, path: str) -> QdrantClient:
        if path not in cls._clients:
            logger.info(f"Creating new Qdrant client for path: {path}")
            cls._clients[path] = QdrantClient(path=path)
        return cls._clients[path]
    
    @classmethod
    def close_all_clients(cls):
        for path, client in cls._clients.items():
            try:
                client.close()
                logger.info(f"Closed Qdrant client for path: {path}")
            except Exception as e:
                logger.warning(f"Error closing client for {path}: {e}")
        cls._clients.clear()


class TextSplitter:
    def __init__(
        self,
        chunk_type: str = "recursive",
        separators: List[str] = ["\n\n", "\n", ". ", "! ", "? ", ":", ";", " "],
        chunk_size: int = VECTORSTORE_CONFIG["chunk_size"],
        chunk_overlap: int = VECTORSTORE_CONFIG["chunk_overlap"],
        separator: str = "\n\n",
    ) -> None:
        self.chunk_type = chunk_type
        
        if chunk_type == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )
        elif chunk_type == "character":
            self.splitter = CharacterTextSplitter(
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )
        else:
            raise ValueError(f"Unsupported chunk_type: {chunk_type}. Use 'recursive' or 'character'.")

    def __call__(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class VectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        documents: List[Document] = None,
        embedding_type: str = "vertexai",
        embedding_model: str = "text-embedding-004",
        enable_hybrid_search: bool = False,
        chunk_type: str = "recursive",
        use_memory: bool = False,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.enable_hybrid_search = enable_hybrid_search
        self.chunk_type = chunk_type
        self.use_memory = use_memory
        
        self.embeddings = self._initialize_embeddings()
        self.sparse_embeddings = None
        
        if self.enable_hybrid_search:
            self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        self.text_splitter = TextSplitter(chunk_type=chunk_type)
        
        self.sources = set()
        self.vectorstore = None
        self.retriever = None
        self._client = None
        
        self.vectorstore, self.retriever = self.create_vectorstore(docs_list=documents)

        if documents:
            self._update_sources(documents)
    
    def _initialize_embeddings(self):
        if self.embedding_type == "vertexai":
            return VertexAIEmbeddings(model=self.embedding_model)
        elif self.embedding_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
    
    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            if self.use_memory:
                self._client = QdrantClient(location=":memory:")
                logger.info("Created in-memory Qdrant client")
            else:
                self._client = QdrantClientManager.get_client(self.persist_directory)
        return self._client

    def _update_sources(self, documents: List[Document]):
        for doc in documents:
            if "source" in doc.metadata:
                self.sources.add(doc.metadata["source"])

    def _get_embedding_size(self) -> int:
        try:
            test_embedding = self.embeddings.embed_query("test")
            return len(test_embedding)
        except Exception as e:
            logger.warning(f"Could not determine embedding size, using default 768: {e}")
            return 768

    def _ensure_collection_exists(self, client: QdrantClient):
        try:
            collections = client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            if not collection_exists:
                embedding_size = self._get_embedding_size()
                logger.info(f"Creating Qdrant collection '{self.collection_name}' with embedding size {embedding_size}")
                
                if self.enable_hybrid_search:
                    client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            "dense": VectorParams(size=embedding_size, distance=Distance.COSINE)
                        },
                        sparse_vectors_config={
                            "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                        }
                    )
                    logger.info(f"Created hybrid collection '{self.collection_name}' with dense + sparse vectors")
                else:
                    client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=embedding_size,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created dense collection '{self.collection_name}'")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except ValueError as e:
            if "already exists" in str(e):
                logger.info(f"Collection '{self.collection_name}' already exists")
            else:
                logger.error(f"Error ensuring collection exists: {e}")
                raise
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def check_vectorstore_exists(self) -> bool:
        try:
            client = self._get_client()
            collections = client.get_collections()
            return any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
        except Exception as e:
            logger.warning(f"Error checking if vectorstore exists: {e}")
            return False

    def _load_existing_sources(self):
        try:
            if self.vectorstore is None:
                return
            
            client = self._get_client()
            scroll_result = client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            self.sources.clear()
            for point in scroll_result[0]:
                if point.payload and "metadata" in point.payload:
                    metadata = point.payload["metadata"]
                    if isinstance(metadata, dict) and "source" in metadata:
                        self.sources.add(metadata["source"])
                        
        except Exception as e:
            logger.warning(f"Error loading existing sources: {e}")

    def create_vectorstore(
        self,
        reload_vectordb: bool = True,
        docs_list: Union[Document, List[Document]] = None,
    ):
        os.makedirs(self.persist_directory, exist_ok=True)
        
        client = self._get_client()
        vectorstore_exists = self.check_vectorstore_exists()

        if reload_vectordb and vectorstore_exists:
            logger.info("Loading existing Qdrant vector database...")
            try:
                self._ensure_collection_exists(client)
                if self.enable_hybrid_search:
                    self.vectorstore = QdrantVectorStore(
                        client=client,
                        collection_name=self.collection_name,
                        embedding=self.embeddings,
                        sparse_embedding=self.sparse_embeddings,
                        retrieval_mode=RetrievalMode.HYBRID,
                        vector_name="dense",
                        sparse_vector_name="sparse"
                    )
                else:
                    self.vectorstore = QdrantVectorStore(
                        client=client,
                        collection_name=self.collection_name,
                        embedding=self.embeddings,
                        retrieval_mode=RetrievalMode.DENSE
                    )
                
                self._load_existing_sources()
                
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": VECTORSTORE_CONFIG["k"]})
                logger.info(f"Loaded existing vectorstore with {len(self.sources)} sources")
                return self.vectorstore, self.retriever
                
            except Exception as e:
                logger.error(f"Error loading existing vectorstore: {e}")
                logger.info("Creating new vectorstore instead...")
                vectorstore_exists = False

        if reload_vectordb and not vectorstore_exists:
            logger.warning(
                "Reload_vectordb flag is True but no existing vectorstore found. Creating a new one..."
            )

        if docs_list is None:
            logger.info(
                "No documents provided, creating an empty vectorstore with placeholder..."
            )
            placeholder_doc = Document(
                page_content="Placeholder content", metadata={"source": "placeholder"}
            )
            doc_splits = self.text_splitter([placeholder_doc])
        else:
            logger.info(
                f"Creating vectorstore from {len(docs_list) if isinstance(docs_list, list) else 1} document(s)..."
            )
            docs_to_process = docs_list if isinstance(docs_list, list) else [docs_list]
            
            text_documents = [doc for doc in docs_to_process if doc.metadata.get("type") == "text"]
            table_documents = [doc for doc in docs_to_process if doc.metadata.get("type") == "table"]
            
            split_text_documents = self.text_splitter(text_documents) if text_documents else []
            
            doc_splits = split_text_documents + table_documents
            
            logger.info(f"Text documents split into {len(split_text_documents)} chunks")
            logger.info(f"Table documents kept whole: {len(table_documents)} tables")

        self._ensure_collection_exists(client)

        if self.enable_hybrid_search:
            self.vectorstore = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                sparse_embedding=self.sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                vector_name="dense",
                sparse_vector_name="sparse"
            )
        else:
            self.vectorstore = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                retrieval_mode=RetrievalMode.DENSE
            )
        
        if doc_splits:
            self.vectorstore.add_documents(documents=doc_splits)
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": VECTORSTORE_CONFIG["k"]})
        logger.info(f"Created new Qdrant vectorstore with {len(doc_splits)} documents")
        return self.vectorstore, self.retriever

    def add_documents(self, documents: List[Document]):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        
        try:
            client = self._get_client()
            
            text_documents = [doc for doc in documents if doc.metadata.get("type") == "text"]
            table_documents = [doc for doc in documents if doc.metadata.get("type") == "table"]
            image_documents = [doc for doc in documents if doc.metadata.get("type") == "image"]
            
            split_text_documents = self.text_splitter(text_documents) if text_documents else []
            
            processed_docs = split_text_documents + table_documents + image_documents
            
            new_docs = []
            new_ids = []
            
            for doc in processed_docs:
                doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
                
                try:
                    existing_point = client.retrieve(
                        collection_name=self.collection_name,
                        ids=[doc_id]
                    )
                    if not existing_point:
                        new_docs.append(doc)
                        new_ids.append(doc_id)
                except Exception:
                    new_docs.append(doc)
                    new_ids.append(doc_id)
            
            if new_docs:
                self.vectorstore.add_documents(documents=new_docs, ids=new_ids)
                self._update_sources(new_docs)
                logger.info(f"Successfully added {len(new_docs)} new documents to vectorstore "
                           f"({len(split_text_documents)} text chunks, {len(table_documents)} tables, {len(image_documents)} images)")
            else:
                logger.info("No new documents to add; all were duplicates")
                
        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {e}")
            raise

    def retrieve_documents(
        self,
        query: str,
        filter_sources: Optional[List[str]] = None,
        filter_types: Optional[List[str]] = None,
        num_docs: Optional[int] = None,
    ):
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")
        
        filter_conditions = []
        
        if filter_sources:
            source_conditions = [
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source)
                )
                for source in filter_sources
            ]
            if len(source_conditions) == 1:
                filter_conditions.append(source_conditions[0])
            else:
                filter_conditions.append(
                    models.Filter(
                        should=source_conditions
                    )
                )
        
        if filter_types:
            type_conditions = [
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value=type_val)
                )
                for type_val in filter_types
            ]
            if len(type_conditions) == 1:
                filter_conditions.append(type_conditions[0])
            else:
                filter_conditions.append(
                    models.Filter(
                        should=type_conditions
                    )
                )
        
        if filter_conditions:
            if len(filter_conditions) == 1:
                final_filter = filter_conditions[0]
            else:
                final_filter = models.Filter(
                    must=filter_conditions
                )
            
            try:
                if num_docs:
                    k = num_docs
                else:
                    k = VECTORSTORE_CONFIG["k"]
                    
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=final_filter
                )
                logger.info(f"Retrieved {len(results)} documents with filters - sources: {filter_sources}, types: {filter_types}")
                return results
            except Exception as e:
                logger.warning(f"Error with filtered search, falling back to unfiltered: {e}")
                return self.retriever.invoke(query)
        else:
            return self.retriever.invoke(query)

    def get_unique_sources(self) -> List[str]:
        return sorted(list(self.sources)) if self.sources else ["No sources available"]

    def clear_vectorstore(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")

        try:
            client = self._get_client()
            
            try:
                client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}'")
            except Exception as e:
                logger.warning(f"Error deleting collection: {e}")
            
            self._ensure_collection_exists(client)

            self.sources.clear()

            placeholder_doc = Document(
                page_content="Placeholder content", metadata={"source": "placeholder"}
            )
            doc_splits = self.text_splitter([placeholder_doc])

            if self.enable_hybrid_search:
                self.vectorstore = QdrantVectorStore(
                    client=client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                    sparse_embedding=self.sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID,
                    vector_name="dense",
                    sparse_vector_name="sparse"
                )
            else:
                self.vectorstore = QdrantVectorStore(
                    client=client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                    retrieval_mode=RetrievalMode.DENSE
                )
            
            if doc_splits:
                self.vectorstore.add_documents(documents=doc_splits)
            
            self._update_sources(doc_splits)
            logger.info("Added placeholder document to vectorstore")

            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": VECTORSTORE_CONFIG["k"]})

        except Exception as e:
            logger.error(f"Error clearing vectorstore: {e}")
            raise

    def __del__(self):
        try:
            if hasattr(self, '_client') and self._client is not None:
                pass
        except Exception as e:
            logger.warning(f"Error in vectorstore cleanup: {e}")

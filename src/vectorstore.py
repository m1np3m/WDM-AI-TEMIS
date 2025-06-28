import hashlib
import os
from typing import List, Union, Optional, Dict, Any

from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from loguru import logger

from .file_loader import TextSplitter
from .setting import K


class QdrantClientManager:
    """Singleton pattern to ensure only one Qdrant client per path to avoid conflicts."""
    _clients: Dict[str, QdrantClient] = {}
    
    @classmethod
    def get_client(cls, path: str) -> QdrantClient:
        """Get or create a Qdrant client for the given path."""
        if path not in cls._clients:
            logger.info(f"Creating new Qdrant client for path: {path}")
            cls._clients[path] = QdrantClient(path=path)
        return cls._clients[path]
    
    @classmethod
    def close_all_clients(cls):
        """Close all clients and clear the cache."""
        for path, client in cls._clients.items():
            try:
                client.close()
                logger.info(f"Closed Qdrant client for path: {path}")
            except Exception as e:
                logger.warning(f"Error closing client for {path}: {e}")
        cls._clients.clear()


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
        use_memory: bool = False,  # New option for in-memory storage
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.enable_hybrid_search = enable_hybrid_search
        self.chunk_type = chunk_type
        self.use_memory = use_memory
        
        # Initialize embeddings based on type
        self.embeddings = self._initialize_embeddings()
        self.sparse_embeddings = None
        
        # Initialize sparse embeddings for hybrid search
        if self.enable_hybrid_search:
            self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        self.sources = set()
        self.vectorstore = None
        self.retriever = None
        self._client = None
        
        # Initialize the vectorstore
        self.vectorstore, self.retriever = self.create_vectorstore(docs_list=documents)

        if documents:
            self._update_sources(documents)
    
    def _initialize_embeddings(self):
        """Initialize embeddings based on the specified type."""
        if self.embedding_type == "vertexai":
            return VertexAIEmbeddings(model=self.embedding_model)
        elif self.embedding_type == "huggingface":
            # Use CPU for HuggingFace to avoid meta tensor issues with CUDA
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_client(self) -> QdrantClient:
        """Get the Qdrant client, ensuring only one instance per path."""
        if self._client is None:
            if self.use_memory:
                # Use in-memory storage to avoid file locking issues
                self._client = QdrantClient(location=":memory:")
                logger.info("Created in-memory Qdrant client")
            else:
                self._client = QdrantClientManager.get_client(self.persist_directory)
        return self._client

    def _update_sources(self, documents: List[Document]):
        """Update list of sources from new documents."""
        for doc in documents:
            if "source" in doc.metadata:
                self.sources.add(doc.metadata["source"])

    def _get_embedding_size(self) -> int:
        """Get the actual embedding size by creating a test embedding."""
        try:
            test_embedding = self.embeddings.embed_query("test")
            return len(test_embedding)
        except Exception as e:
            logger.warning(f"Could not determine embedding size, using default 768: {e}")
            return 768

    def _ensure_collection_exists(self, client: QdrantClient):
        """Ensure the collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            if not collection_exists:
                # Get actual embedding size
                embedding_size = self._get_embedding_size()
                logger.info(f"Creating Qdrant collection '{self.collection_name}' with embedding size {embedding_size}")
                
                if self.enable_hybrid_search:
                    # Create collection with both dense and sparse vectors for hybrid search
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
                    # Create collection with dense vectors only
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
        """Check if the vectorstore already exists."""
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
        """Load sources from existing collection."""
        try:
            if self.vectorstore is None:
                return
            
            # Get all points to extract sources
            client = self._get_client()
            scroll_result = client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your needs
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
        """Create a vectorstore with provided documents or load existing one if reload_vectordb is True."""
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        client = self._get_client()
        vectorstore_exists = self.check_vectorstore_exists()

        # Load existing vectorstore if reload_vectordb is True and vectorstore exists
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
                
                # Load existing sources
                self._load_existing_sources()
                
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K})
                logger.info(f"Loaded existing vectorstore with {len(self.sources)} sources")
                return self.vectorstore, self.retriever
                
            except Exception as e:
                logger.error(f"Error loading existing vectorstore: {e}")
                logger.info("Creating new vectorstore instead...")
                vectorstore_exists = False

        # Create new vectorstore if either reload_vectordb is False or vectorstore doesn't exist
        if reload_vectordb and not vectorstore_exists:
            logger.warning(
                "Reload_vectordb flag is True but no existing vectorstore found. Creating a new one..."
            )

        if docs_list is None:
            # Create a placeholder document if no documents are provided
            logger.info(
                "No documents provided, creating an empty vectorstore with placeholder..."
            )
            placeholder_doc = Document(
                page_content="Placeholder content", metadata={"source": "placeholder"}
            )
            text_splitter = TextSplitter(chunk_type=self.chunk_type)
            doc_splits = text_splitter(documents=[placeholder_doc])
        else:
            # Process the provided documents
            logger.info(
                f"Creating vectorstore from {len(docs_list) if isinstance(docs_list, list) else 1} document(s)..."
            )
            text_splitter = TextSplitter(chunk_type=self.chunk_type)
            docs_to_process = docs_list if isinstance(docs_list, list) else [docs_list]
            doc_splits = text_splitter(documents=docs_to_process)

        # Ensure collection exists before creating vectorstore
        self._ensure_collection_exists(client)

        # Create the vectorstore - use client directly instead of from_documents
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
        
        # Add documents to the vectorstore
        if doc_splits:
            self.vectorstore.add_documents(documents=doc_splits)
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K})
        logger.info(f"Created new Qdrant vectorstore with {len(doc_splits)} documents")
        return self.vectorstore, self.retriever

    def add_documents(self, documents: List[Document]):
        """Add pre-split documents to the existing vectorstore, avoiding duplicates.

        Args:
            documents: List of already split/processed Document objects
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        
        try:
            client = self._get_client()
            new_docs = []
            new_ids = []
            
            for doc in documents:
                doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
                
                # Check if document already exists
                try:
                    existing_point = client.retrieve(
                        collection_name=self.collection_name,
                        ids=[doc_id]
                    )
                    if not existing_point:
                        new_docs.append(doc)
                        new_ids.append(doc_id)
                except Exception:
                    # If point doesn't exist, add it
                    new_docs.append(doc)
                    new_ids.append(doc_id)
            
            if new_docs:
                self.vectorstore.add_documents(documents=new_docs, ids=new_ids)
                self._update_sources(new_docs)
                logger.info(f"Successfully added {len(new_docs)} new documents to vectorstore")
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
        """
        Retrieve documents with optional filtering by sources and types
        
        Args:
            query (str): Search query
            filter_sources (Optional[List[str]]): Filter by specific sources
            filter_types (Optional[List[str]]): Filter by specific types
            
        Returns:
            List of retrieved documents
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")
        
        # Build filter conditions
        filter_conditions = []
        
        if filter_sources:
            # Create OR condition for sources
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
            # Create OR condition for types  
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
        
        # Combine all conditions with AND
        if filter_conditions:
            if len(filter_conditions) == 1:
                final_filter = filter_conditions[0]
            else:
                final_filter = models.Filter(
                    must=filter_conditions
                )
            
            try:
                # If num_docs is provided, use it, otherwise use K
                if num_docs:
                    k = num_docs
                else:
                    k = K
                    
                # Use vectorstore search with filter
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=final_filter
                )
                logger.info(f"Retrieved {len(results)} documents with filters - sources: {filter_sources}, types: {filter_types}")
                return results
            except Exception as e:
                logger.warning(f"Error with filtered search, falling back to unfiltered: {e}")
                # Fallback to unfiltered search
                return self.retriever.invoke(query)
        else:
            # No filters, use normal retrieval
            return self.retriever.invoke(query)

    def get_unique_sources(self) -> List[str]:
        """return self.sources"""
        return sorted(list(self.sources)) if self.sources else ["No sources available"]

    def clear_vectorstore(self):
        """Remove all stored documents and keep only a placeholder."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")

        try:
            client = self._get_client()
            
            # Delete the entire collection and recreate it
            try:
                client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}'")
            except Exception as e:
                logger.warning(f"Error deleting collection: {e}")
            
            # Recreate the collection
            self._ensure_collection_exists(client)

            # Clear sources
            self.sources.clear()

            # Create new vectorstore instance
            placeholder_doc = Document(
                page_content="Placeholder content", metadata={"source": "placeholder"}
            )
            text_splitter = TextSplitter(chunk_type=self.chunk_type)
            doc_splits = text_splitter(documents=[placeholder_doc])

            # Recreate vectorstore with placeholder
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
            
            # Add placeholder documents
            if doc_splits:
                self.vectorstore.add_documents(documents=doc_splits)
            
            self._update_sources(doc_splits)
            logger.info("Added placeholder document to vectorstore")

            # Update retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K})

        except Exception as e:
            logger.error(f"Error clearing vectorstore: {e}")
            raise

    def __del__(self):
        """Cleanup method to properly close client connection."""
        try:
            if hasattr(self, '_client') and self._client is not None:
                # Note: We don't close the client here since it might be shared
                # The QdrantClientManager handles client lifecycle
                pass
        except Exception as e:
            logger.warning(f"Error in vectorstore cleanup: {e}")

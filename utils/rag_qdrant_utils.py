import os
import time
from typing import List, Optional, Callable
import torch

from langchain.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts import PromptTemplate
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

class QdrantRAG:
    def __init__(self, client: QdrantClient):
        self.client = client

    def _get_text_splitter(self, chunk_type, chunk_size, chunk_overlap):
        if chunk_type == 'recursive':
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
                separators=["\n\n", "\n", ".", " ", ""],
            )
        elif chunk_type == 'character':
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
                separator="\n\n",
            )
        elif chunk_type == 'semantic':
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            return SemanticChunker(
                embedding_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
            )
        else:
            raise ValueError("Invalid chunk_type")

    def add_documents(self, collection_name, documents, tables, chunk_type, chunk_size, chunk_overlap, embedding_model_name):
        text_splitter = self._get_text_splitter(chunk_type, chunk_size, chunk_overlap)
        docs_processed = []
        for doc in documents:
            docs_processed += text_splitter.split_documents([doc])
        docs_processed += tables

        docs_contents = [doc.page_content for doc in docs_processed if hasattr(doc, 'page_content')]
        docs_metadatas = [doc.metadata for doc in docs_processed if hasattr(doc, 'metadata')]

        self.client.set_model(embedding_model_name=embedding_model_name)
        self.client.add(collection_name=collection_name, metadata=docs_metadatas, documents=docs_contents)

    def add_documents_with_gemini(self, collection_name, documents, tables, chunk_size, chunk_overlap, embedding_model_name):
        embedding_model = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        docs_processed = []
        for doc in documents:
            docs_processed += text_splitter.split_documents([doc])
        docs_processed += tables

        docs_contents = [doc.page_content for doc in docs_processed if hasattr(doc, 'page_content')]
        docs_metadatas = [doc.metadata for doc in docs_processed if hasattr(doc, 'metadata')]
        
        test_embedding = embedding_model.embed_query("test")
        actual_vector_size = len(test_embedding)
        
        if collection_name not in [col.name for col in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=actual_vector_size, distance=Distance.COSINE)
            )
        else:
            pass

        vectorstore = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=embedding_model,
        )

        vectorstore.add_texts(texts=docs_contents, metadatas=docs_metadatas)

    def add_documents_hybrid(
        self,
        collection_name,
        documents,
        tables,
        chunk_type,
        chunk_size,
        chunk_overlap,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        sparse_model_name="Qdrant/bm25",
    ):
        # Split documents
        text_splitter = self._get_text_splitter(chunk_type, chunk_size, chunk_overlap)
        docs_processed = []
        for doc in documents:
            docs_processed += text_splitter.split_documents([doc])
        docs_processed += tables

        docs_contents = [doc.page_content for doc in docs_processed if hasattr(doc, 'page_content')]
        docs_metadatas = [doc.metadata for doc in docs_processed if hasattr(doc, 'metadata')]

        # Create collection with BOTH dense & sparse vector configs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dense_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})
        sparse_embeddings = FastEmbedSparse(model_name=sparse_model_name)
        vector_size = len(dense_embeddings.embed_query("test"))

        if collection_name not in [col.name for col in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=Distance.COSINE),  # change size if needed
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=True))

                },
            )


        vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        # Add texts to vector store (it will handle both vectors)
        vectorstore.add_texts(texts=docs_contents, metadatas=docs_metadatas)


    def get_documents(self, collection_name, query, embedding_model, num_documents=5):
        self.client.set_model(embedding_model_name=embedding_model)
        search_results = self.client.query(
            collection_name=collection_name,
            query_text=query,
            limit=num_documents,
        )
        return [r.metadata['document'] for r in search_results]

    def get_documents_gemini(self, collection_name, query, embedding_model="models/embedding-001", num_documents=5):
        embedding_model = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        vectorstore = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=embedding_model,
        )

        search_results = vectorstore.similarity_search(query, k=num_documents)
        return [r.page_content for r in search_results]

    def get_documents_hybrid(self, collection_name, query, embedding_model_name, num_documents=5, reranker: Optional[Callable] = None):
        dense_embeddings = FastEmbedEmbeddings(model_name=embedding_model_name)
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse"
        )

        raw_results = vectorstore.similarity_search(query)
        documents = [doc.page_content for doc in raw_results]


        if reranker:
            documents = reranker(query, documents, top_k=num_documents)

        return documents[:num_documents]

    def generate_model_output(self, user_query: str, documents: List[str]) -> str:
        prompt = PromptTemplate.from_template(
            f"""
            You are an assistant that can answer questions based on the content of documents and filter information to give the best answer.
            Here are some document fragments retrieved from a PDF document:
            {documents}
            Based on the content of the documents, please answer the following question in no more than 1-2 sentences with the most relevant and concise answer:
            {user_query}
            """
        )
        return ""
    
    def doc_retrieval_function(self, collection_name, query, embedding_model, chunk_type=1, chunk_size=500, chunk_overlap=50, num_documents=5, reranker=None):
        documents = self.get_documents_hybrid(
            collection_name=collection_name,
            query=query,
            embedding_model_name=embedding_model,
            num_documents=num_documents,
            reranker=reranker,
        )
        return documents


    def full_pipeline(self, collection_name, query, embedding_model, chunk_type=1, chunk_size=500, chunk_overlap=50, num_documents=5, reranker=None):
        documents = self.get_documents_hybrid(
            collection_name=collection_name,
            query=query,
            embedding_model_name=embedding_model,
            num_documents=num_documents,
            reranker=reranker,
        )
        return self.generate_model_output(user_query=query, documents=documents)

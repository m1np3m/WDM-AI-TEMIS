import asyncio
import atexit
import logging
import os
import time
from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langfuse import Langfuse
from loguru import logger
from pydantic import BaseModel, Field

from .file_loader import PDFLoader
from .prompts import GENERATE_PROMPT, QUERY_ANALYSIS_PROMPT
from .reranker import Reranker
from .setting import REANKER_MODEL_NAME, K
from .vectorstore import QdrantClientManager, VectorStore


def cleanup_qdrant_clients():
    """Cleanup function to close all Qdrant clients on app shutdown."""
    try:
        QdrantClientManager.close_all_clients()
        logger.info("All Qdrant clients closed successfully")
    except Exception as e:
        logger.warning(f"Error during Qdrant cleanup: {e}")


# Register cleanup function
atexit.register(cleanup_qdrant_clients)


class RAG:
    def __init__(
        self,
        embedding_type: str,
        embedding_model: str,
        enable_hybrid_search: bool,
        chunk_type: str,
        use_memory: bool,
        collection_name: str,
        persist_dir: str,
        use_reranker: bool,
        langfuse_client: Optional[Langfuse] = None,
    ):
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.enable_hybrid_search = enable_hybrid_search
        self.chunk_type = chunk_type
        self.use_memory = use_memory
        self.collection_name = collection_name

        self.vectorstore_key = f"vs_{embedding_type}_{embedding_model}_{enable_hybrid_search}_{chunk_type}_{use_memory}_{collection_name}"
        self.vectorstore = VectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_type=embedding_type,
            embedding_model=embedding_model,
            enable_hybrid_search=enable_hybrid_search,
            chunk_type=chunk_type,
            use_memory=use_memory,
        )
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = Reranker(method=REANKER_MODEL_NAME)
            self.reranker_name = REANKER_MODEL_NAME
        else:
            self.reranker = None
        self.langfuse = langfuse_client

    def _format_time(self, seconds):
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

    async def _process_pdf_async(self, pdf_file_path, loader, debug_mode=False):
        """Asynchronous wrapper for PDF processing"""
        start_time = time.time()
        try:
            logger.info(f"Starting processing {pdf_file_path}")

            # Chạy phần load trong executor để không block event loop
            loop = asyncio.get_event_loop()
            splits = await loop.run_in_executor(
                None,
                lambda: loader.load(
                    path_string=pdf_file_path,
                    original_filename=os.path.basename(pdf_file_path),
                ),
            )

            processing_time = time.time() - start_time

            logger.info(
                f"Completed {pdf_file_path} in {processing_time:.1f}s - {len(splits)} documents"
            )

            return {
                "file_name": os.path.basename(pdf_file_path),
                "success": True,
                "splits": splits,
                "count": len(splits),
                "processing_time": processing_time,
                "file_size_mb": os.path.getsize(pdf_file_path) / (1024 * 1024),
            }
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Error processing {pdf_file_path} after {processing_time:.1f}s: {str(e)}"
            )
            return {
                "file_name": os.path.basename(pdf_file_path),
                "success": False,
                "error": str(e),
                "splits": [],
                "processing_time": processing_time,
                "file_size_mb": os.path.getsize(pdf_file_path) / (1024 * 1024),
            }

    async def load_pdfs(
        self,
        pdf_files: List[str],
        credential_path: Optional[str] = None,
        temp_dir: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """Process multiple PDF files asynchronously"""
        loader = PDFLoader(
            credential_path=credential_path or "",
            debug=debug_mode,
            temp_dir=temp_dir or "",
            enrich=False,
        )

        all_splits = []

        start_time = time.time()

        # Tạo các task bất đồng bộ cho mỗi file PDF
        tasks = [
            self._process_pdf_async(pdf_file_path, loader, debug_mode)
            for pdf_file_path in pdf_files
        ]

        # Chạy tất cả tasks bất đồng bộ
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Xử lý kết quả
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                # Nếu có exception, tạo result object với thông tin lỗi
                processed_results.append(
                    {
                        "file_name": "unknown",
                        "success": False,
                        "error": str(result),
                        "splits": [],
                        "processing_time": 0,
                        "file_size_mb": 0,
                    }
                )
            else:
                processed_results.append(result)
                if isinstance(result, dict) and result.get("success", False):
                    all_splits.extend(result.get("splits", []))

        total_time = time.time() - start_time
        successful_files = len([r for r in processed_results if r.get("success", False)])
        total_docs = sum(r.get("count", 0) for r in processed_results if r.get("success", False))
        text_docs = len(
            [doc for doc in all_splits if doc.metadata.get("type") == "text"]
        )
        table_docs = len(
            [doc for doc in all_splits if doc.metadata.get("type") == "table"]
        )

        logger.info(
            f"Processing completed: {successful_files}/{len(pdf_files)} files, {total_docs} documents, {self._format_time(total_time)}"
        )

        return (
            all_splits,
            processed_results,
            {
                "successful_files": successful_files,
                "total_files": len(pdf_files),
                "total_docs": total_docs,
                "text_docs": text_docs,
                "table_docs": table_docs,
                "total_time": total_time,
            },
        )

    def add_documents(
        self,
        documents: List[Document],
    ):
        if self.langfuse:
            with self.langfuse.start_as_current_span(
                name="document-ingestion",
                metadata={
                    "collection_name": self.collection_name,
                    "document_count": len(documents),
                },
                input={"document_sources": list(set(d.metadata.get("source", "Unknown") for d in documents))}
            ) as trace:
                with trace.start_as_current_span(
                    name="add-to-vectorstore",
                    input={"document_count": len(documents)}
                ) as span:
                    self.vectorstore.add_documents(documents)
                    span.update(output={"status": "success"})
        else:
            self.vectorstore.add_documents(documents)

    def retrieve_documents(
        self,
        query: str,
        filter_sources: Optional[List[str]] = None,
        filter_types: Optional[List[str]] = None,        
    ):
        
        if not self.use_reranker:
            return self.vectorstore.retrieve_documents(
                query=query, filter_sources=filter_sources, filter_types=filter_types
            )
        else:
            num_docs = K * 3
            docs = self.vectorstore.retrieve_documents(
                query=query, filter_sources=filter_sources, filter_types=filter_types, num_docs=num_docs
            )
            
            # Safer approach: Use index-based mapping instead of content mapping
            contents = [doc.page_content for doc in docs]
            reranked_contents = self.reranker.rerank(query, contents, K)
            
            # Map back using indices to handle duplicates properly
            reranked_docs = []
            for reranked_content in reranked_contents:
                # Find first occurrence of this content
                for i, original_content in enumerate(contents):
                    if original_content == reranked_content and i < len(docs):
                        reranked_docs.append(docs[i])
                        contents[i] = None  # Mark as used to avoid duplicates
                        break
            
            # Add this for debugging
            if len(reranked_docs) != len(reranked_contents):
                logger.warning(f"Some documents lost in reranking: expected {len(reranked_contents)}, got {len(reranked_docs)}")
            
            return reranked_docs

    def clear_vectorstore(self):
        self.vectorstore.clear_vectorstore()

    def get_unique_sources(self):
        return self.vectorstore.get_unique_sources()

    def get_vectorstore(self):
        return self.vectorstore

    def query_analysis(
        self,
        query: str,
        available_sources: Optional[List[str]] = None,
        available_types: Optional[List[str]] = ["text", "table"],
        callbacks: Optional[list] = None
    ):
        """
        Analyze query and return relevant sources and types

        Args:
            query (str): User input query
            available_sources (Optional[List[str]]): List of available sources to filter against
            available_types (Optional[List[str]]): List of available types to filter against

        Returns:
            QueryAnalysis: Object containing analysis information

        Raises:
            ValueError: When unable to parse results
            Exception: Other errors from LLM
        """

        class QueryAnalysis(BaseModel):
            """Schema for query analysis results"""

            sources: List[str] = Field(
                default=[],
                description="List of sources relevant to the query"
            )
            types: List[str] = Field(
                default=[],
                description="List of types relevant to the query"
            )
            confidence_score: Optional[float] = Field(
                default=None,
                description="Confidence level of the analysis (0-1)",
                ge=0.0,
                le=1.0,
            )
            reasoning: Optional[str] = Field(
                default=None,
                description="Reasoning for why these sources/types were selected"
            )

        parser = PydanticOutputParser(pydantic_object=QueryAnalysis)

        try:
            llm = ChatVertexAI(
                model_name="gemini-2.0-flash",
                temperature=0.1,  # Lower temperature for more stable results
                max_tokens=1024,  # Limit tokens to avoid overly long responses
            )

            # Create context about available sources/types if provided
            context_info = ""
            if available_sources:
                context_info += f"\nAvailable sources: {', '.join(available_sources)}"
            if available_types:
                context_info += f"\nAvailable types: {', '.join(available_types)}"

            template = QUERY_ANALYSIS_PROMPT

            prompt_template = PromptTemplate(
                template=template,
                input_variables=["query", "context_info"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            # Create chain
            chain = prompt_template | llm | parser

            # Invoke with error handling AND callbacks
            response = chain.invoke({"query": query, "context_info": ""}, config={"callbacks": callbacks})

            # Validate and filter results with improved source matching
            if available_sources:
                filtered_sources = []
                for suggested_source in response.sources:
                    # Check for exact match first
                    if suggested_source in available_sources:
                        filtered_sources.append(suggested_source)
                    else:
                        # Check for partial match (filename without extension)
                        for available_source in available_sources:
                            # Remove extension from available source for comparison
                            available_name = (
                                available_source.rsplit(".", 1)[0]
                                if "." in available_source
                                else available_source
                            )
                            suggested_name = (
                                suggested_source.rsplit(".", 1)[0]
                                if "." in suggested_source
                                else suggested_source
                            )

                            # Check if suggested name matches available name (case insensitive)
                            if suggested_name.lower() == available_name.lower():
                                filtered_sources.append(available_source)
                                break
                            # Also check if suggested name is contained in available name
                            elif (
                                suggested_name.lower() in available_name.lower()
                                or available_name.lower() in suggested_name.lower()
                            ):
                                filtered_sources.append(available_source)
                                break

                response.sources = list(set(filtered_sources))  # Remove duplicates

            if available_types:
                # Filter only available types
                response.types = [t for t in response.types if t in available_types]

            return response

        except Exception as e:
            logger.error(f"Error in query_analysis: {str(e)}")
            # Return default result when error occurs
            return QueryAnalysis(
                sources=[],
                types=[],
                confidence_score=0.0,
                reasoning=f"Error occurred during analysis: {str(e)}",
            )

    def prepare_context(self, docs: List[Document]) -> str:
        context_parts = ["<documents>"]

        for i, doc in enumerate(docs, 1):
            doc_str = f'\n<document index="{i}">'

            doc_str += "\n  <metadata>"
            for key, value in doc.metadata.items():
                doc_str += f"\n    <{key}>{value}</{key}>"
            doc_str += "\n  </metadata>"

            doc_str += f"\n  <content>\n{doc.page_content}\n  </content>"

            doc_str += "\n</document>"

            context_parts.append(doc_str)

        context_parts.append("\n</documents>")

        return "".join(context_parts)

    def generate_response(self, prompt: str, context: str, callbacks: Optional[list] = None) -> str:
        prompt_template = PromptTemplate(
            template=GENERATE_PROMPT,
            input_variables=["context", "question"],
        )

        llm = ChatVertexAI(
            model_name="gemini-2.0-flash",
            temperature=0.3,
        )

        chain = prompt_template | llm
        response = chain.invoke(
            {"context": context, "question": prompt},
            config={"callbacks": callbacks}
        )
        # Extract content from AIMessage if needed
        if hasattr(response, "content"):
            content = response.content
            # Handle case where content might be a list
            if isinstance(content, list):
                return str(content)
            return str(content) if content is not None else ""
        return str(response)

    def __call__(self, query: str, filter: bool = True) -> dict:
        
        if not self.langfuse:
            # Fallback to original behavior if Langfuse is not configured
            analysis = self.query_analysis(query, available_sources=self.get_unique_sources())
            filter_sources = analysis.sources if filter and analysis.sources else None
            filter_types = analysis.types if filter and analysis.types else None
            docs = self.retrieve_documents(query, filter_sources, filter_types)
            context = self.prepare_context(docs)
            response = self.generate_response(query, context)
            return {"response": response, "context": context, "docs": docs, "query": query, "analysis": analysis}

        
        with self.langfuse.start_as_current_span(
            name="rag-query",
            input={"query": query},
            metadata={
                "embedding_type": self.embedding_type,
                "embedding_model": self.embedding_model,
                "hybrid_search": self.enable_hybrid_search,
                "chunk_type": self.chunk_type,
            }
        ) as trace:
            # 1. Query Analysis Step
            with trace.start_as_current_span(name="query-analysis") as analysis_span:
                # Import CallbackHandler inside the method to get the handler
                from langfuse.langchain import CallbackHandler
                langfuse_handler = CallbackHandler()
                
                analysis = self.query_analysis(
                    query,
                    available_sources=self.get_unique_sources(),
                    available_types=["text", "table"],
                    callbacks=[langfuse_handler]
                )
                analysis_span.update(
                    output=analysis.dict()
                )
                
            filter_sources = analysis.sources if filter and analysis.sources else None
            filter_types = analysis.types if filter and analysis.types else None
            
            # 2. Retrieval Step
            with trace.start_as_current_span(name="retrieval") as retrieval_span:
                docs = self.retrieve_documents(
                    query=query, filter_sources=filter_sources, filter_types=filter_types
                )
                retrieval_span.update(
                    input={"query": query, "filter_sources": filter_sources, "filter_types": filter_types},
                    output={
                        "retrieved_documents_count": len(docs),
                        "retrieved_sources": list(set(d.metadata.get("source", "Unknown") for d in docs))
                    }
                )

            context = self.prepare_context(docs)
            
            # 3. Generation Step
            with trace.start_as_current_span(name="generation") as generation_span:
                # Import CallbackHandler again for this step
                from langfuse.langchain import CallbackHandler
                langfuse_handler = CallbackHandler()
                
                response = self.generate_response(
                    query,
                    context,
                    callbacks=[langfuse_handler]
                )

                generation_span.update(
                    input={"query": query, "context_length": len(context)},
                    output={"response": response}
                )

            # Finalize the main trace
            trace.update(output={"final_response": response})

            return {
                "response": response,
                "context": context,
                "docs": docs,
                "query": query,
                "analysis": analysis,
            }
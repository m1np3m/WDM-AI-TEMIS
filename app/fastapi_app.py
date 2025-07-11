import asyncio
import json
import os
import time
from typing import List, Optional, Dict, Any, Union
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag import RAG
from src.setting import K, CHUNK_SIZE, CHUNK_OVERLAP, ENRICH_TABLES
from src.WDMParser.WDMParser import WDMPDFParser

# Default configuration
EMBEDDING_TYPE = "vertexai"
EMBEDDING_MODEL = "text-embedding-004"
ENABLE_HYBRID_SEARCH = False
CHUNK_TYPE = "recursive"
USE_MEMORY = False
PERSIST_DIR = "qdrant_db"
COLLECTION_NAME = "wdm-ai-temis"


# Pydantic models for API
class ProcessPDFRequest(BaseModel):
    collection_name: Optional[str] = Field(default=None, description="Collection name to use")
    credential_path: Optional[str] = Field(default=None, description="Path to Google Cloud credentials")
    debug_mode: bool = Field(default=False, description="Enable debug logging")
    merge_span_tables: bool = Field(default=True, description="Merge tables that span across pages")
    enrich_tables: bool = Field(default=False, description="Enrich tables using AI")
    extract_text: bool = Field(default=True, description="Extract text content")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Question to ask")
    collection_name: Optional[str] = Field(default=None, description="Collection name to query")
    filter_sources: Optional[List[str]] = Field(default=None, description="Filter by specific sources")
    filter_types: Optional[List[str]] = Field(default=None, description="Filter by content types (text, table)")
    enable_filter: bool = Field(default=True, description="Enable smart filtering")


class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    progress: int  # 0-100
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RAGResponse(BaseModel):
    response: str
    context: str
    query: str
    sources: List[str]
    document_count: int
    processing_time: float


# Global variables
app = FastAPI(
    title="WDM AI-TEMIS FastAPI",
    description="Advanced PDF processing and RAG system with table extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
rag_instances: Dict[str, RAG] = {}
processing_tasks: Dict[str, ProcessingStatus] = {}
task_counter = 0


def get_rag_instance(collection_name: str = None) -> RAG:
    """Get or create RAG instance for a collection."""
    if collection_name is None:
        collection_name = COLLECTION_NAME
    
    if collection_name not in rag_instances:
        logger.info(f"Creating new RAG instance for collection: {collection_name}")
        rag_instances[collection_name] = RAG(
            embedding_type=EMBEDDING_TYPE,
            embedding_model=EMBEDDING_MODEL,
            enable_hybrid_search=ENABLE_HYBRID_SEARCH,
            chunk_type=CHUNK_TYPE,
            use_memory=USE_MEMORY,
            collection_name=collection_name,
            persist_dir=PERSIST_DIR,
            use_reranker=True,
            langfuse_client=None,  # Configure if needed
        )
    
    return rag_instances[collection_name]


def check_collection_exists_and_has_data(collection_name: str = None) -> Dict[str, Any]:
    """Check if collection exists and has data."""
    try:
        rag = get_rag_instance(collection_name or COLLECTION_NAME)
        vectorstore = rag.get_vectorstore()
        
        exists = vectorstore.check_vectorstore_exists()
        sources = rag.get_unique_sources()
        has_data = len(sources) > 1 or (len(sources) == 1 and sources[0] != "No sources available")
        
        return {
            "exists": exists,
            "has_data": has_data,
            "sources": sources,
            "source_count": len(sources) if has_data else 0
        }
    except Exception as e:
        logger.error(f"Error checking collection: {e}")
        return {
            "exists": False,
            "has_data": False,
            "sources": [],
            "source_count": 0,
            "error": str(e)
        }


async def process_pdfs_background(
    task_id: str,
    pdf_files_data: List[bytes],
    file_names: List[str],
    collection_name: str,
    credential_path: Optional[str],
    debug_mode: bool,
    merge_span_tables: bool,
    enrich_tables: bool,
    extract_text: bool
):
    """Background task for processing PDFs."""
    try:
        processing_tasks[task_id].status = "processing"
        processing_tasks[task_id].progress = 10
        processing_tasks[task_id].message = "Initializing PDF processing..."
        
        # Create WDMParser settings
        settings = WDMPDFParser.create_settings(
            credential_path=credential_path,
            debug=debug_mode,
            debug_level=1,
            max_concurrent_files=2,  # Limit for API
            max_memory_mb=2048,      # 2GB limit for API
            batch_size=3,
            cleanup_interval=2
        )
        
        parser = WDMPDFParser(settings=settings)
        
        processing_tasks[task_id].progress = 20
        processing_tasks[task_id].message = f"Processing {len(pdf_files_data)} PDF files..."
        
        # Process PDFs
        results, failed_files = await parser.process_documents(
            pdf_documents=pdf_files_data,
            merge_span_tables=merge_span_tables,
            enrich=enrich_tables,
            extract_text=extract_text,
            return_failed=True
        )
        
        processing_tasks[task_id].progress = 60
        processing_tasks[task_id].message = "Adding documents to vector store..."
        
        # Combine all documents
        all_documents = []
        for identifier, documents in results.items():
            # Update metadata with actual filenames
            idx = int(identifier.split('-')[-1]) if 'in-memory' in identifier else 0
            if idx < len(file_names):
                actual_filename = file_names[idx]
                for doc in documents:
                    doc.metadata["source"] = actual_filename
            all_documents.extend(documents)
        
        # Add to RAG system
        rag = get_rag_instance(collection_name)
        rag.add_documents(all_documents)
        
        processing_tasks[task_id].progress = 90
        processing_tasks[task_id].message = "Finalizing..."
        
        # Prepare results
        successful_files = len([r for r in results.values() if r])
        total_docs = len(all_documents)
        text_docs = len([doc for doc in all_documents if doc.metadata.get("type") == "text"])
        table_docs = len([doc for doc in all_documents if doc.metadata.get("type") == "table"])
        
        result_data = {
            "successful_files": successful_files,
            "total_files": len(pdf_files_data),
            "failed_files": len(failed_files),
            "total_documents": total_docs,
            "text_documents": text_docs,
            "table_documents": table_docs,
            "sources": rag.get_unique_sources(),
            "failed_file_names": [file_names[i] if i < len(file_names) else f"file_{i}" 
                                 for i, f in enumerate(failed_files)]
        }
        
        processing_tasks[task_id].status = "completed"
        processing_tasks[task_id].progress = 100
        processing_tasks[task_id].message = f"Successfully processed {successful_files}/{len(pdf_files_data)} files"
        processing_tasks[task_id].results = result_data
        
        logger.info(f"Task {task_id} completed: {result_data}")
        
    except Exception as e:
        error_msg = str(e)
        processing_tasks[task_id].status = "failed"
        processing_tasks[task_id].progress = 0
        processing_tasks[task_id].message = f"Processing failed: {error_msg}"
        processing_tasks[task_id].error = error_msg
        logger.error(f"Task {task_id} failed: {error_msg}\n{traceback.format_exc()}")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "WDM AI-TEMIS FastAPI Server",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload-pdfs/",
            "query": "/query/",
            "status": "/status/{task_id}",
            "collections": "/collections/",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_collections": list(rag_instances.keys()),
        "active_tasks": len([t for t in processing_tasks.values() if t.status == "processing"])
    }


@app.get("/collections/")
async def list_collections():
    """List all available collections and their status."""
    collections = {}
    
    for collection_name in rag_instances.keys():
        status = check_collection_exists_and_has_data(collection_name)
        collections[collection_name] = status
    
    # Also check default collection if not in instances
    if COLLECTION_NAME not in collections:
        status = check_collection_exists_and_has_data(COLLECTION_NAME)
        collections[COLLECTION_NAME] = status
    
    return {
        "collections": collections,
        "default_collection": COLLECTION_NAME
    }


@app.get("/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    """Get detailed information about a specific collection."""
    try:
        status = check_collection_exists_and_has_data(collection_name)
        
        if status["has_data"]:
            rag = get_rag_instance(collection_name)
            memory_info = {}
            
            # Try to get memory info if available
            try:
                if hasattr(rag.vectorstore, '_client') and hasattr(rag.vectorstore._client, 'get_memory_info'):
                    memory_info = rag.vectorstore._client.get_memory_info()
            except:
                pass
            
            return {
                "collection_name": collection_name,
                "status": status,
                "memory_info": memory_info,
                "settings": {
                    "embedding_type": EMBEDDING_TYPE,
                    "embedding_model": EMBEDDING_MODEL,
                    "hybrid_search": ENABLE_HYBRID_SEARCH,
                    "chunk_type": CHUNK_TYPE
                }
            }
        else:
            return {
                "collection_name": collection_name,
                "status": status,
                "message": "Collection is empty or doesn't exist"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")


@app.post("/upload-pdfs/")
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    collection_name: Optional[str] = Query(default=None),
    credential_path: Optional[str] = Query(default=None),
    debug_mode: bool = Query(default=False),
    merge_span_tables: bool = Query(default=True),
    enrich_tables: bool = Query(default=False),
    extract_text: bool = Query(default=True)
):
    """Upload and process multiple PDF files."""
    global task_counter
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Validate file types
    pdf_files = []
    file_names = []
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        try:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail=f"File {file.filename} is empty")
            
            pdf_files.append(content)
            file_names.append(file.filename)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file {file.filename}: {str(e)}")
    
    # Create task
    task_counter += 1
    task_id = f"task_{task_counter}_{int(time.time())}"
    
    processing_tasks[task_id] = ProcessingStatus(
        task_id=task_id,
        status="queued",
        progress=0,
        message=f"Queued for processing {len(pdf_files)} files"
    )
    
    # Use default collection if none specified
    if collection_name is None:
        collection_name = COLLECTION_NAME
    
    # Start background processing
    background_tasks.add_task(
        process_pdfs_background,
        task_id=task_id,
        pdf_files_data=pdf_files,
        file_names=file_names,
        collection_name=collection_name,
        credential_path=credential_path,
        debug_mode=debug_mode,
        merge_span_tables=merge_span_tables,
        enrich_tables=enrich_tables,
        extract_text=extract_text
    )
    
    return {
        "task_id": task_id,
        "message": f"Started processing {len(pdf_files)} PDF files",
        "files": file_names,
        "collection_name": collection_name,
        "status_url": f"/status/{task_id}"
    }


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a processing task."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return processing_tasks[task_id]


@app.post("/query/", response_model=RAGResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection using RAG."""
    try:
        start_time = time.time()
        
        # Use default collection if none specified
        collection_name = request.collection_name or COLLECTION_NAME
        
        # Check if collection has data
        status = check_collection_exists_and_has_data(collection_name)
        if not status["has_data"]:
            raise HTTPException(
                status_code=404, 
                detail=f"Collection '{collection_name}' is empty or doesn't exist. Please upload PDFs first."
            )
        
        # Get RAG instance
        rag = get_rag_instance(collection_name)
        
        # Perform query with filtering if enabled
        if request.enable_filter:
            # Use smart query analysis
            result = rag(request.query, filter=True)
        else:
            # Direct query without filtering
            docs = rag.retrieve_documents(
                query=request.query,
                filter_sources=request.filter_sources,
                filter_types=request.filter_types
            )
            context = rag.prepare_context(docs)
            response = rag.generate_response(request.query, context)
            result = {
                "response": response,
                "context": context,
                "docs": docs,
                "query": request.query
            }
        
        processing_time = time.time() - start_time
        
        # Extract sources from documents
        sources = list(set(doc.metadata.get("source", "Unknown") for doc in result["docs"]))
        
        return RAGResponse(
            response=result["response"],
            context=result["context"],
            query=result["query"],
            sources=sources,
            document_count=len(result["docs"]),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.delete("/collections/{collection_name}")
async def clear_collection(collection_name: str):
    """Clear all documents from a collection."""
    try:
        if collection_name in rag_instances:
            rag = rag_instances[collection_name]
            rag.clear_vectorstore()
            
            return {
                "message": f"Collection '{collection_name}' cleared successfully",
                "collection_name": collection_name
            }
        else:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")


@app.get("/tasks/")
async def list_tasks():
    """List all processing tasks."""
    return {
        "tasks": list(processing_tasks.values()),
        "active_count": len([t for t in processing_tasks.values() if t.status == "processing"]),
        "completed_count": len([t for t in processing_tasks.values() if t.status == "completed"]),
        "failed_count": len([t for t in processing_tasks.values() if t.status == "failed"])
    }


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task from memory."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = processing_tasks.pop(task_id)
    return {
        "message": f"Task {task_id} deleted",
        "task": task
    }


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/fastapi_app.log", rotation="500 MB")
    
    # Run the application
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

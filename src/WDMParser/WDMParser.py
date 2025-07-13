import asyncio
import base64
import gc
import json
import os
import psutil
from io import StringIO
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import pandas as pd
import pymupdf
from langchain_core.documents import Document
from loguru import logger
from markdown import markdown

from .extract_tables import WDMMergedTable, WDMTable, full_pipeline, get_tables_from_pdf
from ..setting import IGNORE_TABLES, ENRICH_TABLES

def convert_markdown_to_df(markdown_text: str) -> pd.DataFrame:
    try:
        html_table = markdown(markdown_text, extensions=["markdown.extensions.tables"])
        dfs = pd.read_html(StringIO(f"<table>{html_table}</table>"))
        if dfs:
            return dfs[0]
        else:
            print(f"No tables found in {markdown_text}")
            # Return empty DataFrame if no tables found
            return pd.DataFrame()
    except Exception as e:
        # Return empty DataFrame if conversion fails
        return pd.DataFrame()


def convert_table2text(wdm_table: WDMMergedTable) -> str:
    markdown_text = wdm_table['text']
    context_before = wdm_table['context_before']
    table_shape = (wdm_table['n_rows'], wdm_table['n_columns'])
    pages = wdm_table['page']
    source = wdm_table['source']


    df = convert_markdown_to_df(markdown_text.strip())
    
    # Handle empty DataFrame
    if df.empty:
        return ""
    df_processed = df.where(pd.notna(df), None)
    table_data = df_processed.to_dict(orient='records')
    
    df_string = f"Table {context_before} with shape {table_shape} on pages {pages} from {source}\n\n"
    df_string += json.dumps(table_data, indent=2)
    df_string += "\n\n"
    
    return df_string.strip()


class WDMText(TypedDict):
    """Text extracted from a page of a PDF file"""

    text: str
    page: int
    source: str


class WDMImage(TypedDict):
    """Image extracted from a page of a PDF file"""

    base64_image: str
    page: int
    source: str
    bbox: Tuple[float, float, float, float]
    image_path: str


class WDMParserSettings(TypedDict, total=False):
    """Settings for WDMParser configuration"""
    credential_path: Optional[str]
    debug: bool
    debug_level: int
    max_concurrent_files: int
    max_memory_mb: int
    batch_size: int
    cleanup_interval: int


class WDMPDFParser:
    def __init__(
        self,
        settings: Optional[Dict] = None,
        file_path: Optional[str] = None,
        credential_path: Optional[str] = None,
        debug: bool = False,
        debug_level: int = 1,
    ):
        """
        Initialize WDMPDFParser with settings or backward compatibility parameters.
        
        Args:
            settings: Dictionary containing parser configuration
            file_path: (Deprecated) Single file path for backward compatibility
            credential_path: (Deprecated) Credential path for backward compatibility
            debug: (Deprecated) Debug flag for backward compatibility
            debug_level: (Deprecated) Debug level for backward compatibility
        """
        # Handle backward compatibility
        if settings is None:
            settings = {}
            
        # If old-style parameters are provided, use them
        if file_path is not None or credential_path is not None:
            if self._is_debug_mode(settings.get('debug', debug)):
                logger.warning(
                    "Using deprecated constructor parameters. "
                    "Consider using settings dictionary instead."
                )
            self.file_path = file_path
            self.credential_path = credential_path or settings.get('credential_path')
            self.debug = debug or settings.get('debug', False)
            self.debug_level = debug_level or settings.get('debug_level', 1)
        else:
            # New settings-based approach
            self.file_path = None  # Will be set per operation
            self.credential_path = settings.get('credential_path')
            self.debug = settings.get('debug', False)
            self.debug_level = settings.get('debug_level', 1)
        
        # Memory management settings
        self.max_concurrent_files = settings.get('max_concurrent_files', 3)
        self.max_memory_mb = settings.get('max_memory_mb', 8192)  # 8GB default
        self.batch_size = settings.get('batch_size', 5)
        self.cleanup_interval = settings.get('cleanup_interval', 10)
        
        # Create semaphore for concurrent file processing
        self._semaphore = asyncio.Semaphore(self.max_concurrent_files)
        
        if self.debug:
            logger.info(f"WDMParser initialized with max_concurrent_files={self.max_concurrent_files}, "
                       f"max_memory_mb={self.max_memory_mb}, batch_size={self.batch_size}")

    def _is_debug_mode(self, debug_flag: bool) -> bool:
        """Helper method to determine debug mode"""
        return debug_flag or self.debug

    def _check_memory_usage(self) -> bool:
        """Check if current memory usage is within limits"""
        if self.max_memory_mb <= 0:
            return True
            
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            if memory_mb > self.max_memory_mb:
                if self.debug:
                    logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.max_memory_mb}MB)")
                return False
            return True
        except Exception as e:
            if self.debug:
                logger.warning(f"Could not check memory usage: {e}")
            return True

    def _cleanup_memory(self):
        """Force garbage collection and memory cleanup"""
        if self.debug:
            memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        gc.collect()
        
        if self.debug:
            memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
            logger.info(f"Memory cleanup: {memory_before:.1f}MB -> {memory_after:.1f}MB")

    async def process_pdf(
        self,
        pdf_data: Union[str, bytes],
        pages: Optional[List[int]] = None,
        merge_span_tables: bool = True,
        enrich: bool = ENRICH_TABLES,
        extract_text: bool = True,
    ) -> List[Document]:
        """
        Process a single PDF from file path or bytes.
        
        Args:
            pdf_data: File path (str) or PDF content (bytes)
            pages: List of page numbers to process
            merge_span_tables: Whether to merge spanning tables
            enrich: Whether to enrich tables
            extract_text: Whether to extract text content
            
        Returns:
            List of Document objects containing extracted content
        """
        async with self._semaphore:
            try:
                # Determine source name for metadata
                if isinstance(pdf_data, bytes):
                    source_name = "<in-memory>"
                    if self.debug:
                        logger.info(f"Processing PDF from bytes ({len(pdf_data)} bytes)")
                else:
                    source_name = pdf_data
                if self.debug:
                        logger.info(f"Processing PDF from file: {os.path.basename(pdf_data)}")
                
                # Check memory before processing
                if not self._check_memory_usage():
                    await asyncio.sleep(1)
                    self._cleanup_memory()
                    
                    if not self._check_memory_usage():
                        raise MemoryError(f"Memory limit exceeded before processing {source_name}")
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                # Extract tables
                def extract_tables_task():
                    return self._extract_tables_sync(pdf_data, pages, merge_span_tables, enrich)
                
                tables_task = loop.run_in_executor(None, extract_tables_task)
                
                # Extract text if requested
                text_task = None
                if extract_text:
                    def extract_text_task():
                        return self._extract_text_sync(pdf_data, pages)
                    
                    text_task = loop.run_in_executor(None, extract_text_task)
                
                # Wait for completion
                table_docs = await tables_task
                text_docs = await text_task if text_task else []
                
                # Combine results
                all_docs = table_docs + text_docs
                
                if self.debug:
                    logger.info(f"Completed processing {source_name}: "
                               f"{len(table_docs)} tables, {len(text_docs)} text blocks")
                
                return all_docs
                
            except Exception as e:
                source_name = "<in-memory>" if isinstance(pdf_data, bytes) else pdf_data
                if self.debug:
                    logger.error(f"Error processing {source_name}: {str(e)}")
                return []

    def _extract_tables_sync(
        self,
        pdf_data: Union[str, bytes],
        pages: Optional[List[int]],
        merge_span_tables: bool,
        enrich: bool,
    ) -> List[Document]:
        """Synchronous table extraction supporting both file paths and bytes"""
        all_tables: List[WDMMergedTable] = []
        
        if merge_span_tables or enrich:
            if not self.credential_path:
                error_msg = (
                    f"❌ Credentials required for advanced features (merge_span_tables={merge_span_tables}, enrich={enrich})\n"
                    f"Please provide credential_path parameter when initializing WDMPDFParser"
                )
                raise ValueError(error_msg)

            if not os.path.exists(self.credential_path):
                raise FileNotFoundError(
                    f"Credentials file not found: {self.credential_path}\n"
                    "Please ensure the path is correct and the file exists."
                )

            if self.debug:
                logger.info(f"✅ Using credentials: {self.credential_path}")

        if merge_span_tables:
            # Use full_pipeline for table merging
            merged_tables = full_pipeline(
                pdf_data,  # Now supports both str and bytes
                pages=pages,
                debug=self.debug,
                debug_level=self.debug_level,
                enrich=enrich,
                credential_path=self.credential_path,
            )
            all_tables = merged_tables
        else:
            # Use simple table extraction without merging
            individual_tables = get_tables_from_pdf(
                pdf_data,  # Now supports both str and bytes
                pages=pages,
                debug=self.debug,
                debug_level=self.debug_level,
                enrich=enrich,
                use_ai_analysis=False,
                credential_path=self.credential_path,
            )
            # Convert WDMTable to WDMMergedTable format for consistency
            all_tables = []
            for table in individual_tables:
                merged_table: WDMMergedTable = {
                    'text': table['text'],
                    'page': [table['page']] if isinstance(table['page'], int) else table['page'],
                    'source': table['source'],
                    'bbox': [table['bbox']] if isinstance(table['bbox'], tuple) else table['bbox'],
                    'headers': [table.get('headers', [])],
                    'n_rows': table['n_rows'],
                    'n_columns': table['n_columns'],
                    'context_before': table['context_before'],
                    'image_paths': [table['image_path']],
                }
                all_tables.append(merged_table)
        
        documents: List[Document] = [
            Document(
                page_content=convert_table2text(table),
                metadata={
                    "page": str(table["page"]) if isinstance(table["page"], int) else ",".join(map(str, table["page"])),
                    "source": table["source"],
                    "type": "table",
                },
            )
            for table in all_tables
        ]
        return documents

    def _extract_text_sync(
        self,
        pdf_data: Union[str, bytes],
        pages: Optional[List[int]],
    ) -> List[Document]:
        """Synchronous text extraction supporting both file paths and bytes"""
        # Open document based on data type
        if isinstance(pdf_data, bytes):
            doc = pymupdf.open(stream=pdf_data, filetype="pdf")
            source_name = "<in-memory>"
        else:
            doc = pymupdf.open(pdf_data)
            source_name = pdf_data
            
        if pages is None:
            pages = list(range(1, len(doc) + 1))
        
        all_text: List[WDMText] = []

        try:
            for page_number in pages:
                try:
                    # Convert 1-indexed page number to 0-indexed for PyMuPDF
                    page = doc[page_number - 1]
                    
                    # Capture the full text up front so we can always fall back to it if any
                    # of the redaction steps fail or remove everything.
                    full_text = page.get_text().strip()

                    # Default: we will keep whatever we captured before any redaction.
                    text_content = full_text
                    
                    if IGNORE_TABLES:
                        try:
                            # Attempt to locate tables and redact them.
                            tables = page.find_tables(strategy="lines_strict").tables
                            if tables:
                                for tab in tables:
                                    page.add_redact_annot(tab.bbox)
                                page.apply_redactions()

                                # After redaction, re-extract the text. If it is not empty, we
                                # prefer this redacted version; otherwise, we will keep the full
                                # version captured earlier.
                                redacted_text = page.get_text().strip()
                                if redacted_text:
                                    text_content = redacted_text
                        except Exception as e:
                            # If anything goes wrong during table redaction, fall back to the
                            # non-redacted text instead of skipping the page altogether.
                            if self.debug:
                                logger.warning(f"Redaction failed on page {page_number}: {e}. Using full text instead.")
                    # If after all attempts we still have no content, skip adding this page.
                    if not text_content:
                        if self.debug:
                            logger.warning(f"Source: {source_name}, Page {page_number} is empty")
                        continue
                        
                    all_text.append(
                        WDMText(
                            text=text_content, 
                            page=page_number, 
                            source=source_name
                        )
                    )
                    
                    if self.debug:
                        char_count = len(text_content.strip())
                        mode = "with table redaction" if IGNORE_TABLES else "without table redaction"
                        logger.info(f"Extracted text from page {page_number} ({mode}): {char_count} characters")
                        
                except Exception as e:
                    if self.debug:
                        logger.warning(f"Failed to extract text from page {page_number}: {e}")
                    # Skip this page and continue with next one
                    continue
                    
        finally:
            doc.close()
            
        documents: List[Document] = [
            Document(
                page_content=text["text"],
                metadata={
                    "page": str(text["page"]),
                    "source": text["source"],
                    "type": "text",
                },
            )
            for text in all_text
            if text["text"].strip()  # Only include pages with actual text content
        ]
        
        if self.debug:
            mode = "with table redaction" if IGNORE_TABLES else "without table redaction"
            logger.info(f"Successfully extracted {len(documents)} text documents ({mode})")
            
        return documents

    async def process_documents(
        self,
        pdf_documents: Union[List[str], List[bytes], List[Union[str, bytes]]],
        pages: Optional[List[int]] = None,
        merge_span_tables: bool = True,
        enrich: bool = ENRICH_TABLES,
        extract_text: bool = True,
        return_failed: bool = False,
    ) -> Union[Dict[str, List[Document]], Tuple[Dict[str, List[Document]], List[str]]]:
        """
        Process multiple PDF documents asynchronously with memory management.
        
        Args:
            pdf_documents: List of PDF file paths, bytes, or mixed
            pages: List of page numbers to process (applied to all documents)
            merge_span_tables: Whether to merge spanning tables
            enrich: Whether to enrich tables
            extract_text: Whether to extract text content
            return_failed: Whether to return list of failed files
            
        Returns:
            Dictionary mapping identifiers to extracted documents.
            If return_failed=True, returns tuple of (results, failed_files)
        """
        if not pdf_documents:
            return {} if not return_failed else ({}, [])
        
        if self.debug:
            logger.info(f"Starting async processing of {len(pdf_documents)} documents")
            logger.info(f"Memory management: max_concurrent={self.max_concurrent_files}, "
                       f"batch_size={self.batch_size}, max_memory={self.max_memory_mb}MB")
        
        results = {}
        failed_files = []
        processed_count = 0
        
        # Process documents in batches to manage memory
        for batch_start in range(0, len(pdf_documents), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(pdf_documents))
            batch_documents = pdf_documents[batch_start:batch_end]
            
            if self.debug:
                logger.info(f"Processing batch {batch_start//self.batch_size + 1}: "
                           f"documents {batch_start+1}-{batch_end} of {len(pdf_documents)}")
            
            # Create tasks for current batch
            tasks = []
            for i, pdf_data in enumerate(batch_documents):
                # Create identifier for results mapping
                if isinstance(pdf_data, bytes):
                    identifier = f"<in-memory-{batch_start + i}>"
                else:
                    identifier = pdf_data
                
                task = self._process_single_pdf_with_id(
                    identifier, pdf_data, pages, merge_span_tables, enrich, extract_text
                )
                tasks.append(task)
            
            # Execute batch with error handling
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for (identifier, pdf_data), result in zip(
                    [(f"<in-memory-{batch_start + i}>" if isinstance(doc, bytes) else doc, doc) 
                     for i, doc in enumerate(batch_documents)], 
                    batch_results
                ):
                    if isinstance(result, BaseException):
                        if self.debug:
                            logger.error(f"Failed to process {identifier}: {result}")
                        failed_files.append(identifier)
                    else:
                        result_id, documents = result
                        results[result_id] = documents
                        processed_count += 1
                
            except Exception as e:
                if self.debug:
                    logger.error(f"Batch processing error: {e}")
                # Add all batch documents to failed list
                for i, pdf_data in enumerate(batch_documents):
                    identifier = f"<in-memory-{batch_start + i}>" if isinstance(pdf_data, bytes) else pdf_data
                    failed_files.append(identifier)
            
            # Memory cleanup after each batch
            if (batch_start + self.batch_size) % (self.cleanup_interval * self.batch_size) == 0:
                if self.debug:
                    logger.info("Performing periodic memory cleanup...")
                self._cleanup_memory()
                
                # Brief pause to allow system recovery
                await asyncio.sleep(0.5)
        
        # Final cleanup
        self._cleanup_memory()
        
        if self.debug:
            logger.info(f"Async processing completed: {processed_count}/{len(pdf_documents)} successful, "
                       f"{len(failed_files)} failed")
        
        if return_failed:
            return results, failed_files
        return results

    async def _process_single_pdf_with_id(
        self,
        identifier: str,
        pdf_data: Union[str, bytes],
        pages: Optional[List[int]],
        merge_span_tables: bool,
        enrich: bool,
        extract_text: bool,
    ) -> Tuple[str, List[Document]]:
        """Helper method to process a single PDF and return with identifier"""
        try:
            documents = await self.process_pdf(
                pdf_data=pdf_data,
                pages=pages,
                merge_span_tables=merge_span_tables,
                enrich=enrich,
                extract_text=extract_text,
            )
            return identifier, documents
        except Exception as e:
            if self.debug:
                logger.error(f"Error processing {identifier}: {str(e)}")
            return identifier, []

    # Maintain backward compatibility methods
    def extract_tables(
        self,
        pages: Optional[List[int]] = None,
        merge_span_tables: bool = True,
        enrich: bool = ENRICH_TABLES,
    ) -> List[Document]:
        """
        Extract tables from PDF pages (backward compatibility method).
        
        Args:
            pages: List of page numbers to extract (1-indexed). If None, extracts all pages.
            merge_span_tables: Whether to merge tables that span across pages.
            enrich: Whether to enrich tables using AI.
        
        Returns:
            List of Document objects containing the extracted tables.
        """
        if self.file_path is None:
            raise ValueError(
                "No file_path specified. Either use the old constructor with file_path "
                "or use the new async methods with explicit file paths."
            )
            
        return self._extract_tables_sync(self.file_path, pages, merge_span_tables, enrich)

    def extract_text(self, pages: Optional[List[int]] = None) -> List[Document]:
        """
        Extract text from PDF pages (backward compatibility method).
        
        Args:
            pages: List of page numbers to extract (1-indexed). If None, extracts all pages.

        Returns:
            List of Document objects containing the extracted text.
        """
        if self.file_path is None:
            raise ValueError(
                "No file_path specified. Either use the old constructor with file_path "
                "or use the new async methods with explicit file paths."
            )
            
        return self._extract_text_sync(self.file_path, pages)

    # Utility methods for the new async interface
    @classmethod
    def create_settings(
        cls,
        credential_path: Optional[str] = None,
        debug: bool = False,
        debug_level: int = 1,
        max_concurrent_files: int = 3,
        max_memory_mb: int = 8192,
        batch_size: int = 5,
        cleanup_interval: int = 10,
    ) -> Dict:
        """
        Create a settings dictionary for WDMParser initialization.
        
        Args:
            credential_path: Path to Google Cloud service account JSON key file
            debug: Enable debug logging
            debug_level: Debug verbosity level (1-2)
            max_concurrent_files: Maximum number of files to process simultaneously
            max_memory_mb: Maximum memory usage in MB (0 = no limit)
            batch_size: Number of files to process in each batch
            cleanup_interval: Number of batches between memory cleanups
            
        Returns:
            Settings dictionary ready for WDMParser initialization
        """
        return {
            'credential_path': credential_path,
            'debug': debug,
            'debug_level': debug_level,
            'max_concurrent_files': max_concurrent_files,
            'max_memory_mb': max_memory_mb,
            'batch_size': batch_size,
            'cleanup_interval': cleanup_interval,
        }

    def get_memory_info(self) -> Dict[str, Union[float, str, bool]]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent(),
                'limit_mb': float(self.max_memory_mb),
                'within_limit': memory_info.rss / (1024 * 1024) <= self.max_memory_mb if self.max_memory_mb > 0 else True
            }
        except Exception as e:
            return {'error': str(e)}


# Convenience function for quick async processing
async def process_pdf_documents(
    pdf_files: Union[List[str], List[bytes], List[Union[str, bytes]]],
    credential_path: Optional[str] = None,
    **kwargs
) -> Dict[str, List[Document]]:
    """
    Convenience function for processing multiple PDF documents asynchronously.
    
    Args:
        pdf_files: List of PDF file paths, bytes, or mixed
        credential_path: Path to Google Cloud service account JSON key file
        **kwargs: Additional arguments passed to WDMParser settings
        
    Returns:
        Dictionary mapping identifiers to extracted documents
    """
    settings = WDMPDFParser.create_settings(credential_path=credential_path, **kwargs)
    parser = WDMPDFParser(settings=settings)
    result = await parser.process_documents(pdf_files)
    # Ensure we always return the dict, not the tuple
    if isinstance(result, tuple):
        return result[0]
    return result


# Maintain legacy interface
WDMParser = WDMPDFParser  # Alias for backward compatibility

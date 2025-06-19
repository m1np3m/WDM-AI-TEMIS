from loguru import logger
import os
import tempfile
from typing import List

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from .setting import CHUNK_OVERLAP, CHUNK_SIZE
from .WDMParser import WDMPDFParser


class TextSplitter:
    def __init__(
        self,
        chunk_type: str = "recursive",
        separators: List[str] = ["\n\n", "\n", ". ", "! ", "? ", ":", ";", " "],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
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


class PDFLoader:
    def __init__(self, text_splitter=None, credential_path: str = None, debug: bool = False, temp_dir: str = None, enrich: bool = False):
        """
        Initialize the PDF loader with an optional text splitter and credentials for WDMPDFParser.

        Args:
            text_splitter: An instance of TextSplitter to process text documents
            credential_path: Path to Google service account credentials (optional, required for advanced features)
            debug: Enable debug mode for detailed logging
            temp_dir: Custom temporary directory path (optional)
        """
        self.text_splitter = text_splitter or TextSplitter()
        if not credential_path:
            self.credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        else:
            self.credential_path = credential_path
        self.debug = debug
        self.temp_dir = temp_dir
        self.enrich = enrich
        # Create temp directory if specified and doesn't exist
        if self.temp_dir and not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
            if self.debug:
                logger.info(f"Created temporary directory: {self.temp_dir}")

    def _load_from_file_object(self, pdf_file, original_filename: str = None):
        """
        Helper method to load a PDF from a file object using WDMPDFParser.

        Args:
            pdf_file: A file object from Streamlit's file_uploader
            original_filename: Original filename for metadata

        Returns:
            List[Document]: Documents from the PDF (text and tables)
        """
        if self.debug:
            logger.info(f"Processing PDF file: {original_filename or 'unnamed'}")
            logger.info(f"File size: {len(pdf_file.getvalue())} bytes")
        
        # Create a temporary file to store the uploaded PDF
        temp_file_kwargs = {"delete": False, "suffix": ".pdf"}
        if self.temp_dir:
            temp_file_kwargs["dir"] = self.temp_dir
            
        with tempfile.NamedTemporaryFile(**temp_file_kwargs) as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            temp_path = tmp_file.name
            
        if self.debug:
            logger.info(f"Created temporary file: {temp_path}")

        try:
            # Initialize WDMPDFParser
            parser = WDMPDFParser(
                file_path=temp_path,
                credential_path=self.credential_path,
                debug=self.debug  # Pass debug flag to parser
            )
            
            if self.debug:
                logger.info("Extracting text documents...")
            
            # Extract text documents
            text_documents = parser.extract_text()
            
            if self.debug:
                logger.info(f"Extracted {len(text_documents)} text documents")
                logger.info("Extracting table documents...")
            
            # Extract table documents (only if credentials are available)
            table_documents = []
            try:
                if self.credential_path:
                    if self.debug:
                        logger.info("Using advanced mode (with credentials)")
                    table_documents = parser.extract_tables(
                        merge_span_tables=True,
                        enrich=self.enrich
                    )
                    logger.info(f"Extracted {len(table_documents)} tables from PDF (advanced mode)")
                else:
                    if self.debug:
                        logger.info("Using basic mode (without credentials)")
                    # Extract tables without advanced features if no credentials
                    table_documents = parser.extract_tables(
                        merge_span_tables=False,
                        enrich=self.enrich
                    )
                    logger.info(f"Extracted {len(table_documents)} tables from PDF (basic mode)")
            except Exception as e:
                logger.warning(f"Failed to extract tables from {original_filename or 'unnamed'}: {e}")
                if self.debug:
                    logger.exception("Table extraction error details:")
                table_documents = []
            
            # Combine all documents
            all_documents = text_documents + table_documents
            
            # Update metadata with original filename if provided
            if original_filename:
                for doc in all_documents:
                    doc.metadata["source"] = original_filename
            
            if self.debug:
                logger.info(f"Total documents extracted from {original_filename or 'unnamed'}: {len(all_documents)} (Text: {len(text_documents)}, Tables: {len(table_documents)})")
                
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to process PDF {original_filename or 'unnamed'}: {e}")
            if self.debug:
                logger.exception("PDF processing error details:")
            return []
            
        finally:
            # Ensure cleanup happens even if loading fails
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    if self.debug:
                        logger.info(f"Cleaned up temporary file: {temp_path}")
            except OSError as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")

    def _load_from_path(self, path_string):
        """
        Helper method to load a PDF from a file path using WDMPDFParser.

        Args:
            path_string: A string path to a PDF file

        Returns:
            List[Document]: Documents from the PDF (text and tables)
        """
        if self.debug:
            logger.info(f"Processing PDF from path: {path_string}")
            
        # Check if file exists
        if not os.path.exists(path_string):
            logger.error(f"PDF file not found: {path_string}")
            return []
            
        try:
            # Initialize WDMPDFParser
            parser = WDMPDFParser(
                file_path=path_string,
                credential_path=self.credential_path,
                debug=self.debug  # Pass debug flag to parser
            )
            
            if self.debug:
                logger.info("Extracting text documents...")
            
            # Extract text documents
            text_documents = parser.extract_text()
            
            if self.debug:
                logger.info(f"Extracted {len(text_documents)} text documents")
                logger.info("Extracting table documents...")
            
            # Extract table documents (only if credentials are available)
            table_documents = []
            try:
                if self.credential_path:
                    if self.debug:
                        logger.info("Using advanced mode (with credentials)")
                    table_documents = parser.extract_tables(
                        merge_span_tables=True,
                        enrich=self.enrich
                    )
                    logger.info(f"Extracted {len(table_documents)} tables from PDF (advanced mode)")
                else:
                    if self.debug:
                        logger.info("Using basic mode (without credentials)")
                    # Extract tables without advanced features if no credentials
                    table_documents = parser.extract_tables(
                        merge_span_tables=False,
                        enrich=False
                    )
                    logger.info(f"Extracted {len(table_documents)} tables from PDF (basic mode)")
            except Exception as e:
                logger.warning(f"Failed to extract tables from {path_string}: {e}")
                if self.debug:
                    logger.exception("Table extraction error details:")
                table_documents = []
            
            # Combine all documents
            all_documents = text_documents + table_documents
            
            if self.debug:
                logger.info(f"Total documents extracted from {path_string}: {len(all_documents)} (Text: {len(text_documents)}, Tables: {len(table_documents)})")
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to process PDF {path_string}: {e}")
            if self.debug:
                logger.exception("PDF processing error details:")
            return []

    def load(self, pdf_file=None, path_string=None, original_filename: str = None):
        """
        Load PDF content either from a file uploaded via Streamlit or from a path.

        Args:
            pdf_file: A file object from Streamlit's file_uploader
            path_string: A string path to a PDF file
            original_filename: Original filename for metadata

        Returns:
            List[Document]: A list of document chunks after splitting (only text is split, tables remain whole)

        Raises:
            ValueError: If neither pdf_file nor path_string is provided
        """
        if pdf_file is not None:
            documents = self._load_from_file_object(pdf_file, original_filename)
        elif path_string is not None:
            documents = self._load_from_path(path_string)
        else:
            raise ValueError("Either pdf_file or path_string must be provided")

        if not documents:
            return []

        # Separate text and table documents
        text_documents = [doc for doc in documents if doc.metadata.get("type") == "text"]
        table_documents = [doc for doc in documents if doc.metadata.get("type") == "table"]
        
        # Apply text splitter only to text documents
        split_text_documents = self.text_splitter(text_documents) if text_documents else []
        
        # Tables are not split - they remain as whole documents
        logger.info(f"Text documents split into {len(split_text_documents)} chunks")
        logger.info(f"Table documents kept whole: {len(table_documents)} tables")
        
        # Combine split text documents with whole table documents
        return split_text_documents + table_documents


# class WebLoader:
#     def __init__(self, text_splitter=None):
#         """
#         Initialize the Web loader with an optional text splitter.

#         Args:
#             text_splitter: An instance of TextSplitter to process documents
#         """
#         self.text_splitter = text_splitter or TextSplitter()

#     def check_valid_url(self, url: str) -> bool:
#         """
#         Check if the provided URL is valid.

#         Args:
#             url: The URL to validate

#         Returns:
#             bool: True if URL is valid and accessible, False otherwise
#         """

#         # First check URL format without making a request
#         if not validators.url(url):
#             logger.error(f"Invalid URL format: {url}")
#             return False

#         try:
#             # Use a HEAD request with timeout to check availability
#             response = requests.head(url, timeout=5, allow_redirects=True)
#             if response.status_code >= 400:  # Consider all 4xx and 5xx as errors
#                 logger.warning(
#                     f"URL {url} returned status code {response.status_code}"
#                 )
#                 return False
#             return True
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error connecting to {url}: {e}")
#             return False

#     def load(self, urls: Union[str, List[str]]) -> List[Document]:
#         """
#         Load web content from a URL.

#         Args:
#             url: The URL to load content from

#         Returns:
#             List[Document]: A list of document chunks after splitting
#         """

#         if isinstance(urls, str):
#             urls = [urls]
#         # Validate each URL
#         valid_urls = []
#         for url in urls:
#             if self.check_valid_url(url):
#                 valid_urls.append(url)
#             else:
#                 logger.error(f"Invalid URL: {url}")

#         docs = [WebBaseLoader(url, {"User-Agent": USER_AGENT}).load() for url in valid_urls]
#         docs_list = [item for sublist in docs for item in sublist]

#         doc_splits = self.text_splitter(documents=docs_list)

#         return doc_splits

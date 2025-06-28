import base64
import json
import os
from io import StringIO
from typing import List, Tuple, TypedDict

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


class WDMPDFParser:
    def __init__(
        self,
        file_path: str = None,
        credential_path: str = None,
        debug: bool = False,
        debug_level: int = 1,
    ):
        self.file_path = file_path
        self.credential_path = credential_path
        self.debug = debug
        self.debug_level = debug_level
        

    def extract_tables(
        self,
        pages: List[int] = None,
        merge_span_tables: bool = True,
        enrich: bool = ENRICH_TABLES,
    ) -> List[Document]:
        all_tables: List[WDMMergedTable] = []
        if merge_span_tables or enrich:
            if not self.credential_path:
                error_msg = (
                    f"❌ Credentials required for advanced features (merge_span_tables={merge_span_tables}, enrich={enrich})\n"
                    f"Please provide credential_path parameter when initializing WDMPDFParser:\n\n"
                    "💡 Example:\n"
                    "parser = WDMPDFParser(\n"
                    "    file_path='your_file.pdf',\n"
                    "    credential_path='/path/to/your/service-account-key.json'\n"
                    ")\n"
                )
                raise ValueError(error_msg)

            # Validate credentials file exists
            if not os.path.exists(self.credential_path):
                raise FileNotFoundError(
                    f"Credentials file not found: {self.credential_path}\n"
                    "Please ensure the path is correct and the file exists."
                )

            if self.debug:
                print(f"✅ Using credentials: {self.credential_path}")

        if merge_span_tables:
            # Use full_pipeline for table merging
            merged_tables = full_pipeline(
                self.file_path,
                pages=pages,
                debug=self.debug,
                debug_level=self.debug_level,
                enrich=enrich,
                credential_path=self.credential_path,
            )
            all_tables = merged_tables
        else:
            # Use simple table extraction without merging
            from .extract_tables import get_tables_from_pdf
            individual_tables = get_tables_from_pdf(
                self.file_path,
                pages=pages,
                debug=self.debug,
                debug_level=self.debug_level,
                enrich=enrich,
                use_ai_analysis=False,  # Use basic mode for non-merged tables
                credential_path=self.credential_path,
            )
            # Convert WDMTable to WDMMergedTable format for consistency
            all_tables = []
            for table in individual_tables:
                merged_table = {
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
                # page_content=table['text'],
                metadata={
                    "page": str(table["page"]) if isinstance(table["page"], int) else ",".join(map(str, table["page"])),
                    "source": table["source"],
                    "type": "table",
                },
            )
            for table in all_tables
        ]
        return documents

    def extract_text(self, pages: List[int] = None) -> List[Document]:
        """
        Extract text from PDF pages, with option to ignore text within tables.
        Uses fallback mechanism: if IGNORE_TABLES=True fails, retry with IGNORE_TABLES=False.

        Args:
            pages: List of page numbers to extract (1-indexed). If None, extracts all pages.

        Returns:
            List of Document objects containing the extracted text.
        """
        # First attempt with ignore_tables setting
        try:
            return self._extract_text_internal(pages)
        except Exception as e:
            if IGNORE_TABLES:
                logger.warning(f"Text extraction with ignore_tables=True failed: {e}")
                logger.info("Retrying with ignore_tables=False (fallback mode)")
                try:
                    return self._extract_text_internal(pages)
                except Exception as fallback_error:
                    logger.error(f"Both text extraction methods failed: {fallback_error}")
                    return []
            else:
                logger.error(f"Text extraction failed: {e}")
                return []

    def _extract_text_internal(self, pages: List[int] = None) -> List[Document]:
        """
        Internal method to extract text from PDF pages.
        """
        doc = pymupdf.open(self.file_path)
        if pages is None:
            pages = list(range(1, len(doc) + 1))
        all_text: List[WDMText] = []

        try:
            for page_number in pages:
                try:
                    # Convert 1-indexed page number to 0-indexed for PyMuPDF
                    page = doc[page_number - 1]
                    
                    if IGNORE_TABLES:
                        # Find tables and create redaction annotations
                        table_bboxes = []
                        for tab in page.find_tables(strategy="lines_strict"):
                            table_bboxes.append(tab.bbox)
                            page.add_redact_annot(tab.bbox)
                        
                        # Only apply redactions if we found tables
                        if table_bboxes:
                            page.apply_redactions()
                    
                    # Extract text content
                    text_content = page.get_text()
                    
                    all_text.append(
                        WDMText(
                            text=text_content, 
                            page=page_number, 
                            source=self.file_path
                        )
                    )
                    
                    if self.debug:
                        char_count = len(text_content.strip())
                        mode = "with table redaction" if IGNORE_TABLES else "without table redaction"
                        logger.info(f"Extracted text from page {page_number} ({mode}): {char_count} characters")
                        
                except Exception as e:
                    # If we're in ignore_tables mode and get an error, raise it to trigger fallback
                    if IGNORE_TABLES and ("not a textpage" in str(e) or "textpage" in str(e)):
                        raise e
                    
                    logger.warning(f"Failed to extract text from page {page_number}: {e}")
                    # Continue with empty text for this page
                    all_text.append(
                        WDMText(
                            text="", 
                            page=page_number, 
                            source=self.file_path
                        )
                    )
                    
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
    
    
    # TODO: Phát triển tính năng trích xuất hình ảnh
    # def extract_images(
    #     self, pages: List[int] = None, stored_path: str = None
    # ) -> List[WDMImage]:
    #     # Create output directory if it doesn't exist
    #     if stored_path:
    #         os.makedirs(stored_path, exist_ok=True)

    #     images: List[WDMImage] = []
    #     doc = pymupdf.open(self.file_path)
    #     if pages is None:
    #         pages = range(1, len(doc) + 1)
    #     for page_index in pages:
    #         page = doc[page_index - 1]
    #         image_list = page.get_images(full=True)
    #         for img in image_list:
    #             xref = img[0]
    #             image = doc.extract_image(xref)
    #             image_bytes = image["image"]
    #             image_ext = image["ext"]

    #             # Only save to file if stored_path is provided
    #             image_path = None
    #             if stored_path:
    #                 image_filename = f"page_{page_index}_image_{xref}.{image_ext}"
    #                 image_path = os.path.join(stored_path, image_filename)
    #                 with open(image_path, "wb") as f:
    #                     f.write(image_bytes)

    #             images.append(
    #                 WDMImage(
    #                     base64_image=base64.b64encode(image_bytes).decode("utf-8"),
    #                     page=page_index,
    #                     source=self.file_path,
    #                     bbox=img[1],
    #                     image_path=image_path,
    #                 )
    #             )
    #     documents: List[Document] = [
    #         Document(
    #             page_content=image["base64_image"],
    #             metadata={
    #                 "page": image["page"],
    #                 "source": image["source"],
    #                 "type": "image",
    #             },
    #         )
    #         for image in images
    #     ]
    #     doc.close()
    #     return documents

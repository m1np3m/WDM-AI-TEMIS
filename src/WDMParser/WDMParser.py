import base64
import os
from typing import List, Tuple, TypedDict

import pymupdf

from .extract_tables import WDMMergedTable, WDMTable, full_pipeline, get_tables_from_pdf


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
        merge_span_tables: bool = False,
        enrich: bool = False,
    ) -> List[WDMTable | WDMMergedTable]:
        # Check credentials if advanced features are requested
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
            return full_pipeline(
                self.file_path,
                pages=pages,
                debug=self.debug,
                debug_level=self.debug_level,
                enrich=enrich,
                credential_path=self.credential_path,
            )
        else:
            # Use AI analysis when credentials are available
            use_ai_analysis = merge_span_tables or enrich
            return get_tables_from_pdf(
                self.file_path,
                pages=pages,
                debug=self.debug,
                debug_level=self.debug_level,
                enrich=enrich,
                use_ai_analysis=use_ai_analysis,
                credential_path=self.credential_path,
            )

    def extract_text(self, pages: List[int] = None) -> str:
        doc = pymupdf.open(self.file_path)
        if pages is None:
            pages = range(1, len(doc) + 1)
        all_text = []

        for page_number in pages:
            page = doc[page_number - 1]
            all_text.append(
                WDMText(
                    text=page.get_text(), page=page_number - 1, source=self.file_path
                )
            )
        return all_text

    def extract_images(
        self, pages: List[int] = None, stored_path: str = None
    ) -> List[WDMImage]:
        # Create output directory if it doesn't exist
        if stored_path:
            os.makedirs(stored_path, exist_ok=True)

        images = []
        doc = pymupdf.open(self.file_path)
        if pages is None:
            pages = range(1, len(doc) + 1)
        for page_index in pages:
            page = doc[page_index - 1]
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                image = doc.extract_image(xref)
                image_bytes = image["image"]
                image_ext = image["ext"]

                # Only save to file if stored_path is provided
                image_path = None
                if stored_path:
                    image_filename = f"page_{page_index}_image_{xref}.{image_ext}"
                    image_path = os.path.join(stored_path, image_filename)
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                images.append(
                    WDMImage(
                        base64_image=base64.b64encode(image_bytes).decode("utf-8"),
                        page=page_index,
                        source=self.file_path,
                        bbox=img[1],
                        image_path=image_path,
                    )
                )
        return images

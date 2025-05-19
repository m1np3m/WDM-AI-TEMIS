from typing import Optional


class Result:
    """A class representing a search result from a document.

    This class encapsulates information about a single result from a document search,
    including the document ID, page number, relevance score, and optional metadata
    and base64-encoded content.

    Attributes:
        doc_id (str): The unique identifier of the document.
        page_num (int): The page number within the document.
        score (float): The relevance score of the result.
        metadata (dict, optional): Additional metadata associated with the result.
        base64 (str, optional): Base64-encoded content of the result.
    """

    def __init__(
        self,
        doc_id: str,
        page_num: int,
        score: float,
        metadata: Optional[dict] = None,
        base64: Optional[str] = None,
    ):
        """Initialize a new Result instance.

        Args:
            doc_id (str): The unique identifier of the document.
            page_num (int): The page number within the document.
            score (float): The relevance score of the result.
            metadata (Optional[dict], optional): Additional metadata. Defaults to None.
            base64 (Optional[str], optional): Base64-encoded content. Defaults to None.
        """
        self.doc_id = doc_id
        self.page_num = page_num
        self.score = score
        self.metadata = metadata or {}
        self.base64 = base64

    def dict(self) -> dict:
        """Convert the Result instance to a dictionary.

        Returns:
            dict: A dictionary containing all the result attributes.
        """
        return {
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "score": self.score,
            "metadata": self.metadata,
            "base64": self.base64,
        }

    def __getitem__(self, key):
        """Allow dictionary-like access to attributes using square bracket notation.

        Args:
            key (str): The attribute name to access.

        Returns:
            The value of the requested attribute.

        Raises:
            AttributeError: If the requested attribute doesn't exist.
        """
        return getattr(self, key)

    def __str__(self) -> str:
        """Return a string representation of the Result instance.

        Returns:
            str: A string containing the dictionary representation of the result.
        """
        return str(self.dict())

    def __repr__(self) -> str:
        """Return a string representation of the Result instance.

        Returns:
            str: A string containing the dictionary representation of the result.
        """
        return self.__str__()
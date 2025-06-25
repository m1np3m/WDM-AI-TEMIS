"""WDM-AI-TEMIS: AI-powered document analysis and extraction system."""

__version__ = "0.1.0"
__author__ = "WDM Team"
__description__ = "AI-powered document analysis and extraction system"

# Import main classes for easy access
from src import WDMPDFParser, WDMText, WDMImage

__all__ = ["WDMPDFParser", "WDMText", "WDMImage"] 
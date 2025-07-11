#!/usr/bin/env python3
"""
Startup script for WDM AI-TEMIS FastAPI application
"""

import os
import sys
import uvicorn
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    """Main function to run the FastAPI application."""
    
    # Configure logging
    logger.add("logs/fastapi_app.log", rotation="500 MB", level="INFO")
    logger.info("Starting WDM AI-TEMIS FastAPI Server...")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1  # Single worker for development
    )

if __name__ == "__main__":
    main() 
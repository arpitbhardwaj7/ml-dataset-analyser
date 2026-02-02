#!/usr/bin/env python3
"""
Entry point for running the ML Dataset Analyser backend.
This script handles Python path setup and starts the FastAPI application.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Now we can import and run the app
if __name__ == "__main__":
    import uvicorn
    from app.main import app
    from app.core.config import settings
    
    print(f"🚀 Starting {settings.project_name} v{settings.project_version}")
    print(f"📖 API Documentation: http://{settings.host}:{settings.port}/docs")
    print(f"🔧 Configuration loaded successfully")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
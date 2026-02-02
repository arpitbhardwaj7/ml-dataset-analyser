from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.api.v1.router import api_router

# Create FastAPI application
app = FastAPI(
    title=settings.project_name,
    version=settings.project_version,
    description="""
    ML Dataset Analyser - A comprehensive tool for analyzing datasets and providing ML insights.
    
    ## Features
    
    * **Dataset Analysis**: Upload CSV/XLSX files for comprehensive analysis
    * **Quality Scoring**: Get quality scores across 5 dimensions (completeness, consistency, balance, dimensionality, separability)
    * **Auto-Detection**: Automatically detect problem type (classification/regression) and target columns
    * **Model Recommendations**: Get top 3 ML model recommendations based on dataset characteristics
    * **Preprocessing Guidance**: Receive prioritized preprocessing recommendations
    * **LLM Insights**: Get AI-powered insights and recommendations using GPT-4o
    * **Issue Detection**: Identify potential problems like class imbalance, missing values, outliers
    
    ## Supported File Formats
    
    * CSV files (UTF-8 encoding)
    * XLSX files (Excel format)
    
    ## API Endpoints
    
    * `POST /api/v1/analyze` - Main analysis endpoint
    * `GET /api/v1/health` - Health check
    * `GET /api/v1/models/supported` - List supported ML models
    * `GET /api/v1/preprocessing/methods` - List preprocessing methods
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add explicit OPTIONS handler for preflight requests
@app.options("/{path:path}")
async def options_handler(path: str):
    return {"message": "OK"}

# Include API router
app.include_router(api_router, prefix=settings.api_v1_prefix)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "HTTPException",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "InternalServerError", 
                "message": "An unexpected error occurred",
                "details": str(exc) if settings.debug else None
            }
        }
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.project_name}",
        "version": settings.project_version,
        "docs": "/docs",
        "status": "running"
    }

# Health check endpoint
@app.get("/ping")
async def ping():
    return {"message": "pong"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
import json

from app.services.analysis_service import AnalysisService
from app.utils.file_handler import FileHandler
from app.models.requests import AnalyzeDatasetRequest
from app.models.responses import AnalyzeDatasetResponse

router = APIRouter()

async def get_analysis_service() -> AnalysisService:
    """Dependency to get analysis service"""
    return AnalysisService()

@router.post("/analyze", response_model=AnalyzeDatasetResponse)
async def analyze_dataset(
    file: UploadFile = File(..., description="CSV or XLSX file to analyze"),
    target_column: Optional[str] = Form(None, description="Target column name (auto-detected if not provided)"),
    problem_type: Optional[str] = Form("auto", description="Problem type: 'classification', 'regression', or 'auto'"),
    use_llm_insights: bool = Form(True, description="Enable LLM-powered insights"),
    analysis_service: AnalysisService = Depends(get_analysis_service)
) -> AnalyzeDatasetResponse:
    """
    Analyze a dataset and provide comprehensive ML insights
    
    This endpoint accepts a CSV or XLSX file and returns:
    - Quality assessment with scoring
    - Data profiling and statistics
    - Model recommendations
    - Preprocessing suggestions
    - LLM-generated insights
    
    Parameters:
    - **file**: The dataset file (CSV or XLSX)
    - **target_column**: Name of target variable (optional - will auto-detect)
    - **problem_type**: Type of ML problem ('classification', 'regression', or 'auto')
    - **use_llm_insights**: Whether to generate LLM-powered insights (default: True)
    """
    
    # Normalize empty strings to None for target_column
    if target_column == "":
        target_column = None
    
    try:
        # Validate problem type
        valid_problem_types = ["auto", "classification", "regression"]
        if problem_type not in valid_problem_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid problem_type. Must be one of: {', '.join(valid_problem_types)}"
            )
        
        # Read and validate file
        df, metadata = await FileHandler.read_file(file)
        
        # Debug logging
        print(f"[DEBUG] Target column received: {target_column!r}")
        print(f"[DEBUG] Available columns in dataframe: {df.columns.tolist()}")
        
        # Perform analysis
        result = await analysis_service.analyze_dataset(
            df=df,
            metadata=metadata,
            target_column=target_column,
            problem_type=problem_type,
            use_llm_insights=use_llm_insights
        )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        print(f"[ERROR] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from app.core.config import settings
    from app.models.responses import HealthResponse
    from datetime import datetime
    
    return HealthResponse(
        status="healthy",
        version=settings.project_version,
        timestamp=datetime.now(),
        openai_configured=bool(settings.openai_api_key)
    )

@router.get("/models/supported")
async def get_supported_models():
    """Get list of supported machine learning models"""
    
    classification_models = [
        "XGBoost Classifier",
        "Random Forest Classifier", 
        "LightGBM Classifier",
        "Logistic Regression",
        "Support Vector Machine (SVM)"
    ]
    
    regression_models = [
        "XGBoost Regressor",
        "Random Forest Regressor",
        "Linear Regression", 
        "LightGBM Regressor",
        "ElasticNet Regression"
    ]
    
    return {
        "classification": classification_models,
        "regression": regression_models,
        "total_models": len(classification_models) + len(regression_models)
    }

@router.get("/preprocessing/methods")
async def get_preprocessing_methods():
    """Get list of available preprocessing methods"""
    
    return {
        "imputation": [
            "Simple Imputation",
            "KNN Imputation", 
            "Iterative Imputation"
        ],
        "encoding": [
            "One-Hot Encoding",
            "Label Encoding",
            "Target Encoding"
        ],
        "scaling": [
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ],
        "sampling": [
            "SMOTE",
            "Random Oversampling",
            "Random Undersampling",
            "Class Weights"
        ],
        "feature_selection": [
            "Univariate Selection",
            "Recursive Feature Elimination",
            "L1-based Selection"
        ]
    }
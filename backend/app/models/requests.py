from typing import Optional
from pydantic import BaseModel

class AnalyzeDatasetRequest(BaseModel):
    target_column: Optional[str] = None
    problem_type: Optional[str] = None  # "classification", "regression", or "auto"
    use_llm_insights: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_column": "churn",
                "problem_type": "auto",
                "use_llm_insights": True
            }
        }
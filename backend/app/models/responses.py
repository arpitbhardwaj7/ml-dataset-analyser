from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime

class QualityScoreBreakdown(BaseModel):
    score: float
    grade: str
    description: str

class QualityScore(BaseModel):
    overall: float
    grade: str
    breakdown: Dict[str, QualityScoreBreakdown]

class DatasetInfo(BaseModel):
    filename: str
    rows: int
    columns: int
    size_mb: float
    detected_problem_type: str
    detected_target_column: Optional[str]

class NumericSummary(BaseModel):
    total_features: int
    high_correlation_pairs: int
    low_variance_features: int

class CategoricalSummary(BaseModel):
    total_features: int
    high_cardinality: int
    binary_features: int

class DataProfile(BaseModel):
    missing_values_percentage: float
    duplicate_rows: int
    duplicate_percentage: float
    column_types: Dict[str, int]
    memory_usage_mb: float
    numeric_summary: NumericSummary
    categorical_summary: CategoricalSummary

class DetectedIssue(BaseModel):
    severity: str
    category: str
    title: str
    description: str
    impact: str
    affected_columns: List[str]
    recommendation: str

class ModelRecommendation(BaseModel):
    rank: int
    model_name: str
    confidence_score: int
    reasoning: str
    pros: List[str]
    cons: List[str]
    recommended_hyperparameters: Dict[str, str]
    expected_performance: str

class PreprocessingRecommendation(BaseModel):
    priority: int
    step: str
    category: str
    methods: List[str]
    code_snippet: str

class ActionItem(BaseModel):
    action: str
    impact: str
    effort: str

class RiskAssessment(BaseModel):
    overfitting_risk: str
    underfitting_risk: str
    data_leakage_risk: str
    curse_of_dimensionality: str

class LLMInsights(BaseModel):
    executive_summary: str
    detailed_analysis: str
    top_action_items: List[ActionItem]
    risk_assessment: RiskAssessment
    complete_preprocessing_pipeline: str


class Metadata(BaseModel):
    processing_time_seconds: float
    llm_tokens_used: Optional[int]
    api_version: str

class AnalyzeDatasetResponse(BaseModel):
    analysis_id: str
    timestamp: datetime
    dataset_info: DatasetInfo
    quality_score: QualityScore
    data_profile: DataProfile
    detected_issues: List[DetectedIssue]
    model_recommendations: List[ModelRecommendation]
    preprocessing_recommendations: List[PreprocessingRecommendation]
    llm_insights: Optional[LLMInsights]
    metadata: Metadata

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    openai_configured: bool
// TypeScript interfaces for the ML Dataset Analyzer

export interface DatasetInfo {
  filename: string;
  rows: number;
  columns: number;
  size_mb: number;
  detected_problem_type: string;
  detected_target_column?: string;
}

export interface QualityScore {
  overall: number;
  grade: string;
  breakdown: {
    [dimension: string]: {
      score: number;
      grade: string;
      description: string;
    };
  };
}

export interface DetectedIssue {
  id: string;
  title: string;
  description: string;
  severity: 'high' | 'medium' | 'low';
  category: string;
  recommendation: string;
  affected_columns?: string[];
}

export interface ModelRecommendation {
  model_name: string;
  confidence_score: number;
  reasoning: string;
  pros: string[];
  cons: string[];
  expected_performance?: string;
}

export interface LLMInsights {
  executive_summary?: string;
  risk_assessment?: {
    overfitting_risk: string;
    underfitting_risk: string;
    data_leakage_risk: string;
    curse_of_dimensionality: string;
  };
  top_action_items?: Array<{
    action: string;
    impact: 'HIGH' | 'MEDIUM' | 'LOW';
    effort: string;
  }>;
}

export interface DataProfile {
  missing_values_percentage: number;
  duplicate_rows: number;
  memory_usage_mb: number;
  column_types: {
    numerical: number;
    categorical: number;
    datetime: number;
    boolean: number;
  };
}

export interface AnalysisMetadata {
  processing_time_seconds: number;
  llm_tokens_used?: number;
  analysis_timestamp: string;
}

export interface AnalysisResults {
  success: boolean;
  dataset_info: DatasetInfo;
  quality_score: QualityScore;
  data_profile: DataProfile;
  detected_issues: DetectedIssue[];
  model_recommendations: ModelRecommendation[];
  llm_insights?: LLMInsights;
  metadata: AnalysisMetadata;
}

export interface AnalysisRequest {
  target_column?: string;
  problem_type: 'auto' | 'classification' | 'regression';
  use_llm_insights: boolean;
}

export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// UI State interfaces
export interface UploadState {
  file: File | null;
  isDragging: boolean;
  isUploading: boolean;
  uploadProgress: number;
}

export interface AnalysisState {
  isAnalyzing: boolean;
  progress: number;
  currentStep: string;
  results: AnalysisResults | null;
  error: string | null;
}

export interface MetricCardData {
  title: string;
  value: string | number;
  subtitle?: string;
  color?: string;
  icon?: string;
}

// Chart data interfaces
export interface ChartData {
  name: string;
  value: number;
  color?: string;
}

export interface GaugeData {
  value: number;
  max: number;
  grade: string;
  color: string;
}
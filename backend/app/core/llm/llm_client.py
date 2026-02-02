import json
from typing import Dict, Any, Optional
from openai import OpenAI
from app.core.config import settings
from app.core.llm.prompt_templates import PromptTemplates

class LLMClient:
    def __init__(self):
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.client = None
        self.prompt_templates = PromptTemplates()
    
    def is_available(self) -> bool:
        """Check if LLM client is available and configured"""
        return self.client is not None and settings.enable_llm
    
    async def generate_insights(
        self, 
        profile_data: Dict[str, Any], 
        quality_score: Dict[str, Any],
        model_recommendations: list,
        detected_issues: list,
        preprocessing_recommendations: list,
        dataset_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate comprehensive insights using GPT-4o"""
        
        if not self.is_available():
            return None
        
        try:
            # Prepare context for LLM
            analysis_context = {
                "dataset_info": dataset_info,
                "quality_score": quality_score,
                "profile_summary": self._extract_profile_summary(profile_data),
                "detected_issues": detected_issues,
                "model_recommendations": model_recommendations[:3],  # Top 3 only
                "preprocessing_recommendations": preprocessing_recommendations[:5]  # Top 5 only
            }
            
            # Generate prompt
            prompt = self.prompt_templates.create_analysis_prompt(analysis_context)
            
            # Call GPT-4o
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": self.prompt_templates.get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=settings.llm_max_tokens,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            insights_text = response.choices[0].message.content
            insights = json.loads(insights_text)
            
            # Add metadata
            insights["tokens_used"] = response.usage.total_tokens if response.usage else 0
            
            return insights
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse LLM response: {str(e)}",
                "executive_summary": "LLM analysis failed - JSON parsing error",
                "detailed_analysis": "Unable to generate detailed analysis due to parsing error",
                "top_action_items": [],
                "risk_assessment": {
                    "overfitting_risk": "Unknown",
                    "underfitting_risk": "Unknown", 
                    "data_leakage_risk": "Unknown",
                    "curse_of_dimensionality": "Unknown"
                },
                "complete_preprocessing_pipeline": "# LLM analysis unavailable",
                "tokens_used": 0
            }
        except Exception as e:
            return {
                "error": f"LLM analysis failed: {str(e)}",
                "executive_summary": "LLM analysis unavailable due to technical error",
                "detailed_analysis": "Unable to generate detailed analysis due to technical error",
                "top_action_items": [],
                "risk_assessment": {
                    "overfitting_risk": "Unknown",
                    "underfitting_risk": "Unknown",
                    "data_leakage_risk": "Unknown", 
                    "curse_of_dimensionality": "Unknown"
                },
                "complete_preprocessing_pipeline": "# LLM analysis unavailable",
                "tokens_used": 0
            }
    
    def _extract_profile_summary(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from profile data for LLM"""
        
        summary = {}
        
        # Basic info
        basic_info = profile_data.get("basic_info", {})
        summary["shape"] = basic_info.get("shape", (0, 0))
        summary["memory_usage_mb"] = round(basic_info.get("memory_usage_mb", 0), 2)
        
        # Missing values
        missing_values = profile_data.get("missing_values", {})
        summary["missing_percentage"] = missing_values.get("overall_missing_percentage", 0)
        summary["columns_with_missing"] = len(missing_values.get("columns_with_missing", []))
        
        # Duplicates
        duplicates = profile_data.get("duplicates", {})
        summary["duplicate_percentage"] = duplicates.get("duplicate_percentage", 0)
        
        # Column types
        column_types = profile_data.get("column_types", {})
        summary["numerical_columns"] = column_types.get("numerical", 0)
        summary["categorical_columns"] = column_types.get("categorical", 0)
        summary["datetime_columns"] = column_types.get("datetime", 0)
        
        # Correlations
        correlations = profile_data.get("correlations", {})
        summary["max_correlation"] = correlations.get("max_correlation", 0)
        summary["high_correlation_pairs"] = len(correlations.get("high_correlation_pairs", []))
        
        # Outliers summary
        outliers = profile_data.get("outliers", {})
        outlier_percentages = [info.get("outlier_percentage", 0) for info in outliers.values()]
        summary["avg_outlier_percentage"] = round(sum(outlier_percentages) / len(outlier_percentages), 2) if outlier_percentages else 0
        summary["columns_with_outliers"] = sum(1 for pct in outlier_percentages if pct > 5)
        
        # Target analysis
        target_analysis = profile_data.get("target_analysis", {})
        if target_analysis:
            summary["target_column"] = target_analysis.get("column_name")
            summary["problem_type"] = target_analysis.get("problem_type")
            summary["target_missing"] = target_analysis.get("missing_values", 0)
            
            # Class balance for classification
            if "class_balance" in target_analysis:
                class_balance = target_analysis["class_balance"]
                summary["class_imbalance_ratio"] = class_balance.get("imbalance_ratio", 1)
                summary["is_balanced"] = class_balance.get("balanced", True)
        
        # Feature importance summary
        feature_importance = profile_data.get("feature_importance", {})
        if isinstance(feature_importance, dict) and "error" not in feature_importance:
            importance_values = list(feature_importance.values())
            if importance_values:
                summary["max_feature_importance"] = max(importance_values)
                summary["avg_feature_importance"] = sum(importance_values) / len(importance_values)
                summary["features_with_high_importance"] = sum(1 for imp in importance_values if imp > 0.1)
        
        return summary
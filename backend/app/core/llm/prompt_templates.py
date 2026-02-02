from typing import Dict, Any

class PromptTemplates:
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """You are an expert ML engineering assistant specializing in dataset analysis and machine learning recommendations. Your role is to analyze dataset profiles and provide actionable insights for ML engineers.

Key responsibilities:
1. Interpret statistical analysis results and translate them into actionable insights
2. Provide prioritized recommendations for data preprocessing and model selection
3. Identify potential ML pitfalls (overfitting, underfitting, data leakage, etc.)
4. Generate practical Python code for preprocessing pipelines
5. Explain complex concepts in a clear, technical manner suitable for ML engineers

Guidelines:
        - Be specific and actionable in your recommendations
        - Rank suggestions based on impact and effort required (use Low/Medium/High categories)
        - Focus on practical ML engineering concerns
- Use technical terminology appropriate for ML practitioners
- Provide concrete next steps and code examples
- Always respond with valid JSON format as specified

Your response must be a valid JSON object with the exact structure specified in the user prompt."""

    def create_analysis_prompt(self, analysis_context: Dict[str, Any]) -> str:
        """Create the main analysis prompt with context"""
        
        dataset_info = analysis_context["dataset_info"]
        quality_score = analysis_context["quality_score"]
        profile_summary = analysis_context["profile_summary"]
        detected_issues = analysis_context["detected_issues"]
        model_recommendations = analysis_context["model_recommendations"]
        preprocessing_recommendations = analysis_context["preprocessing_recommendations"]
        
        prompt = f"""Analyze this dataset and provide comprehensive ML engineering insights.

## Dataset Information:
- Filename: {dataset_info.get('filename', 'Unknown')}
- Shape: {dataset_info.get('rows', 0)} rows × {dataset_info.get('columns', 0)} columns
- Size: {dataset_info.get('size_mb', 0):.2f} MB
- Problem Type: {dataset_info.get('detected_problem_type', 'Unknown')}
- Target Column: {dataset_info.get('detected_target_column', 'None')}

## Quality Assessment:
- Overall Score: {quality_score.get('overall', 0)}/100 (Grade: {quality_score.get('grade', 'F')})
- Completeness: {quality_score.get('breakdown', {}).get('completeness', {}).get('score', 0)}/100
- Consistency: {quality_score.get('breakdown', {}).get('consistency', {}).get('score', 0)}/100  
- Balance: {quality_score.get('breakdown', {}).get('balance', {}).get('score', 0)}/100
- Dimensionality: {quality_score.get('breakdown', {}).get('dimensionality', {}).get('score', 0)}/100
- Separability: {quality_score.get('breakdown', {}).get('separability', {}).get('score', 0)}/100

## Dataset Profile Summary:
- Missing Data: {profile_summary.get('missing_percentage', 0):.1f}% overall
- Duplicates: {profile_summary.get('duplicate_percentage', 0):.1f}%
- Column Types: {profile_summary.get('numerical_columns', 0)} numerical, {profile_summary.get('categorical_columns', 0)} categorical
- Max Correlation: {profile_summary.get('max_correlation', 0):.3f}
- Outlier Issues: {profile_summary.get('columns_with_outliers', 0)} columns with >5% outliers
- Class Imbalance: {profile_summary.get('class_imbalance_ratio', 1):.2f}:1 ratio
- Feature Importance: Max = {profile_summary.get('max_feature_importance', 0):.3f}

## Detected Issues:
{self._format_issues(detected_issues)}

## Top Model Recommendations:
{self._format_model_recommendations(model_recommendations)}

## Preprocessing Recommendations:
{self._format_preprocessing_recommendations(preprocessing_recommendations)}

Based on this analysis, provide comprehensive insights in the following JSON format:

{{
  "executive_summary": "2-3 sentence summary of dataset quality and key recommendations",
  "detailed_analysis": "Detailed technical analysis paragraph (200-300 words) explaining the dataset characteristics, main challenges, and why certain approaches are recommended",
  "top_action_items": [
    {
      "action": "Specific action to take",
      "impact": "HIGH/MEDIUM/LOW - expected impact on model performance",
      "effort": "1-8 hours or Priority 1-5 - estimated time/priority to implement"
    }
  ],
  "risk_assessment": {{
    "overfitting_risk": "High/Medium/Low - explanation of overfitting risk factors",
    "underfitting_risk": "High/Medium/Low - explanation of underfitting risk factors", 
    "data_leakage_risk": "High/Medium/Low - potential data leakage concerns",
    "curse_of_dimensionality": "High/Medium/Low - dimensionality-related risks"
  }},
  "complete_preprocessing_pipeline": "# Complete Python preprocessing pipeline\\n# Include all major preprocessing steps\\n# Use appropriate libraries (pandas, sklearn, etc.)\\n# Handle missing values, encoding, scaling, etc.\\n# Add comments explaining each step\\n\\nimport pandas as pd\\nfrom sklearn.preprocessing import StandardScaler\\n# ... complete working code here"
}}

Ensure your response is valid JSON and addresses all the key concerns for this specific dataset."""

        return prompt
    
    def _format_issues(self, detected_issues: list) -> str:
        """Format detected issues for the prompt"""
        if not detected_issues:
            return "No major issues detected."
        
        formatted = []
        for issue in detected_issues[:5]:  # Limit to top 5 issues
            formatted.append(f"- {issue.get('severity', 'unknown').upper()}: {issue.get('title', 'Unknown issue')} - {issue.get('description', 'No description')}")
        
        return "\n".join(formatted)
    
    def _format_model_recommendations(self, model_recommendations: list) -> str:
        """Format model recommendations for the prompt"""
        if not model_recommendations:
            return "No model recommendations available."
        
        formatted = []
        for i, model in enumerate(model_recommendations[:3], 1):
            formatted.append(f"{i}. {model.get('model_name', 'Unknown')} (Confidence: {model.get('confidence_score', 0)}%) - {model.get('reasoning', 'No reasoning provided')}")
        
        return "\n".join(formatted)
    
    def _format_preprocessing_recommendations(self, preprocessing_recommendations: list) -> str:
        """Format preprocessing recommendations for the prompt"""
        if not preprocessing_recommendations:
            return "No preprocessing recommendations available."
        
        formatted = []
        for i, rec in enumerate(preprocessing_recommendations[:5], 1):  # Top 5
            methods_text = ', '.join(rec.get('methods', [])) if rec.get('methods') else 'various methods'
            formatted.append(f"{i}. {rec.get('step', 'Unknown step')} - Category: {rec.get('category', 'unknown')} using {methods_text}")
        
        return "\n".join(formatted)

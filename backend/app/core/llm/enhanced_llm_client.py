import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
from app.core.config import settings
from app.core.llm.prompt_templates import PromptTemplates
from app.core.analyzer.smart_row_sampler import SmartRowSampler


class EnhancedLLMClient:
    """Enhanced LLM client with advanced data quality assessment capabilities"""
    
    def __init__(self):
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.client = None
        self.prompt_templates = PromptTemplates()
        self.sampler = SmartRowSampler(max_samples=50)
    
    def is_available(self) -> bool:
        """Check if LLM client is available and configured"""
        return self.client is not None and settings.enable_llm
    
    async def assess_data_quality_issues(
        self,
        df,
        profile_data: Dict[str, Any], 
        quality_score: Dict[str, Any],
        detected_issues: List[Dict[str, Any]],
        dataset_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Enhanced data quality assessment using problematic row samples
        Returns: Severity-adjusted analysis with AI reasoning
        """
        
        if not self.is_available():
            return None
        
        try:
            # 1. Select smart samples of problematic rows
            problem_samples = self.sampler.select_for_llm_analysis(df, profile_data)
            
            # 2. Create enhanced quality assessment prompt
            prompt = self._create_quality_assessment_prompt(
                problem_samples=problem_samples,
                profile_data=profile_data,
                quality_score=quality_score,
                detected_issues=detected_issues,
                dataset_context=dataset_context
            )
            
            # 3. Call GPT-4o for analysis
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_enhanced_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=settings.llm_max_tokens * 2,  # Allow more tokens for detailed analysis
                temperature=0.2,  # Lower temperature for more consistent analysis
                response_format={"type": "json_object"}
            )
            
            # 4. Parse response
            analysis_text = response.choices[0].message.content
            analysis = json.loads(analysis_text)
            
            # 5. Add metadata
            analysis["tokens_used"] = response.usage.total_tokens if response.usage else 0
            analysis["samples_analyzed"] = problem_samples['total_samples']
            analysis["critical_issues_found"] = problem_samples['summary']['critical_count']
            analysis["high_issues_found"] = problem_samples['summary']['high_count']
            
            return analysis
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse LLM response: {str(e)}",
                "adjusted_quality_score": quality_score.get('overall', 0),
                "severity_explanation": "LLM analysis failed - using original quality score",
                "blocking_issues": [],
                "recommended_fixes": [],
                "cleanup_effort_hours": 0,
                "tokens_used": 0
            }
        except Exception as e:
            return {
                "error": f"Enhanced LLM analysis failed: {str(e)}",
                "adjusted_quality_score": quality_score.get('overall', 0),
                "severity_explanation": "LLM analysis unavailable due to technical error",
                "blocking_issues": [],
                "recommended_fixes": [],
                "cleanup_effort_hours": 0,
                "tokens_used": 0
            }
    
    async def generate_cleaning_pipeline(
        self,
        problem_samples: Dict[str, Any],
        profile_data: Dict[str, Any],
        dataset_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate automated data cleaning pipeline based on detected issues
        """
        
        if not self.is_available():
            return None
        
        try:
            prompt = self._create_cleaning_pipeline_prompt(
                problem_samples, profile_data, dataset_context
            )
            
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data engineer. Generate complete, executable Python data cleaning pipelines. Always respond in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=settings.llm_max_tokens * 2,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            pipeline_text = response.choices[0].message.content
            pipeline = json.loads(pipeline_text)
            pipeline["tokens_used"] = response.usage.total_tokens if response.usage else 0
            
            return pipeline
            
        except Exception as e:
            return {
                "error": f"Pipeline generation failed: {str(e)}",
                "pipeline_code": "# Pipeline generation failed",
                "steps": [],
                "tokens_used": 0
            }
    
    def _get_enhanced_system_prompt(self) -> str:
        """Enhanced system prompt for quality assessment"""
        return """You are an expert ML data quality analyst with deep expertise in identifying data issues that impact machine learning pipelines.

Your role is to provide **accurate, calibrated, and evidence-based** quality assessments.

Key responsibilities:
1. Identify issues that prevent or degrade ML model training
2. Detect semantic data problems (mixed types, non-standard nulls, inconsistent formats)
3. Provide **fair and accurate** quality scores based on actual impact to ML readiness
4. Recommend specific, actionable fixes with Python code
5. Estimate realistic cleanup effort in hours

Guidelines for severity assessment:
- CRITICAL: Issues that prevent model training (mixed data types, semantic nulls in >20% of data)
- HIGH: Issues that significantly reduce model accuracy (format inconsistencies affecting >10%, high missing data >30%)
- MEDIUM: Issues that moderately impact performance (outliers, minor inconsistencies affecting 5-10%)
- LOW: Issues that have minimal impact (few statistical outliers <5%, cosmetic issues)

Quality scoring guidelines:
- Score based on **actual ML impact**, not theoretical perfection
- Datasets with CRITICAL issues blocking training: <50%
- Datasets with HIGH severity issues significantly impacting accuracy: 50-70%
- Datasets with MEDIUM issues requiring moderate cleanup: 70-85%
- Datasets with only minor issues: 85-95%
- Near-perfect datasets (rare): 95-100%
- **Adjust scores UP or DOWN** based on what you find - be accurate, not pessimistic or optimistic
- Provide confidence level: HIGH (very certain), MEDIUM (somewhat certain), LOW (uncertain)

Always respond with valid JSON in the exact structure specified."""
    
    def _create_quality_assessment_prompt(
        self,
        problem_samples: Dict[str, Any],
        profile_data: Dict[str, Any],
        quality_score: Dict[str, Any],
        detected_issues: List[Dict[str, Any]],
        dataset_context: Dict[str, Any]
    ) -> str:
        """Create enhanced quality assessment prompt with sample analysis"""
        
        # Format sample issues by severity
        samples_by_severity = problem_samples.get('by_severity', {})
        
        critical_samples = self._format_samples_for_prompt(samples_by_severity.get('CRITICAL', []), 'CRITICAL')
        high_samples = self._format_samples_for_prompt(samples_by_severity.get('HIGH', []), 'HIGH')
        medium_samples = self._format_samples_for_prompt(samples_by_severity.get('MEDIUM', []), 'MEDIUM')
        baseline_samples = self._format_samples_for_prompt(samples_by_severity.get('NONE', []), 'BASELINE')
        
        prompt = f"""Analyze this dataset's quality issues using the provided problematic sample rows.

## Dataset Information:
- Filename: {dataset_context.get('filename', 'Unknown')}
- Total rows: {dataset_context.get('total_rows', 0):,}
- Total columns: {len(dataset_context.get('columns', []))}
- Columns: {', '.join(dataset_context.get('columns', []))}

## Current Deterministic Quality Score:
- Overall: {quality_score.get('overall', 0)}/100 (Grade: {quality_score.get('grade', 'F')})
- This is a heuristic-based score that may miss semantic issues

## Analyzed {problem_samples.get('total_samples', 0)} Representative Samples:

### CRITICAL ISSUES ({problem_samples['summary']['critical_count']} samples):
These issues completely block ML model training:
{critical_samples}

### HIGH SEVERITY ISSUES ({problem_samples['summary']['high_count']} samples):
These significantly reduce model accuracy:
{high_samples}

### MEDIUM SEVERITY ISSUES ({problem_samples['summary']['medium_count']} samples):
These moderately impact performance:
{medium_samples}

### BASELINE SAMPLES ({problem_samples['summary']['baseline_count']} samples):
Relatively clean rows for comparison:
{baseline_samples}

## Original Issues Detected by Heuristics:
{self._format_original_issues(detected_issues)}

## Your Task:
Review the sample data and determine if the deterministic score is accurate, too high, or too low.
Consider:
- Are there semantic issues the heuristics missed?
- Are the detected issues actually as severe as flagged?
- What's the realistic impact on ML model training?

Provide your assessment in JSON format:

{{
  "confidence_level": "HIGH|MEDIUM|LOW - how confident are you in this assessment",
  "adjusted_quality_score": "number 0-100 based on actual ML impact (can be higher OR lower than {quality_score.get('overall', 0)})",
  "adjusted_grade": "letter grade A-F",
  "score_adjustment_reasoning": "2-3 sentences explaining why you kept the score the same, raised it, or lowered it",
  
  "blocking_issues": [
    {{
      "issue_type": "specific issue category",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "description": "clear description with evidence",
      "affected_columns": ["col1", "col2"],
      "sample_evidence": ["row 5: has 'thirty' mixed with 30", "row 12: has 'N/A' string"],
      "estimated_rows_affected": "percentage or count",
      "impact_on_ml": "specific impact description",
      "fix_priority": "1-10"
    }}
  ],
  
  "recommended_fixes": [
    {{
      "issue_type": "issue being fixed",
      "fix_description": "what to do",
      "python_code": "# Complete working Python code\\nimport pandas as pd\\n...",
      "estimated_effort_hours": "0.5-8 hours",
      "data_loss_risk": "High|Medium|Low"
    }}
  ],
  
  "total_cleanup_effort_hours": "sum of all fixes",
  "ml_readiness_assessment": "Ready|Needs Minor Cleanup|Needs Major Work|Not Ready",
  "biggest_concerns": ["ranked list of most serious issues"],
  
  "validation_recommendations": {{
    "should_validate_llm_findings": true,
    "specific_rows_to_check": [5, 12, 47],
    "columns_to_inspect": ["col1", "col2"]
  }}
}}"""
        
        return prompt
    
    def _create_cleaning_pipeline_prompt(
        self,
        problem_samples: Dict[str, Any],
        profile_data: Dict[str, Any],
        dataset_context: Dict[str, Any]
    ) -> str:
        """Create prompt for generating cleaning pipeline"""
        
        critical_issues = problem_samples.get('by_severity', {}).get('CRITICAL', [])
        high_issues = problem_samples.get('by_severity', {}).get('HIGH', [])
        
        prompt = f"""Generate a complete Python data cleaning pipeline for this dataset.

Dataset: {dataset_context.get('filename')} ({dataset_context.get('total_rows')} rows x {len(dataset_context.get('columns', []))} columns)

Critical Issues Found:
{self._format_samples_for_prompt(critical_issues, 'CRITICAL')}

High Priority Issues:
{self._format_samples_for_prompt(high_issues, 'HIGH')}

Generate a complete cleaning pipeline in JSON format:

{{
  "pipeline_code": "# Complete Python script\\nimport pandas as pd\\nimport numpy as np\\nfrom sklearn.preprocessing import LabelEncoder\\n\\ndef clean_dataset(df):\\n    # Step 1: Handle mixed data types\\n    # ... complete implementation\\n    return df\\n\\n# Usage\\n# df_clean = clean_dataset(df)",
  
  "steps": [
    {{
      "step_number": 1,
      "description": "Handle mixed data types in Age column",
      "code_snippet": "# Convert text numbers to numeric\\nage_map = {{'thirty': 30, 'twenty': 20}}\\ndf['Age'] = df['Age'].map(age_map).fillna(df['Age'])",
      "validation": "assert df['Age'].dtype in ['int64', 'float64']"
    }}
  ],
  
  "before_after_comparison": {{
    "original_issues": "list of problems in original data",
    "expected_improvements": "what will be fixed",
    "remaining_challenges": "what issues might persist"
  }},
  
  "execution_notes": [
    "Important warnings or considerations",
    "Order of operations matters",
    "Backup data before running"
  ]
}}

Ensure the pipeline handles all detected issues and is production-ready."""
        
        return prompt
    
    def _format_samples_for_prompt(self, samples: List[Dict[str, Any]], severity: str) -> str:
        """Format samples for inclusion in LLM prompt"""
        if not samples:
            return f"No {severity} issues found in samples."
        
        formatted = []
        for i, sample in enumerate(samples[:5], 1):  # Limit to 5 examples per severity
            issue_desc = sample.get('issue_description', 'Unknown issue')
            row_data = sample.get('row_data', {})
            additional_info = sample.get('additional_info', {})
            
            # Show only relevant columns to save tokens
            relevant_data = {k: v for k, v in row_data.items() if v is not None and str(v).strip() != ''}
            if len(relevant_data) > 6:  # Limit columns shown
                relevant_data = dict(list(relevant_data.items())[:6])
            
            formatted.append(f"""
Example {i}: {issue_desc}
Row data: {relevant_data}
Additional context: {additional_info}""")
        
        return "\n".join(formatted) if formatted else f"No {severity} samples available."
    
    def _format_original_issues(self, detected_issues: List[Dict[str, Any]]) -> str:
        """Format original detected issues for prompt"""
        if not detected_issues:
            return "No issues detected by basic analysis."
        
        formatted = []
        for issue in detected_issues[:5]:  # Top 5 issues
            formatted.append(f"- {issue.get('severity', 'UNKNOWN')}: {issue.get('title', 'Unknown')} - {issue.get('description', 'No description')}")
        
        return "\n".join(formatted)
    
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
    
    def add_confidence_scores(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Add confidence scores to LLM response based on multiple factors"""
        
        if not llm_response or 'error' in llm_response:
            return llm_response
        
        confidence_factors = []
        confidence_score = 0.0
        
        # Factor 1: Confidence level reported by LLM itself
        stated_confidence = llm_response.get('confidence_level', 'MEDIUM')
        if stated_confidence == 'HIGH':
            confidence_score += 0.4
            confidence_factors.append('High LLM confidence')
        elif stated_confidence == 'MEDIUM':
            confidence_score += 0.25
            confidence_factors.append('Medium LLM confidence')
        else:  # LOW
            confidence_score += 0.1
            confidence_factors.append('Low LLM confidence')
        
        # Factor 2: Evidence quality (sample evidence provided)
        blocking_issues = llm_response.get('blocking_issues', [])
        evidence_quality = 0.0
        for issue in blocking_issues:
            sample_evidence = issue.get('sample_evidence', [])
            if len(sample_evidence) >= 2:
                evidence_quality += 0.1  # Good evidence
            elif len(sample_evidence) == 1:
                evidence_quality += 0.05  # Some evidence
        
        evidence_quality = min(evidence_quality, 0.3)  # Cap at 0.3
        confidence_score += evidence_quality
        if evidence_quality > 0.2:
            confidence_factors.append('Strong sample evidence')
        elif evidence_quality > 0.1:
            confidence_factors.append('Moderate sample evidence')
        elif evidence_quality > 0:
            confidence_factors.append('Limited sample evidence')
        else:
            confidence_factors.append('No sample evidence provided')
        
        # Factor 3: Response completeness
        required_fields = ['adjusted_quality_score', 'blocking_issues', 'recommended_fixes']
        completeness = sum(1 for field in required_fields if field in llm_response and llm_response[field])
        completeness_score = (completeness / len(required_fields)) * 0.2
        confidence_score += completeness_score
        
        if completeness == len(required_fields):
            confidence_factors.append('Complete response')
        else:
            confidence_factors.append(f'Partial response ({completeness}/{len(required_fields)} fields)')
        
        # Factor 4: Consistency with samples analyzed
        samples_analyzed = llm_response.get('samples_analyzed', 0)
        critical_found = llm_response.get('critical_issues_found', 0)
        high_found = llm_response.get('high_issues_found', 0)
        
        consistency_score = 0.0
        if samples_analyzed > 20:  # Good sample size
            consistency_score += 0.05
            confidence_factors.append('Large sample size analyzed')
        elif samples_analyzed > 10:
            consistency_score += 0.03
            confidence_factors.append('Moderate sample size')
        elif samples_analyzed > 0:
            consistency_score += 0.01
            confidence_factors.append('Small sample size')
        
        # Check if findings align with samples
        if critical_found > 0 and len(blocking_issues) > 0:
            consistency_score += 0.05
            confidence_factors.append('Findings align with critical samples')
        
        confidence_score += consistency_score
        
        # Cap confidence score at 1.0
        confidence_score = min(confidence_score, 1.0)
        
        # Convert to percentage and categorize
        confidence_percentage = int(confidence_score * 100)
        
        if confidence_percentage >= 80:
            confidence_level = 'VERY_HIGH'
            confidence_description = 'Very confident in assessment'
        elif confidence_percentage >= 65:
            confidence_level = 'HIGH'
            confidence_description = 'Confident in assessment'
        elif confidence_percentage >= 50:
            confidence_level = 'MEDIUM'
            confidence_description = 'Moderately confident'
        elif confidence_percentage >= 35:
            confidence_level = 'LOW'
            confidence_description = 'Low confidence'
        else:
            confidence_level = 'VERY_LOW'
            confidence_description = 'Very low confidence'
        
        # Add confidence metadata to response
        llm_response['confidence_metadata'] = {
            'overall_confidence_score': confidence_percentage,
            'confidence_level': confidence_level,
            'confidence_description': confidence_description,
            'confidence_factors': confidence_factors,
            'reliability_assessment': self._assess_reliability(confidence_level, llm_response),
            'recommendation': self._get_confidence_recommendation(confidence_level)
        }
        
        return llm_response
    
    def _assess_reliability(self, confidence_level: str, response: Dict[str, Any]) -> str:
        """Assess overall reliability of the LLM response"""
        
        if confidence_level in ['VERY_HIGH', 'HIGH']:
            return 'RELIABLE'
        elif confidence_level == 'MEDIUM':
            # Check if critical issues were found with evidence
            blocking_issues = response.get('blocking_issues', [])
            critical_issues = [issue for issue in blocking_issues if issue.get('severity') == 'CRITICAL']
            
            if critical_issues and any(issue.get('sample_evidence') for issue in critical_issues):
                return 'MODERATELY_RELIABLE'
            else:
                return 'QUESTIONABLE'
        else:  # LOW or VERY_LOW
            return 'UNRELIABLE'
    
    def _get_confidence_recommendation(self, confidence_level: str) -> str:
        """Get recommendation based on confidence level"""
        
        recommendations = {
            'VERY_HIGH': 'Trust this assessment - high confidence with strong evidence',
            'HIGH': 'Generally reliable assessment - proceed with normal validation',
            'MEDIUM': 'Moderate confidence - validate key findings before making decisions',
            'LOW': 'Low confidence - manually verify findings and consider alternative assessment',
            'VERY_LOW': 'Very low confidence - do not rely on this assessment without extensive validation'
        }
        
        return recommendations.get(confidence_level, 'Review assessment carefully')
    
    def calculate_response_consistency(self, 
                                      llm_response: Dict[str, Any], 
                                      deterministic_score: float,
                                      samples_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how consistent the LLM response is with input data"""
        
        consistency_metrics = {
            'score_alignment': 0.0,
            'sample_alignment': 0.0, 
            'severity_alignment': 0.0,
            'overall_consistency': 0.0
        }
        
        # Score alignment (how close is LLM score to deterministic)
        llm_score = llm_response.get('adjusted_quality_score', deterministic_score)
        score_diff = abs(llm_score - deterministic_score)
        
        if score_diff <= 5:
            consistency_metrics['score_alignment'] = 1.0
        elif score_diff <= 10:
            consistency_metrics['score_alignment'] = 0.8
        elif score_diff <= 20:
            consistency_metrics['score_alignment'] = 0.6
        elif score_diff <= 30:
            consistency_metrics['score_alignment'] = 0.4
        else:
            consistency_metrics['score_alignment'] = 0.2
        
        # Sample alignment (do findings match sample severity?)
        critical_samples = samples_data.get('summary', {}).get('critical_count', 0)
        high_samples = samples_data.get('summary', {}).get('high_count', 0)
        
        blocking_issues = llm_response.get('blocking_issues', [])
        critical_findings = len([issue for issue in blocking_issues if issue.get('severity') == 'CRITICAL'])
        high_findings = len([issue for issue in blocking_issues if issue.get('severity') == 'HIGH'])
        
        sample_alignment = 0.0
        if critical_samples > 0 and critical_findings > 0:
            sample_alignment += 0.5
        elif critical_samples == 0 and critical_findings == 0:
            sample_alignment += 0.5
        
        if high_samples > 0 and high_findings > 0:
            sample_alignment += 0.5
        elif high_samples == 0 and high_findings == 0:
            sample_alignment += 0.5
        
        consistency_metrics['sample_alignment'] = sample_alignment
        
        # Overall consistency
        consistency_metrics['overall_consistency'] = (
            consistency_metrics['score_alignment'] * 0.5 + 
            consistency_metrics['sample_alignment'] * 0.5
        )
        
        return consistency_metrics

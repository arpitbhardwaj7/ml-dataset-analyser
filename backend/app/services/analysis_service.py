import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import HTTPException

from app.core.analyzer.data_profiler import DataProfiler
from app.core.analyzer.quality_scorer import QualityScorer
from app.core.analyzer.model_recommender import ModelRecommender
from app.core.analyzer.blended_scorer import BlendedScorer
from app.core.analyzer.llm_validator import LLMFindingValidator
from app.core.llm.llm_client import LLMClient
from app.core.llm.enhanced_llm_client import EnhancedLLMClient
from app.utils.file_handler import FileHandler
from app.models.responses import *

class AnalysisService:
    def __init__(self):
        self.llm_client = LLMClient()
        self.enhanced_llm_client = EnhancedLLMClient()
    
    async def analyze_dataset(
        self, 
        df: pd.DataFrame, 
        metadata: Dict[str, Any], 
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        use_llm_insights: bool = True
    ) -> AnalyzeDatasetResponse:
        """Perform complete dataset analysis"""
        
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        
        try:
            # Step 1: Auto-detect target column if not provided
            detected_target_column = FileHandler.detect_target_column(df, target_column)
            
            # Step 2: Auto-detect problem type
            if problem_type == "auto" or not problem_type:
                if detected_target_column:
                    detected_problem_type = FileHandler.detect_problem_type(df[detected_target_column])
                else:
                    detected_problem_type = "classification"  # Default assumption
            else:
                detected_problem_type = problem_type
            
            # Step 3: Generate data profile
            profiler = DataProfiler(df, detected_target_column)
            profile_data = profiler.generate_profile()
            
            # Step 4: Calculate quality scores
            scorer = QualityScorer(df, profile_data, detected_target_column)
            quality_score_data = scorer.calculate_quality_scores()
            
            # Step 5: Get model recommendations
            recommender = ModelRecommender(df, profile_data, detected_problem_type, detected_target_column)
            model_recommendations = recommender.get_model_recommendations()
            
            # Step 6: Generate detected issues
            detected_issues = self._generate_detected_issues(profile_data, quality_score_data)
            
            # Step 7: Generate preprocessing recommendations
            preprocessing_recommendations = self._generate_preprocessing_recommendations(
                profile_data, detected_problem_type, detected_target_column
            )
            
            # Step 8: Generate Enhanced LLM insights with validation and blended scoring (if enabled)
            llm_insights_data = None
            enhanced_quality_assessment = None
            tokens_used = 0
            blended_scorer = BlendedScorer()
            
            if use_llm_insights and self.enhanced_llm_client.is_available():
                dataset_context = {
                    "filename": metadata["filename"],
                    "total_rows": metadata["rows"], 
                    "columns": [col for col in df.columns.tolist()],
                    "size_mb": metadata["size_mb"],
                    "detected_problem_type": detected_problem_type,
                    "detected_target_column": detected_target_column
                }
                
                # Enhanced quality assessment with smart sampling
                enhanced_quality_assessment = await self.enhanced_llm_client.assess_data_quality_issues(
                    df=df,
                    profile_data=profile_data,
                    quality_score=quality_score_data,
                    detected_issues=detected_issues,
                    dataset_context=dataset_context
                )
                
                if enhanced_quality_assessment and "error" not in enhanced_quality_assessment:
                    tokens_used += enhanced_quality_assessment.get("tokens_used", 0)
                    
                    # Validate LLM findings against actual data
                    validator = LLMFindingValidator(df)
                    validated_assessment = validator.validate_all_findings(enhanced_quality_assessment)
                    
                    # Extract validation info
                    validation_summary = validated_assessment.get("validation_summary", {})
                    llm_reliability = validation_summary.get("llm_reliability", "MEDIUM")
                    llm_confidence = validated_assessment.get("confidence_level", "MEDIUM")
                    
                    # Use blended scoring instead of direct override
                    if "adjusted_quality_score" in validated_assessment:
                        deterministic_score = quality_score_data["ml_readiness_score"]
                        llm_suggested_score = validated_assessment["adjusted_quality_score"]
                        
                        # Calculate blended score
                        final_score, scoring_metadata = blended_scorer.calculate_final_score(
                            deterministic_score=deterministic_score,
                            llm_score=llm_suggested_score,
                            llm_confidence=llm_confidence,
                            validation_reliability=llm_reliability
                        )
                        
                        # Update quality score with blended result
                        quality_score_data["original_ml_readiness_score"] = deterministic_score
                        quality_score_data["llm_suggested_score"] = llm_suggested_score
                        quality_score_data["ml_readiness_score"] = final_score
                        quality_score_data["ml_readiness_grade"] = blended_scorer.calculate_grade_from_score(final_score)
                        quality_score_data["scoring_method"] = "blended"
                        quality_score_data["scoring_metadata"] = scoring_metadata
                        quality_score_data["validation_summary"] = validation_summary
                        
                        # Store enhanced assessment for insights
                        enhanced_quality_assessment = validated_assessment
                
                # Fallback to original LLM insights if enhanced analysis fails
                if not enhanced_quality_assessment or "error" in enhanced_quality_assessment:
                    dataset_info = {
                        "filename": metadata["filename"],
                        "rows": metadata["rows"], 
                        "columns": metadata["columns"],
                        "size_mb": metadata["size_mb"],
                        "detected_problem_type": detected_problem_type,
                        "detected_target_column": detected_target_column
                    }
                    
                    llm_insights_raw = await self.llm_client.generate_insights(
                        profile_data, quality_score_data, model_recommendations,
                        detected_issues, preprocessing_recommendations, dataset_info
                    )
                    
                    if llm_insights_raw:
                        tokens_used = llm_insights_raw.get("tokens_used", 0)
                        llm_insights_data = self._format_llm_insights(llm_insights_raw)
                else:
                    # Format enhanced insights
                    llm_insights_data = self._format_enhanced_llm_insights(enhanced_quality_assessment)
            
            # Step 9: Compile final response
            processing_time = time.time() - start_time
            
            response = AnalyzeDatasetResponse(
                analysis_id=analysis_id,
                timestamp=datetime.now(),
                dataset_info=DatasetInfo(
                    filename=metadata["filename"],
                    rows=metadata["rows"],
                    columns=metadata["columns"],
                    size_mb=round(metadata["size_mb"], 2),
                    detected_problem_type=detected_problem_type,
                    detected_target_column=detected_target_column
                ),
                quality_score=self._format_quality_score(quality_score_data),
                data_profile=self._format_data_profile(profile_data),
                detected_issues=self._format_detected_issues(detected_issues),
                model_recommendations=self._format_model_recommendations(model_recommendations),
                preprocessing_recommendations=self._format_preprocessing_recommendations(preprocessing_recommendations),
                llm_insights=llm_insights_data,
                metadata=Metadata(
                    processing_time_seconds=round(processing_time, 2),
                    llm_tokens_used=tokens_used if tokens_used > 0 else None,
                    api_version="v1"
                )
            )
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    def _generate_detected_issues(self, profile_data: Dict[str, Any], quality_score_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate list of detected issues based on analysis"""
        
        issues = []
        
        # Missing values issue
        missing_data = profile_data.get("missing_values", {})
        overall_missing_pct = missing_data.get("overall_missing_percentage", 0)
        
        if overall_missing_pct > 5:
            severity = "high" if overall_missing_pct > 20 else "medium"
            issues.append({
                "severity": severity,
                "category": "missing_values",
                "title": "Missing Values Detected",
                "description": f"{overall_missing_pct:.1f}% of data points are missing",
                "impact": "May reduce model performance and introduce bias",
                "affected_columns": [col["column"] for col in missing_data.get("columns_with_missing", [])],
                "recommendation": "Apply imputation, remove rows, or use models that handle missing values"
            })
        
        # Duplicate values issue  
        duplicates = profile_data.get("duplicates", {})
        duplicate_pct = duplicates.get("duplicate_percentage", 0)
        
        if duplicate_pct > 5:
            severity = "medium" if duplicate_pct > 10 else "low"
            issues.append({
                "severity": severity,
                "category": "duplicates",
                "title": "Duplicate Rows Found",
                "description": f"{duplicate_pct:.1f}% of rows are duplicates",
                "impact": "May lead to overfitting and biased performance estimates",
                "affected_columns": ["All columns"],
                "recommendation": "Remove duplicate rows before training"
            })
        
        # Class imbalance issue
        target_analysis = profile_data.get("target_analysis", {})
        if "class_balance" in target_analysis:
            class_balance = target_analysis["class_balance"]
            if not class_balance.get("balanced", True):
                imbalance_ratio = class_balance.get("imbalance_ratio", 1)
                severity = "high" if imbalance_ratio > 10 else "medium"
                issues.append({
                    "severity": severity,
                    "category": "class_imbalance",
                    "title": "Class Imbalance Detected",
                    "description": f"Target variable shows {imbalance_ratio:.1f}:1 imbalance ratio",
                    "impact": "Model may be biased toward majority class",
                    "affected_columns": [target_analysis.get("column_name", "target")],
                    "recommendation": "Use SMOTE, class weights, or stratified sampling"
                })
        
        # High correlation issue
        correlations = profile_data.get("correlations", {})
        high_corr_pairs = correlations.get("high_correlation_pairs", [])
        
        if len(high_corr_pairs) > 0:
            issues.append({
                "severity": "medium",
                "category": "multicollinearity",
                "title": "High Feature Correlations",
                "description": f"{len(high_corr_pairs)} feature pairs show high correlation (>0.8)",
                "impact": "May cause instability in linear models",
                "affected_columns": list(set([pair["feature1"] for pair in high_corr_pairs] + [pair["feature2"] for pair in high_corr_pairs])),
                "recommendation": "Consider removing highly correlated features or use regularization"
            })
        
        # High cardinality categorical features
        feature_analysis = profile_data.get("feature_analysis", {})
        categorical_features = feature_analysis.get("categorical_features", {})
        
        high_cardinality_cols = []
        for col, info in categorical_features.items():
            if info.get("cardinality") == "high":
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            issues.append({
                "severity": "low",
                "category": "high_cardinality",
                "title": "High Cardinality Categorical Features",
                "description": f"{len(high_cardinality_cols)} categorical columns have >50 unique values",
                "impact": "May create too many features after encoding",
                "affected_columns": high_cardinality_cols,
                "recommendation": "Consider grouping rare categories or using target encoding"
            })
        
        # Low variance features
        numerical_features = feature_analysis.get("numerical_features", {})
        low_variance_cols = []
        
        for col, info in numerical_features.items():
            if info.get("is_low_variance", False):
                low_variance_cols.append(col)
        
        if low_variance_cols:
            issues.append({
                "severity": "low", 
                "category": "low_variance",
                "title": "Low Variance Features",
                "description": f"{len(low_variance_cols)} features have very low variance",
                "impact": "Unlikely to be useful for prediction",
                "affected_columns": low_variance_cols,
                "recommendation": "Consider removing low variance features"
            })
        
        return issues
    
    def _generate_preprocessing_recommendations(self, profile_data: Dict[str, Any], problem_type: str, target_column: Optional[str]) -> List[Dict[str, Any]]:
        """Generate preprocessing recommendations"""
        
        recommendations = []
        priority = 1
        
        # Handle missing values
        missing_data = profile_data.get("missing_values", {})
        if missing_data.get("overall_missing_percentage", 0) > 0:
            methods = ["Simple Imputation", "KNN Imputation", "Iterative Imputation"]
            code_snippet = """from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent'
X_imputed = imputer.fit_transform(X_numerical)"""
            
            recommendations.append({
                "priority": priority,
                "step": "Handle Missing Values", 
                "category": "imputation",
                "methods": methods,
                "code_snippet": code_snippet
            })
            priority += 1
        
        # Handle class imbalance (for classification)
        if problem_type == "classification" and target_column:
            target_analysis = profile_data.get("target_analysis", {})
            if "class_balance" in target_analysis and not target_analysis["class_balance"].get("balanced", True):
                methods = ["SMOTE", "Random Oversampling", "Class Weights"]
                code_snippet = """from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)"""
                
                recommendations.append({
                    "priority": priority,
                    "step": "Address Class Imbalance",
                    "category": "sampling", 
                    "methods": methods,
                    "code_snippet": code_snippet
                })
                priority += 1
        
        # Encode categorical variables
        column_types = profile_data.get("column_types", {})
        if column_types.get("categorical", 0) > 0:
            methods = ["One-Hot Encoding", "Label Encoding", "Target Encoding"]
            code_snippet = """from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_categorical)"""
            
            recommendations.append({
                "priority": priority,
                "step": "Encode Categorical Variables",
                "category": "encoding",
                "methods": methods, 
                "code_snippet": code_snippet
            })
            priority += 1
        
        # Scale numerical features
        if column_types.get("numerical", 0) > 0:
            methods = ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"]
            code_snippet = """from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)"""
            
            recommendations.append({
                "priority": priority,
                "step": "Scale Numerical Features",
                "category": "scaling",
                "methods": methods,
                "code_snippet": code_snippet
            })
            priority += 1
        
        # Remove duplicates
        duplicates = profile_data.get("duplicates", {})
        if duplicates.get("duplicate_percentage", 0) > 5:
            methods = ["Drop Duplicates"]
            code_snippet = """# Remove duplicate rows
df_clean = df.drop_duplicates()"""
            
            recommendations.append({
                "priority": priority,
                "step": "Remove Duplicate Rows",
                "category": "cleaning",
                "methods": methods,
                "code_snippet": code_snippet
            })
            priority += 1
        
        return recommendations
    
    def _format_quality_score(self, quality_score_data: Dict[str, Any]) -> QualityScore:
        """Format quality score data"""
        
        breakdown = {}
        individual_scores = quality_score_data.get("individual_scores", {})
        
        for dimension, data in individual_scores.items():
            breakdown[dimension] = QualityScoreBreakdown(
                score=data["score"],
                grade=data["grade"],
                description=data["description"]
            )
        
        return QualityScore(
            overall=quality_score_data["ml_readiness_score"],
            grade=quality_score_data["ml_readiness_grade"],
            breakdown=breakdown
        )
    
    def _format_data_profile(self, profile_data: Dict[str, Any]) -> DataProfile:
        """Format data profile"""
        
        missing_values = profile_data.get("missing_values", {})
        duplicates = profile_data.get("duplicates", {})
        column_types = profile_data.get("column_types", {})
        basic_info = profile_data.get("basic_info", {})
        
        return DataProfile(
            missing_values_percentage=missing_values.get("overall_missing_percentage", 0),
            duplicate_rows=duplicates.get("duplicate_rows", 0),
            duplicate_percentage=duplicates.get("duplicate_percentage", 0),
            column_types={
                "numerical": column_types.get("numerical", 0),
                "categorical": column_types.get("categorical", 0),
                "datetime": column_types.get("datetime", 0),
                "boolean": column_types.get("boolean", 0)
            },
            memory_usage_mb=round(basic_info.get("memory_usage_mb", 0), 2),
            numeric_summary=NumericSummary(
                total_features=column_types.get("numerical", 0),
                high_correlation_pairs=len(profile_data.get("correlations", {}).get("high_correlation_pairs", [])),
                low_variance_features=sum(1 for feat in profile_data.get("feature_analysis", {}).get("numerical_features", {}).values() if feat.get("is_low_variance", False))
            ),
            categorical_summary=CategoricalSummary(
                total_features=column_types.get("categorical", 0),
                high_cardinality=sum(1 for feat in profile_data.get("feature_analysis", {}).get("categorical_features", {}).values() if feat.get("cardinality") == "high"),
                binary_features=sum(1 for feat in profile_data.get("feature_analysis", {}).get("categorical_features", {}).values() if feat.get("unique_values", 0) == 2)
            )
        )
    
    def _format_detected_issues(self, issues: List[Dict[str, Any]]) -> List[DetectedIssue]:
        """Format detected issues"""
        
        return [
            DetectedIssue(
                severity=issue["severity"],
                category=issue["category"],
                title=issue["title"],
                description=issue["description"],
                impact=issue["impact"],
                affected_columns=issue["affected_columns"],
                recommendation=issue["recommendation"]
            )
            for issue in issues
        ]
    
    def _format_model_recommendations(self, model_recs: List[Dict[str, Any]]) -> List[ModelRecommendation]:
        """Format model recommendations"""
        
        return [
            ModelRecommendation(
                rank=model["rank"],
                model_name=model["model_name"],
                confidence_score=model.get("rule_score", model.get("confidence_score", 50)),
                reasoning=model.get("rule_reasoning", model.get("reasoning", "")),
                pros=model["pros"],
                cons=model["cons"],
                recommended_hyperparameters=model["recommended_hyperparameters"],
                expected_performance=model["expected_performance"]
            )
            for model in model_recs
        ]
    
    def _format_preprocessing_recommendations(self, prep_recs: List[Dict[str, Any]]) -> List[PreprocessingRecommendation]:
        """Format preprocessing recommendations"""
        
        return [
            PreprocessingRecommendation(
                priority=rec["priority"],
                step=rec["step"],
                category=rec["category"],
                methods=rec["methods"],
                code_snippet=rec["code_snippet"]
            )
            for rec in prep_recs
        ]
    
    def _format_llm_insights(self, llm_data: Dict[str, Any]) -> Optional[LLMInsights]:
        """Format LLM insights"""
        
        if not llm_data or "error" in llm_data:
            return None
        
        action_items = []
        for item in llm_data.get("top_action_items", []):
            if isinstance(item, dict):
                # Standardize impact values to HIGH/MEDIUM/LOW
                impact = item.get("impact", "MEDIUM")
                if impact and isinstance(impact, str):
                    impact_upper = impact.upper()
                    if any(word in impact_upper for word in ["HIGH", "CRITICAL"]):
                        standardized_impact = "HIGH"
                    elif any(word in impact_upper for word in ["MEDIUM", "MODERATE"]):
                        standardized_impact = "MEDIUM"
                    elif any(word in impact_upper for word in ["LOW", "MINOR"]):
                        standardized_impact = "LOW"
                    else:
                        standardized_impact = "MEDIUM"  # Default
                else:
                    standardized_impact = "MEDIUM"
                
                # Extract effort value or set default
                effort = item.get("effort", "2-4 hours")
                
                action_items.append(ActionItem(
                    action=item.get("action", ""),
                    impact=standardized_impact,
                    effort=effort
                ))
        
        risk_data = llm_data.get("risk_assessment", {})
        risk_assessment = RiskAssessment(
            overfitting_risk=risk_data.get("overfitting_risk", "Unknown"),
            underfitting_risk=risk_data.get("underfitting_risk", "Unknown"),
            data_leakage_risk=risk_data.get("data_leakage_risk", "Unknown"),
            curse_of_dimensionality=risk_data.get("curse_of_dimensionality", "Unknown")
        )
        
        return LLMInsights(
            executive_summary=llm_data.get("executive_summary", ""),
            detailed_analysis=llm_data.get("detailed_analysis", ""),
            top_action_items=action_items,
            risk_assessment=risk_assessment,
            complete_preprocessing_pipeline=llm_data.get("complete_preprocessing_pipeline", "")
        )
    
    def _format_enhanced_llm_insights(self, enhanced_data: Dict[str, Any]) -> Optional[LLMInsights]:
        """Format enhanced LLM insights from quality assessment"""
        
        if not enhanced_data or "error" in enhanced_data:
            return None
        
        # Convert blocking issues to action items with standardized impact/effort
        action_items = []
        blocking_issues = enhanced_data.get("blocking_issues", [])
        
        for issue in blocking_issues[:5]:  # Top 5 issues
            # Standardize severity to impact format
            severity = issue.get("severity", "MEDIUM")
            if severity in ["CRITICAL", "HIGH"]:
                standardized_impact = "HIGH"
                effort = "4-6 hours"
            elif severity == "MEDIUM":
                standardized_impact = "MEDIUM" 
                effort = "2-4 hours"
            elif severity == "LOW":
                standardized_impact = "LOW"
                effort = "1-2 hours"
            else:
                standardized_impact = "MEDIUM"
                effort = "2-4 hours"
            
            action_items.append(ActionItem(
                action=issue.get("description", "Address data quality issue"),
                impact=standardized_impact,
                effort=effort
            ))
        
        # Add recommended fixes as action items
        recommended_fixes = enhanced_data.get("recommended_fixes", [])
        for fix in recommended_fixes[:3]:  # Top 3 fixes
            # Determine effort based on issue type
            issue_type = fix.get('issue_type', 'unknown')
            if issue_type in ['mixed_data_types', 'semantic_nulls']:
                effort = "4-8 hours"
                impact = "HIGH"
            elif issue_type in ['format_inconsistencies', 'number_format_mix']:
                effort = "2-4 hours"
                impact = "MEDIUM"
            else:
                effort = "1-3 hours"
                impact = "MEDIUM"
                
            action_items.append(ActionItem(
                action=fix.get("fix_description", "Apply recommended fix"),
                impact=impact,
                effort=effort
            ))
        
        # Extract contextual insights for risk assessment
        contextual = enhanced_data.get("contextual_insights", {})
        risk_assessment = RiskAssessment(
            overfitting_risk=contextual.get("preprocessing_complexity", "Unknown"),
            underfitting_risk="Medium",  # Default
            data_leakage_risk="Low",     # Default
            curse_of_dimensionality=contextual.get("preprocessing_complexity", "Unknown")
        )
        
        # Create executive summary from severity explanation
        severity_explanation = enhanced_data.get("severity_explanation", "")
        ml_readiness = enhanced_data.get("ml_readiness_assessment", "Unknown")
        exec_summary = f"Quality Score Adjusted: {severity_explanation} Dataset is currently: {ml_readiness}"
        
        # Create detailed analysis
        biggest_red_flags = enhanced_data.get("biggest_red_flags", [])
        red_flags_text = ", ".join(biggest_red_flags[:3]) if biggest_red_flags else "No major red flags"
        
        detailed_analysis = f"""Enhanced AI Analysis Results:
        
Original quality assessment was too generous. After analyzing sample problematic rows, the adjusted quality score better reflects ML readiness.

Key Issues Identified: {red_flags_text}

Semantic Problems Found: {contextual.get('semantic_issues_found', 'None detected')}

Business Logic Issues: {contextual.get('business_logic_violations', 'None detected')}

Preprocessing Complexity: {contextual.get('preprocessing_complexity', 'Unknown')}

Cleanup Effort Required: {enhanced_data.get('cleanup_effort_hours', 'Unknown')} hours

ML Readiness Assessment: {ml_readiness}"""
        
        # Generate preprocessing pipeline from recommended fixes
        pipeline_parts = []
        for fix in recommended_fixes[:5]:
            if "python_code" in fix:
                pipeline_parts.append(f"# {fix.get('fix_description', 'Fix')}")
                pipeline_parts.append(fix.get("python_code", ""))
                pipeline_parts.append("")
        
        pipeline = "\n".join(pipeline_parts) if pipeline_parts else "# No specific fixes recommended"
        
        return LLMInsights(
            executive_summary=exec_summary,
            detailed_analysis=detailed_analysis,
            top_action_items=action_items[:8],  # Limit to 8 items
            risk_assessment=risk_assessment,
            complete_preprocessing_pipeline=pipeline
        )
    

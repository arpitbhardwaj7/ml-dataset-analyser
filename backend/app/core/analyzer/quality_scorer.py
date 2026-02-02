"""
ML Dataset Quality Scorer - Deterministic 5-Dimension Assessment
Calculates independent quality scores across 5 dimensions with clear heuristics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from app.core.analyzer.baseline_model_trainer import BaselineModelTrainer
from app.core.analyzer.leakage_detector import LeakageDetector
from app.core.analyzer.consistency_checker import ConsistencyChecker

class QualityScorer:
    """
    Calculates 5 independent quality scores using deterministic heuristics:
    1. Data Quantity (20% weight)
    2. Data Completeness (20% weight) 
    3. Data Consistency & Validity (20% weight)
    4. Signal-to-Noise Ratio (25% weight)
    5. Target Suitability (15% weight)
    """
    
    def __init__(self, df: pd.DataFrame, profile_data: Dict[str, Any], target_column: Optional[str] = None):
        self.df = df
        self.profile_data = profile_data
        self.target_column = target_column
        self.rows, self.cols = df.shape
        
        # Initialize helper components
        self.baseline_trainer = BaselineModelTrainer()
        self.leakage_detector = LeakageDetector()
        self.consistency_checker = ConsistencyChecker()
    
    def calculate_quality_scores(self) -> Dict[str, Any]:
        """
        Calculate all 5 independent quality scores plus weighted ML readiness score
        
        Returns:
            Dict with individual scores, ML readiness score, and explanations
        """
        
        # Calculate each dimension independently
        quantity_result = self._calculate_data_quantity_score()
        completeness_result = self._calculate_data_completeness_score()
        consistency_result = self._calculate_data_consistency_score()
        signal_result = self._calculate_signal_to_noise_score()
        target_result = self._calculate_target_suitability_score()
        
        # Calculate weighted ML readiness score
        weights = {
            "data_quantity": 0.20,
            "data_completeness": 0.20,
            "data_consistency_validity": 0.20,
            "signal_to_noise_ratio": 0.25,
            "target_suitability": 0.15
        }
        
        ml_readiness_score = (
            quantity_result["score"] * weights["data_quantity"] +
            completeness_result["score"] * weights["data_completeness"] +
            consistency_result["score"] * weights["data_consistency_validity"] +
            signal_result["score"] * weights["signal_to_noise_ratio"] +
            target_result["score"] * weights["target_suitability"]
        )
        
        return {
            "individual_scores": {
                "data_quantity": quantity_result,
                "data_completeness": completeness_result,
                "data_consistency_validity": consistency_result,
                "signal_to_noise_ratio": signal_result,
                "target_suitability": target_result
            },
            "ml_readiness_score": ml_readiness_score,
            "ml_readiness_grade": self._score_to_grade(ml_readiness_score),
            "weights_used": weights,
            "calculation_method": "deterministic_heuristics"
        }
    
    def _calculate_data_quantity_score(self) -> Dict[str, Any]:
        """
        Score based on samples-per-feature ratio and dataset size adequacy
        Rules of thumb:
        - ≥100 samples per feature: 90-100 points
        - 50-100 samples per feature: 70-89 points  
        - 20-50 samples per feature: 50-69 points
        - <20 samples per feature: <50 points
        """
        
        samples_per_feature = self.rows / max(1, self.cols - 1)  # Exclude target from feature count
        base_score = 50  # Starting score
        
        # Primary scoring based on samples-per-feature ratio
        if samples_per_feature >= 100:
            base_score = 95
            adequacy = "excellent"
        elif samples_per_feature >= 50:
            base_score = 82
            adequacy = "good"
        elif samples_per_feature >= 20:
            base_score = 65
            adequacy = "acceptable"
        elif samples_per_feature >= 10:
            base_score = 45
            adequacy = "marginal"
        else:
            base_score = 25
            adequacy = "insufficient"
        
        # Apply penalties for specific issues
        penalties = []
        penalty_total = 0
        
        # Small dataset penalties
        if self.rows < 100:
            penalty = 20
            penalties.append(f"Very small dataset ({self.rows} rows) - high overfitting risk")
            penalty_total += penalty
        elif self.rows < 500:
            penalty = 10
            penalties.append(f"Small dataset ({self.rows} rows) - moderate overfitting risk")
            penalty_total += penalty
        
        # Class imbalance penalties (for classification)
        target_analysis = self.profile_data.get("target_analysis", {})
        if target_analysis.get("problem_type") == "classification" and "class_balance" in target_analysis:
            class_balance = target_analysis["class_balance"]
            if not class_balance.get("balanced", True):
                imbalance_ratio = class_balance.get("imbalance_ratio", 1)
                
                if imbalance_ratio > 20:
                    penalty = 15
                    penalties.append(f"Severe class imbalance ({imbalance_ratio:.1f}:1)")
                    penalty_total += penalty
                elif imbalance_ratio > 10:
                    penalty = 10
                    penalties.append(f"High class imbalance ({imbalance_ratio:.1f}:1)")
                    penalty_total += penalty
                elif imbalance_ratio > 5:
                    penalty = 5
                    penalties.append(f"Moderate class imbalance ({imbalance_ratio:.1f}:1)")
                    penalty_total += penalty
        
        # High dimensionality penalties
        if samples_per_feature < 10:
            penalty = 15
            penalties.append("High curse of dimensionality risk (features > samples/10)")
            penalty_total += penalty
        elif samples_per_feature < 20:
            penalty = 8
            penalties.append("Moderate curse of dimensionality risk")
            penalty_total += penalty
        
        final_score = max(0, min(100, base_score - penalty_total))
        
        return {
            "score": final_score,
            "grade": self._score_to_grade(final_score),
            "samples_per_feature": round(samples_per_feature, 1),
            "adequacy_level": adequacy,
            "penalties_applied": penalties,
            "penalty_total": penalty_total,
            "calculation": f"Base: {base_score}, Penalties: -{penalty_total}, Final: {final_score}",
            "description": self._get_quantity_description(final_score, samples_per_feature)
        }
    
    def _calculate_data_completeness_score(self) -> Dict[str, Any]:
        """
        Score based on missing values and data completeness
        Penalties:
        - Missing values: up to 40 points penalty
        - Duplicate rows: up to 20 points penalty
        """
        
        base_score = 100
        penalties = []
        penalty_total = 0
        
        # Missing values penalty
        missing_data = self.profile_data.get("missing_values", {})
        overall_missing_pct = missing_data.get("overall_missing_percentage", 0)
        
        if overall_missing_pct > 0:
            # Progressive penalty: 2 points per % missing, max 40 points
            missing_penalty = min(overall_missing_pct * 2, 40)
            penalties.append(f"{overall_missing_pct:.1f}% missing values")
            penalty_total += missing_penalty
        
        # Columns with excessive missing values
        columns_with_missing = missing_data.get("columns_with_missing", [])
        high_missing_columns = [col for col in columns_with_missing 
                               if col["missing_percentage"] > 40]
        
        if high_missing_columns:
            extra_penalty = len(high_missing_columns) * 5
            penalties.append(f"{len(high_missing_columns)} columns with >40% missing")
            penalty_total += extra_penalty
        
        # Duplicate rows penalty
        duplicates = self.profile_data.get("duplicates", {})
        duplicate_pct = duplicates.get("duplicate_percentage", 0)
        
        if duplicate_pct > 0:
            # Progressive penalty: 1.5 points per % duplicated, max 20 points
            duplicate_penalty = min(duplicate_pct * 1.5, 20)
            penalties.append(f"{duplicate_pct:.1f}% duplicate rows")
            penalty_total += duplicate_penalty
        
        # Structural completeness issues
        if self.profile_data.get("basic_info", {}).get("shape", [0, 0])[0] == 0:
            penalties.append("Empty dataset")
            penalty_total += 50
        
        final_score = max(0, base_score - penalty_total)
        
        return {
            "score": final_score,
            "grade": self._score_to_grade(final_score),
            "missing_percentage": overall_missing_pct,
            "duplicate_percentage": duplicate_pct,
            "high_missing_columns": len(high_missing_columns),
            "penalties_applied": penalties,
            "penalty_total": round(penalty_total, 1),
            "calculation": f"Base: {base_score}, Penalties: -{penalty_total:.1f}, Final: {final_score}",
            "description": self._get_completeness_description(final_score, overall_missing_pct, duplicate_pct)
        }
    
    def _calculate_data_consistency_score(self) -> Dict[str, Any]:
        """
        Score based on data type consistency, outliers, and data quality issues
        Uses consistency checker for detailed analysis
        """
        
        base_score = 100
        penalties = []
        penalty_total = 0
        
        # Run comprehensive consistency analysis
        consistency_results = self.consistency_checker.check_consistency(self.df)
        
        # Use the consistency checker's calculated score as base
        consistency_score = consistency_results.get("consistency_score", 100)
        consistency_penalty = 100 - consistency_score
        
        if consistency_penalty > 0:
            penalties.append(f"Data consistency issues detected")
            penalty_total += consistency_penalty
        
        # Additional penalties for outliers (from profile data)
        outliers = self.profile_data.get("outliers", {})
        high_outlier_columns = []
        
        for col, outlier_info in outliers.items():
            outlier_pct = outlier_info.get("outlier_percentage", 0)
            if outlier_pct > 15:  # Only penalize excessive outliers
                high_outlier_columns.append(col)
        
        if high_outlier_columns:
            outlier_penalty = min(len(high_outlier_columns) * 5, 20)
            penalties.append(f"{len(high_outlier_columns)} columns with excessive outliers (>15%)")
            penalty_total += outlier_penalty
        
        # High correlation penalty (multicollinearity)
        correlations = self.profile_data.get("correlations", {})
        high_corr_pairs = correlations.get("high_correlation_pairs", [])
        
        if len(high_corr_pairs) > 5:
            corr_penalty = min(len(high_corr_pairs), 15)
            penalties.append(f"{len(high_corr_pairs)} highly correlated feature pairs")
            penalty_total += corr_penalty
        elif len(high_corr_pairs) > 0:
            corr_penalty = len(high_corr_pairs) * 2
            penalties.append(f"{len(high_corr_pairs)} highly correlated feature pairs")
            penalty_total += corr_penalty
        
        final_score = max(0, base_score - penalty_total)
        
        # Include detailed consistency findings
        consistency_details = {
            "mixed_type_issues": len(consistency_results.get("mixed_type_columns", [])),
            "categorical_inconsistencies": len(consistency_results.get("inconsistent_categorical_columns", [])),
            "string_noise_issues": len(consistency_results.get("string_noise_columns", [])),
            "high_outlier_columns": len(high_outlier_columns),
            "high_correlation_pairs": len(high_corr_pairs)
        }
        
        return {
            "score": final_score,
            "grade": self._score_to_grade(final_score),
            "consistency_details": consistency_details,
            "penalties_applied": penalties,
            "penalty_total": round(penalty_total, 1),
            "calculation": f"Base: {base_score}, Penalties: -{penalty_total:.1f}, Final: {final_score}",
            "description": self._get_consistency_description(final_score, consistency_details)
        }
    
    def _calculate_signal_to_noise_score(self) -> Dict[str, Any]:
        """
        Score based on feature-target relationships using baseline model performance
        Uses actual ML model to assess signal strength
        """
        
        if not self.target_column:
            # Without target, use feature variance and correlations
            return self._calculate_unsupervised_signal_score()
        
        # Determine problem type
        target_analysis = self.profile_data.get("target_analysis", {})
        problem_type = target_analysis.get("problem_type", "classification")
        
        # Get baseline model assessment
        signal_assessment = self.baseline_trainer.assess_signal_quality(
            self.df, self.target_column, problem_type
        )
        
        # Convert assessment to score
        base_score = self.baseline_trainer.convert_signal_to_score(signal_assessment)
        
        penalties = []
        penalty_total = 0
        
        # Additional penalties based on feature analysis
        feature_analysis = self.profile_data.get("feature_analysis", {})
        
        # Low variance features penalty
        numerical_features = feature_analysis.get("numerical_features", {})
        low_variance_count = sum(1 for feat in numerical_features.values() 
                               if feat.get("is_low_variance", False))
        
        if low_variance_count > 0:
            variance_penalty = min(low_variance_count * 3, 15)
            penalties.append(f"{low_variance_count} low variance features")
            penalty_total += variance_penalty
        
        # Near-zero mutual information penalty
        feature_importance = self.profile_data.get("feature_importance", {})
        if isinstance(feature_importance, dict) and "error" not in feature_importance:
            importance_values = list(feature_importance.values())
            if importance_values:
                max_importance = max(importance_values)
                if max_importance < 0.05:  # Very weak signal
                    penalty = 20
                    penalties.append("Very weak feature-target relationships detected")
                    penalty_total += penalty
                elif max_importance < 0.1:  # Weak signal
                    penalty = 10
                    penalties.append("Weak feature-target relationships detected")
                    penalty_total += penalty
        
        final_score = max(0, min(100, base_score - penalty_total))
        
        return {
            "score": final_score,
            "grade": self._score_to_grade(final_score),
            "baseline_performance": signal_assessment.get("baseline_score", 0),
            "signal_quality": signal_assessment.get("signal_quality", "unknown"),
            "signal_strength": signal_assessment.get("signal_strength", "unknown"),
            "problem_type": problem_type,
            "low_variance_features": low_variance_count,
            "max_feature_importance": max(feature_importance.values()) if isinstance(feature_importance, dict) and feature_importance else 0,
            "penalties_applied": penalties,
            "penalty_total": penalty_total,
            "calculation": f"Baseline: {base_score}, Penalties: -{penalty_total}, Final: {final_score}",
            "description": signal_assessment.get("interpretation", "Signal assessment unavailable")
        }
    
    def _calculate_unsupervised_signal_score(self) -> Dict[str, Any]:
        """Calculate signal score when no target is available"""
        
        base_score = 60  # Default score without target
        penalties = []
        penalty_total = 0
        
        # Use correlations between features as proxy for signal
        correlations = self.profile_data.get("correlations", {})
        max_correlation = correlations.get("max_correlation", 0)
        
        if max_correlation > 0.8:
            base_score = 75
            signal_strength = "strong inter-feature relationships"
        elif max_correlation > 0.6:
            base_score = 68
            signal_strength = "moderate inter-feature relationships"
        elif max_correlation > 0.3:
            base_score = 55
            signal_strength = "weak inter-feature relationships"
        else:
            base_score = 40
            signal_strength = "very weak inter-feature relationships"
        
        # Low variance penalty
        feature_analysis = self.profile_data.get("feature_analysis", {})
        numerical_features = feature_analysis.get("numerical_features", {})
        low_variance_count = sum(1 for feat in numerical_features.values() 
                               if feat.get("is_low_variance", False))
        
        if low_variance_count > 0:
            variance_penalty = low_variance_count * 5
            penalties.append(f"{low_variance_count} low variance features")
            penalty_total += variance_penalty
        
        final_score = max(0, base_score - penalty_total)
        
        return {
            "score": final_score,
            "grade": self._score_to_grade(final_score),
            "max_correlation": max_correlation,
            "signal_strength": signal_strength,
            "low_variance_features": low_variance_count,
            "penalties_applied": penalties,
            "penalty_total": penalty_total,
            "calculation": f"Base: {base_score}, Penalties: -{penalty_total}, Final: {final_score}",
            "description": f"No target column available - using {signal_strength} as signal proxy"
        }
    
    def _calculate_target_suitability_score(self) -> Dict[str, Any]:
        """
        Score based on target variable quality and leakage detection
        """
        
        if not self.target_column:
            return {
                "score": 50,  # Neutral score when no target
                "grade": "C",
                "description": "No target column specified - cannot assess target suitability",
                "target_available": False
            }
        
        base_score = 90  # Start high for target suitability
        penalties = []
        penalty_total = 0
        
        # Target missing values penalty
        target_analysis = self.profile_data.get("target_analysis", {})
        target_missing = target_analysis.get("missing_values", 0)
        
        if target_missing > 0:
            missing_penalty = min(target_missing * 3, 30)  # Harsh penalty for missing targets
            penalties.append(f"{target_missing} missing values in target")
            penalty_total += missing_penalty
        
        # Class balance penalty (for classification)
        if target_analysis.get("problem_type") == "classification":
            class_balance = target_analysis.get("class_balance", {})
            if not class_balance.get("balanced", True):
                imbalance_ratio = class_balance.get("imbalance_ratio", 1)
                
                if imbalance_ratio > 50:
                    penalty = 25
                    penalties.append(f"Extreme class imbalance ({imbalance_ratio:.1f}:1)")
                elif imbalance_ratio > 20:
                    penalty = 20
                    penalties.append(f"Severe class imbalance ({imbalance_ratio:.1f}:1)")
                elif imbalance_ratio > 10:
                    penalty = 15
                    penalties.append(f"High class imbalance ({imbalance_ratio:.1f}:1)")
                elif imbalance_ratio > 5:
                    penalty = 8
                    penalties.append(f"Moderate class imbalance ({imbalance_ratio:.1f}:1)")
                else:
                    penalty = 0  # Acceptable imbalance
                
                penalty_total += penalty
        
        # Data leakage detection penalty
        leakage_results = self.leakage_detector.detect_leakage(self.df, self.target_column)
        leakage_risk = leakage_results.get("leakage_risk_level", "low")
        
        if leakage_risk == "high":
            penalty = 30
            penalties.append("High data leakage risk detected")
            penalty_total += penalty
        elif leakage_risk == "medium":
            penalty = 15
            penalties.append("Medium data leakage risk detected")
            penalty_total += penalty
        elif leakage_risk == "low" and len(leakage_results.get("potential_leakage_columns", [])) > 0:
            penalty = 5
            penalties.append("Low data leakage risk detected")
            penalty_total += penalty
        
        # Target clarity penalty
        unique_values = target_analysis.get("unique_values", 0)
        total_rows = len(self.df)
        
        if target_analysis.get("problem_type") == "classification":
            # Too many classes for classification
            if unique_values > total_rows * 0.5:
                penalty = 25
                penalties.append("Too many unique target values for classification")
                penalty_total += penalty
            elif unique_values > min(100, total_rows * 0.1):
                penalty = 15
                penalties.append("High cardinality target variable")
                penalty_total += penalty
        else:
            # Regression - check for sufficient variance
            target_stats = target_analysis.get("statistics", {})
            if "std" in target_stats and target_stats["std"] == 0:
                penalty = 30
                penalties.append("Target variable has zero variance")
                penalty_total += penalty
        
        final_score = max(0, base_score - penalty_total)
        
        return {
            "score": final_score,
            "grade": self._score_to_grade(final_score),
            "target_available": True,
            "problem_type": target_analysis.get("problem_type", "unknown"),
            "target_missing_values": target_missing,
            "leakage_risk": leakage_risk,
            "leakage_columns_found": len(leakage_results.get("potential_leakage_columns", [])),
            "class_balance_info": target_analysis.get("class_balance", {}),
            "penalties_applied": penalties,
            "penalty_total": penalty_total,
            "calculation": f"Base: {base_score}, Penalties: -{penalty_total}, Final: {final_score}",
            "description": self._get_target_description(final_score, target_analysis, leakage_risk)
        }
    
    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _get_quantity_description(self, score: float, samples_per_feature: float) -> str:
        """Generate description for data quantity score"""
        if score >= 90:
            return f"Excellent sample size - {samples_per_feature:.1f} samples per feature"
        elif score >= 80:
            return f"Good sample size - {samples_per_feature:.1f} samples per feature"
        elif score >= 70:
            return f"Acceptable sample size - {samples_per_feature:.1f} samples per feature"
        elif score >= 60:
            return f"Marginal sample size - risk of overfitting with {samples_per_feature:.1f} samples per feature"
        else:
            return f"Insufficient sample size - high overfitting risk with {samples_per_feature:.1f} samples per feature"
    
    def _get_completeness_description(self, score: float, missing_pct: float, duplicate_pct: float) -> str:
        """Generate description for completeness score"""
        if score >= 90:
            return "Excellent data completeness with minimal missing values or duplicates"
        elif score >= 80:
            return "Good data completeness with minor missing value issues"
        elif score >= 70:
            return f"Acceptable completeness - {missing_pct:.1f}% missing, {duplicate_pct:.1f}% duplicates"
        elif score >= 60:
            return f"Moderate completeness issues - {missing_pct:.1f}% missing, {duplicate_pct:.1f}% duplicates"
        else:
            return f"Significant completeness issues - {missing_pct:.1f}% missing, {duplicate_pct:.1f}% duplicates"
    
    def _get_consistency_description(self, score: float, details: Dict[str, int]) -> str:
        """Generate description for consistency score"""
        total_issues = sum(details.values())
        
        if score >= 90:
            return "Excellent data consistency with no major issues"
        elif score >= 80:
            return "Good data consistency with minor issues"
        elif score >= 70:
            return f"Acceptable consistency - {total_issues} issues detected"
        elif score >= 60:
            return f"Moderate consistency issues - {total_issues} problems need attention"
        else:
            return f"Significant consistency issues - {total_issues} problems will impact ML training"
    
    def _get_target_description(self, score: float, target_analysis: Dict, leakage_risk: str) -> str:
        """Generate description for target suitability score"""
        if not target_analysis:
            return "No target column available for assessment"
        
        problem_type = target_analysis.get("problem_type", "unknown")
        
        if score >= 90:
            return f"Excellent target variable for {problem_type} - well-balanced and suitable"
        elif score >= 80:
            return f"Good target variable for {problem_type} - minor issues detected"
        elif score >= 70:
            return f"Acceptable target for {problem_type} - some preprocessing may be needed"
        elif score >= 60:
            return f"Target has moderate issues - {leakage_risk} leakage risk, balance concerns"
        else:
            return f"Target has significant issues - {leakage_risk} leakage risk, major problems detected"
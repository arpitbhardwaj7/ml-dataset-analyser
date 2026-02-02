import pandas as pd
import numpy as np
from typing import Dict, List, Any

class ModelRecommender:
    def __init__(self, df: pd.DataFrame, profile_data: Dict[str, Any], problem_type: str, target_column: str = None):
        self.df = df
        self.profile_data = profile_data
        self.problem_type = problem_type
        self.target_column = target_column
        self.rows, self.cols = df.shape
    
    def get_model_recommendations(self) -> List[Dict[str, Any]]:
        """Get top 3 model recommendations using rule-based fingerprinting"""
        
        if self.problem_type == "classification":
            candidates = self._get_classification_candidates()
        else:
            candidates = self._get_regression_candidates()
        
        # Apply rule-based ranking (deterministic)
        ranked_models = self._rank_models_by_rules(candidates)
        
        # Add rank to top 3 and return
        for i, model in enumerate(ranked_models[:3]):
            model["rank"] = i + 1
        
        return ranked_models[:3]
    
    def _get_classification_candidates(self) -> List[Dict[str, Any]]:
        """Get classification model candidates"""
        
        return [
            {
                "model_name": "XGBoost Classifier",
                "reasoning": "Excellent for tabular data with mixed types, handles missing values, provides feature importance",
                "pros": [
                    "Excellent performance on tabular data",
                    "Handles missing values automatically",
                    "Built-in feature importance",
                    "Robust to outliers",
                    "Handles imbalanced datasets well"
                ],
                "cons": [
                    "Requires hyperparameter tuning",
                    "Less interpretable than simple models",
                    "Can overfit with small datasets"
                ],
                "recommended_hyperparameters": {
                    "max_depth": "3-6",
                    "learning_rate": "0.01-0.1",
                    "n_estimators": "100-1000",
                    "scale_pos_weight": "Adjust for class imbalance"
                },
                "expected_performance": "85-95% accuracy",
                "dataset_size_preference": "medium_to_large",
                "handles_missing": True,
                "handles_categorical": True,
                "handles_imbalance": True,
                "interpretable": False
            },
            {
                "model_name": "Random Forest Classifier",
                "reasoning": "Robust ensemble method, good baseline performance, interpretable feature importance",
                "pros": [
                    "Robust to overfitting",
                    "Handles non-linear relationships",
                    "Provides feature importance",
                    "Works well with mixed data types",
                    "Good default performance"
                ],
                "cons": [
                    "Can be slower than other methods",
                    "May struggle with very high cardinality features",
                    "Less accurate than gradient boosting on some datasets"
                ],
                "recommended_hyperparameters": {
                    "n_estimators": "100-500",
                    "max_depth": "10-20",
                    "min_samples_split": "2-10",
                    "class_weight": "balanced for imbalanced data"
                },
                "expected_performance": "80-90% accuracy",
                "dataset_size_preference": "any",
                "handles_missing": False,
                "handles_categorical": True,
                "handles_imbalance": True,
                "interpretable": True
            },
            {
                "model_name": "LightGBM Classifier",
                "reasoning": "Fast gradient boosting, memory efficient, good for large datasets",
                "pros": [
                    "Very fast training and prediction",
                    "Memory efficient",
                    "Good accuracy",
                    "Handles categorical features well",
                    "Built-in early stopping"
                ],
                "cons": [
                    "Sensitive to overfitting on small datasets",
                    "Less robust than XGBoost",
                    "Requires careful hyperparameter tuning"
                ],
                "recommended_hyperparameters": {
                    "num_leaves": "31-100",
                    "learning_rate": "0.05-0.1",
                    "n_estimators": "100-1000",
                    "class_weight": "balanced"
                },
                "expected_performance": "82-93% accuracy",
                "dataset_size_preference": "large",
                "handles_missing": True,
                "handles_categorical": True,
                "handles_imbalance": True,
                "interpretable": False
            },
            {
                "model_name": "Logistic Regression",
                "reasoning": "Simple, interpretable, fast, good baseline for linear relationships",
                "pros": [
                    "Highly interpretable",
                    "Fast training and prediction",
                    "Probabilistic outputs",
                    "No hyperparameter tuning required",
                    "Works well with regularization"
                ],
                "cons": [
                    "Assumes linear relationships",
                    "Sensitive to outliers",
                    "Requires feature scaling",
                    "May underfit complex patterns"
                ],
                "recommended_hyperparameters": {
                    "C": "0.01-100",
                    "penalty": "l1, l2, or elasticnet",
                    "class_weight": "balanced",
                    "max_iter": "1000-5000"
                },
                "expected_performance": "70-85% accuracy",
                "dataset_size_preference": "any",
                "handles_missing": False,
                "handles_categorical": False,
                "handles_imbalance": True,
                "interpretable": True
            },
            {
                "model_name": "Support Vector Machine (SVM)",
                "reasoning": "Effective for high-dimensional data, works well with small to medium datasets",
                "pros": [
                    "Effective in high dimensions",
                    "Memory efficient",
                    "Versatile (different kernels)",
                    "Works well with small datasets"
                ],
                "cons": [
                    "Slow on large datasets",
                    "Sensitive to feature scaling",
                    "No probabilistic output",
                    "Difficult to interpret"
                ],
                "recommended_hyperparameters": {
                    "C": "0.1-100",
                    "kernel": "rbf or linear",
                    "gamma": "scale or auto",
                    "class_weight": "balanced"
                },
                "expected_performance": "75-88% accuracy",
                "dataset_size_preference": "small_to_medium",
                "handles_missing": False,
                "handles_categorical": False,
                "handles_imbalance": True,
                "interpretable": False
            }
        ]
    
    def _get_regression_candidates(self) -> List[Dict[str, Any]]:
        """Get regression model candidates"""
        
        return [
            {
                "model_name": "XGBoost Regressor",
                "reasoning": "Excellent for tabular data, handles missing values, provides feature importance",
                "pros": [
                    "Excellent performance on tabular data",
                    "Handles missing values automatically",
                    "Built-in feature importance",
                    "Robust to outliers",
                    "Handles non-linear patterns well"
                ],
                "cons": [
                    "Requires hyperparameter tuning",
                    "Less interpretable than linear models",
                    "Can overfit with small datasets"
                ],
                "recommended_hyperparameters": {
                    "max_depth": "3-6",
                    "learning_rate": "0.01-0.1",
                    "n_estimators": "100-1000",
                    "reg_alpha": "0.1-1.0",
                    "reg_lambda": "0.1-1.0"
                },
                "expected_performance": "R² = 0.75-0.95",
                "dataset_size_preference": "medium_to_large",
                "handles_missing": True,
                "handles_categorical": True,
                "handles_imbalance": True,
                "interpretable": False
            },
            {
                "model_name": "Random Forest Regressor",
                "reasoning": "Robust ensemble method, good baseline performance, handles non-linear relationships",
                "pros": [
                    "Robust to overfitting",
                    "Handles non-linear relationships",
                    "Provides feature importance",
                    "Works well with mixed data types",
                    "Good default performance"
                ],
                "cons": [
                    "Can be slower than other methods",
                    "May struggle with extrapolation",
                    "Less accurate than gradient boosting"
                ],
                "recommended_hyperparameters": {
                    "n_estimators": "100-500",
                    "max_depth": "10-20",
                    "min_samples_split": "2-10",
                    "min_samples_leaf": "1-5"
                },
                "expected_performance": "R² = 0.70-0.90",
                "dataset_size_preference": "any",
                "handles_missing": False,
                "handles_categorical": True,
                "handles_imbalance": True,
                "interpretable": True
            },
            {
                "model_name": "Linear Regression",
                "reasoning": "Simple, interpretable, fast, good baseline for linear relationships",
                "pros": [
                    "Highly interpretable",
                    "Fast training and prediction",
                    "No hyperparameters to tune",
                    "Works well with regularization",
                    "Provides coefficient significance"
                ],
                "cons": [
                    "Assumes linear relationships",
                    "Sensitive to outliers",
                    "Requires feature scaling",
                    "May underfit complex patterns"
                ],
                "recommended_hyperparameters": {
                    "fit_intercept": "True",
                    "normalize": "Consider StandardScaler",
                    "regularization": "Ridge or Lasso if needed"
                },
                "expected_performance": "R² = 0.60-0.85",
                "dataset_size_preference": "any",
                "handles_missing": False,
                "handles_categorical": False,
                "handles_imbalance": True,
                "interpretable": True
            },
            {
                "model_name": "LightGBM Regressor",
                "reasoning": "Fast gradient boosting, memory efficient, good for large datasets",
                "pros": [
                    "Very fast training and prediction",
                    "Memory efficient",
                    "Good accuracy",
                    "Handles categorical features well",
                    "Built-in early stopping"
                ],
                "cons": [
                    "Sensitive to overfitting on small datasets",
                    "Less robust than XGBoost",
                    "Requires careful hyperparameter tuning"
                ],
                "recommended_hyperparameters": {
                    "num_leaves": "31-100",
                    "learning_rate": "0.05-0.1",
                    "n_estimators": "100-1000",
                    "reg_alpha": "0.1-1.0"
                },
                "expected_performance": "R² = 0.72-0.92",
                "dataset_size_preference": "large",
                "handles_missing": True,
                "handles_categorical": True,
                "handles_imbalance": True,
                "interpretable": False
            },
            {
                "model_name": "ElasticNet Regression",
                "reasoning": "Regularized linear model, good for feature selection, handles multicollinearity",
                "pros": [
                    "Automatic feature selection",
                    "Handles multicollinearity",
                    "Interpretable coefficients",
                    "Prevents overfitting",
                    "Fast training"
                ],
                "cons": [
                    "Assumes linear relationships",
                    "Requires hyperparameter tuning",
                    "May underfit complex patterns",
                    "Sensitive to feature scaling"
                ],
                "recommended_hyperparameters": {
                    "alpha": "0.01-10.0",
                    "l1_ratio": "0.1-0.9",
                    "max_iter": "1000-5000"
                },
                "expected_performance": "R² = 0.65-0.80",
                "dataset_size_preference": "any",
                "handles_missing": False,
                "handles_categorical": False,
                "handles_imbalance": True,
                "interpretable": True
            }
        ]
    
    def _score_model(self, model: Dict[str, Any]) -> int:
        """Score a model based on dataset characteristics"""
        
        score = 50  # Base score
        
        # Dataset size factor
        if self.rows < 100:
            size_category = "small"
        elif self.rows < 10000:
            size_category = "medium"
        else:
            size_category = "large"
        
        # Dataset size preference scoring
        size_pref = model.get("dataset_size_preference", "any")
        if size_pref == "any":
            score += 10
        elif size_pref == "small_to_medium" and size_category in ["small", "medium"]:
            score += 15
        elif size_pref == "medium_to_large" and size_category in ["medium", "large"]:
            score += 15
        elif size_pref == "large" and size_category == "large":
            score += 20
        elif size_pref == "small" and size_category == "small":
            score += 15
        else:
            score -= 5  # Penalty for size mismatch
        
        # Missing values handling
        missing_pct = self.profile_data.get("missing_values", {}).get("overall_missing_percentage", 0)
        if missing_pct > 5:
            if model.get("handles_missing", False):
                score += 15
            else:
                score -= 10
        
        # Categorical variables handling
        categorical_cols = len(self.profile_data.get("column_types", {}).get("categorical_columns", []))
        if categorical_cols > 0:
            if model.get("handles_categorical", False):
                score += 10
            else:
                score -= 5
        
        # Class imbalance handling (for classification)
        if self.problem_type == "classification" and self.target_column:
            target_analysis = self.profile_data.get("target_analysis", {})
            if "class_balance" in target_analysis:
                if not target_analysis["class_balance"].get("balanced", True):
                    if model.get("handles_imbalance", False):
                        score += 15
                    else:
                        score -= 10
        
        # High dimensionality
        feature_to_sample_ratio = self.cols / self.rows
        if feature_to_sample_ratio > 0.1:  # High dimensionality
            if "SVM" in model["model_name"] or "ElasticNet" in model["model_name"]:
                score += 10
            elif "XGBoost" in model["model_name"] or "Random Forest" in model["model_name"]:
                score += 5
            else:
                score -= 5
        
        # Interpretability bonus for specific scenarios
        if model.get("interpretable", False):
            if self.rows < 1000:  # Small datasets often benefit from interpretable models
                score += 5
        
        # Outliers handling
        outliers = self.profile_data.get("outliers", {})
        high_outlier_cols = sum(1 for col_info in outliers.values() 
                               if col_info.get("outlier_percentage", 0) > 5)
        
        if high_outlier_cols > 0:
            if "XGBoost" in model["model_name"] or "Random Forest" in model["model_name"]:
                score += 10
            elif "Linear" in model["model_name"] or "Logistic" in model["model_name"]:
                score -= 10
        
        # Feature importance availability
        feature_importance = self.profile_data.get("feature_importance", {})
        if isinstance(feature_importance, dict) and "error" not in feature_importance:
            importance_values = list(feature_importance.values())
            if importance_values:
                max_importance = max(importance_values)
                if max_importance < 0.1:  # Weak feature relationships
                    if model.get("interpretable", False):
                        score += 5  # Favor interpretable models when signal is weak
        
        return max(0, min(100, score))
    
    def _rank_models_by_rules(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply deterministic rule-based ranking using dataset fingerprinting"""
        
        # Calculate dataset characteristics
        dataset_fingerprint = self._create_dataset_fingerprint()
        
        # Score each model based on rules
        for model in candidates:
            model["rule_score"] = self._calculate_rule_score(model, dataset_fingerprint)
            model["rule_reasoning"] = self._generate_rule_reasoning(model, dataset_fingerprint)
        
        # Sort by rule score descending
        return sorted(candidates, key=lambda x: x["rule_score"], reverse=True)
    
    def _create_dataset_fingerprint(self) -> Dict[str, Any]:
        """Create a comprehensive dataset fingerprint for rule-based matching"""
        
        # Basic characteristics
        rows, cols = self.rows, self.cols
        samples_per_feature = rows / max(1, cols)
        
        # Data quality metrics
        missing_info = self.profile_data.get("missing_values", {})
        missing_pct = missing_info.get("overall_missing_percentage", 0)
        
        # Column types
        column_types = self.profile_data.get("column_types", {})
        categorical_count = column_types.get("categorical", 0)
        numerical_count = column_types.get("numerical", 0)
        
        # Target analysis (if available)
        target_info = self.profile_data.get("target_analysis", {})
        is_imbalanced = False
        if "class_balance" in target_info:
            is_imbalanced = not target_info["class_balance"].get("balanced", True)
            imbalance_ratio = target_info["class_balance"].get("imbalance_ratio", 1)
        else:
            imbalance_ratio = 1
        
        # Complexity indicators
        outlier_info = self.profile_data.get("outliers", {})
        high_outlier_features = sum(1 for info in outlier_info.values() 
                                   if info.get("outlier_percentage", 0) > 10)
        
        correlations = self.profile_data.get("correlations", {})
        high_corr_pairs = len(correlations.get("high_correlation_pairs", []))
        
        # Signal strength (if available)
        feature_importance = self.profile_data.get("feature_importance", {})
        if isinstance(feature_importance, dict) and "error" not in feature_importance:
            importance_values = list(feature_importance.values())
            max_importance = max(importance_values) if importance_values else 0
        else:
            max_importance = 0
        
        return {
            # Size characteristics
            "size_category": "tiny" if rows < 100 else "small" if rows < 1000 else "medium" if rows < 10000 else "large",
            "samples_per_feature": samples_per_feature,
            "high_dimensional": cols > rows,
            "curse_of_dimensionality": cols > rows * 0.1,
            
            # Data quality
            "missing_data_level": "none" if missing_pct == 0 else "low" if missing_pct < 5 else "medium" if missing_pct < 20 else "high",
            "has_missing": missing_pct > 0,
            "has_outliers": high_outlier_features > 0,
            "high_multicollinearity": high_corr_pairs > 5,
            
            # Feature characteristics
            "has_categorical": categorical_count > 0,
            "categorical_ratio": categorical_count / max(1, categorical_count + numerical_count),
            "mixed_types": categorical_count > 0 and numerical_count > 0,
            
            # Target characteristics (classification only)
            "is_imbalanced": is_imbalanced,
            "severe_imbalance": imbalance_ratio > 10,
            
            # Signal characteristics
            "weak_signal": max_importance < 0.1,
            "strong_signal": max_importance > 0.5,
            
            # Derived flags
            "needs_robust_model": high_outlier_features > 2 or missing_pct > 10,
            "needs_interpretable": samples_per_feature < 20 or max_importance < 0.2,
            "needs_fast_model": rows > 50000,
            "preprocessing_heavy": categorical_count > 5 or missing_pct > 15
        }
    
    def _calculate_rule_score(self, model: Dict[str, Any], fingerprint: Dict[str, Any]) -> int:
        """Calculate rule-based score using dataset fingerprint matching"""
        
        score = 50  # Base score
        model_name = model["model_name"]
        
        # Rule 1: Dataset size matching
        size_category = fingerprint["size_category"]
        
        if "XGBoost" in model_name or "LightGBM" in model_name:
            if size_category in ["medium", "large"]:
                score += 20
            elif size_category == "small":
                score += 10
            else:  # tiny
                score -= 15  # Gradient boosting can overfit on tiny datasets
        
        elif "Random Forest" in model_name:
            if size_category in ["small", "medium", "large"]:
                score += 15  # RF is versatile
            else:  # tiny
                score += 5
        
        elif "Linear" in model_name or "Logistic" in model_name:
            if size_category in ["tiny", "small"]:
                score += 15
            elif fingerprint["high_dimensional"]:
                score += 20  # Linear models handle high dimensions well
            else:
                score += 5
        
        elif "SVM" in model_name:
            if size_category in ["tiny", "small"]:
                score += 15
            elif fingerprint["high_dimensional"]:
                score += 20
            else:
                score -= 10  # SVM doesn't scale well to large datasets
        
        elif "ElasticNet" in model_name:
            if fingerprint["high_dimensional"] or fingerprint["high_multicollinearity"]:
                score += 25  # ElasticNet excels with multicollinearity
            elif size_category in ["small", "medium"]:
                score += 10
            else:
                score += 5
        
        # Rule 2: Missing data handling
        if fingerprint["has_missing"]:
            if model.get("handles_missing", False):
                score += 15
            else:
                score -= 15
        
        # Rule 3: Categorical data handling
        if fingerprint["has_categorical"]:
            if model.get("handles_categorical", False):
                score += 10
                if fingerprint["categorical_ratio"] > 0.5:  # Mostly categorical
                    if "XGBoost" in model_name or "LightGBM" in model_name:
                        score += 10  # Excellent categorical handling
            else:
                penalty = 10 + int(fingerprint["categorical_ratio"] * 10)
                score -= penalty
        
        # Rule 4: Imbalanced data (classification only)
        if self.problem_type == "classification":
            if fingerprint["is_imbalanced"]:
                if model.get("handles_imbalance", False):
                    boost = 15 if fingerprint["severe_imbalance"] else 10
                    score += boost
                else:
                    penalty = 15 if fingerprint["severe_imbalance"] else 8
                    score -= penalty
        
        # Rule 5: Robustness requirements
        if fingerprint["needs_robust_model"]:
            if "XGBoost" in model_name or "Random Forest" in model_name:
                score += 15  # Tree-based models are robust
            elif "Linear" in model_name or "Logistic" in model_name:
                score -= 10  # Linear models sensitive to outliers
        
        # Rule 6: Interpretability requirements
        if fingerprint["needs_interpretable"]:
            if model.get("interpretable", False):
                score += 20
            else:
                score -= 10
        
        # Rule 7: High-dimensional data
        if fingerprint["high_dimensional"]:
            if "ElasticNet" in model_name or "SVM" in model_name:
                score += 25
            elif "Linear" in model_name or "Logistic" in model_name:
                score += 15
            else:
                score -= 5
        
        # Rule 8: Signal strength considerations
        if fingerprint["weak_signal"]:
            if model.get("interpretable", False):
                score += 10  # Favor interpretable models for weak signals
            if "Logistic" in model_name or "Linear" in model_name:
                score += 5  # Simple models for simple patterns
        
        if fingerprint["strong_signal"]:
            if "XGBoost" in model_name or "Random Forest" in model_name:
                score += 15  # Complex models can exploit strong signals
        
        # Rule 9: Speed requirements
        if fingerprint["needs_fast_model"]:
            if "LightGBM" in model_name:
                score += 20
            elif "Linear" in model_name or "Logistic" in model_name:
                score += 15
            elif "Random Forest" in model_name:
                score -= 5
            elif "SVM" in model_name:
                score -= 15
        
        # Rule 10: Preprocessing complexity
        if fingerprint["preprocessing_heavy"]:
            if model.get("handles_missing", False) and model.get("handles_categorical", False):
                score += 15
        
        return max(0, min(100, score))
    
    def _generate_rule_reasoning(self, model: Dict[str, Any], fingerprint: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the model recommendation"""
        
        reasons = []
        model_name = model["model_name"]
        
        # Size-based reasoning
        size_category = fingerprint["size_category"]
        if size_category == "tiny" and ("XGBoost" in model_name or "LightGBM" in model_name):
            reasons.append("⚠️  Dataset is very small - gradient boosting may overfit")
        elif size_category in ["medium", "large"] and ("XGBoost" in model_name or "LightGBM" in model_name):
            reasons.append("✅ Excellent choice for medium-large tabular datasets")
        elif size_category in ["tiny", "small"] and ("Linear" in model_name or "Logistic" in model_name):
            reasons.append("✅ Simple models work well with small datasets")
        
        # Missing data reasoning
        if fingerprint["has_missing"]:
            if model.get("handles_missing", False):
                reasons.append(f"✅ Handles missing data automatically ({fingerprint['missing_data_level']} level)")
            else:
                reasons.append(f"⚠️  Requires preprocessing for missing data ({fingerprint['missing_data_level']} level)")
        
        # Categorical data reasoning
        if fingerprint["has_categorical"]:
            if model.get("handles_categorical", False):
                reasons.append("✅ Excellent categorical feature handling")
            else:
                reasons.append("⚠️  Requires encoding for categorical features")
        
        # Imbalance reasoning
        if self.problem_type == "classification" and fingerprint["is_imbalanced"]:
            if model.get("handles_imbalance", False):
                severity = "severe" if fingerprint["severe_imbalance"] else "moderate"
                reasons.append(f"✅ Handles {severity} class imbalance well")
            else:
                reasons.append("⚠️  May struggle with imbalanced classes")
        
        # High-dimensional reasoning
        if fingerprint["high_dimensional"]:
            if "ElasticNet" in model_name or "SVM" in model_name:
                reasons.append("✅ Excellent for high-dimensional data")
            elif "XGBoost" in model_name or "Random Forest" in model_name:
                reasons.append("⚠️  May need feature selection for high dimensions")
        
        # Interpretability reasoning
        if fingerprint["needs_interpretable"]:
            if model.get("interpretable", False):
                reasons.append("✅ Provides interpretable results")
            else:
                reasons.append("⚠️  Black-box model - limited interpretability")
        
        # Signal strength reasoning
        if fingerprint["weak_signal"]:
            if model.get("interpretable", False):
                reasons.append("✅ Good choice when signal is weak")
        elif fingerprint["strong_signal"]:
            if "XGBoost" in model_name or "Random Forest" in model_name:
                reasons.append("✅ Can exploit strong feature-target relationships")
        
        # Default reasoning if no specific reasons
        if not reasons:
            reasons.append("✅ Solid baseline choice for this dataset type")
        
        return " | ".join(reasons[:4])  # Limit to top 4 reasons

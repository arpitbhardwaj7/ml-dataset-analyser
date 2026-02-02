"""
Baseline Model Trainer for Signal-to-Noise Quality Scoring
Provides quick baseline model performance assessment using RandomForest
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

class BaselineModelTrainer:
    """
    Trains quick baseline models to assess signal-to-noise ratio
    Uses shallow RandomForest with cross-validation for reliable estimates
    """
    
    def __init__(self):
        self.classification_model = RandomForestClassifier(
            n_estimators=50,      # Fast enough for CV
            max_depth=5,          # Shallow to avoid overfitting
            min_samples_split=10, # Conservative splits
            random_state=42,
            n_jobs=-1            # Parallel processing
        )
        
        self.regression_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
    
    def assess_signal_quality(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Assess signal-to-noise ratio using baseline model performance
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            problem_type: 'classification' or 'regression'
            
        Returns:
            Dict with baseline performance metrics and signal assessment
        """
        
        try:
            # Prepare features and target
            X, y, preprocessing_info = self._prepare_data(df, target_column)
            
            if X.shape[0] < 10:
                return {
                    "error": "Insufficient data for baseline model (need at least 10 samples)",
                    "baseline_score": 0,
                    "signal_quality": "insufficient_data",
                    "samples_used": X.shape[0]
                }
            
            # Get baseline performance
            baseline_score = self._get_baseline_performance(X, y, problem_type)
            
            # Assess signal quality based on performance
            signal_assessment = self._assess_signal_strength(baseline_score, problem_type)
            
            return {
                "baseline_score": round(baseline_score, 4),
                "signal_quality": signal_assessment["quality"],
                "signal_strength": signal_assessment["strength"],
                "interpretation": signal_assessment["interpretation"],
                "samples_used": X.shape[0],
                "features_used": X.shape[1],
                "preprocessing_applied": preprocessing_info,
                "problem_type": problem_type
            }
            
        except Exception as e:
            return {
                "error": f"Baseline model training failed: {str(e)}",
                "baseline_score": 0,
                "signal_quality": "assessment_failed",
                "samples_used": 0
            }
    
    def _prepare_data(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Prepare data for baseline model training"""
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        preprocessing_info = {
            "original_features": len(feature_cols),
            "categorical_encoded": 0,
            "missing_imputed": 0,
            "features_dropped": 0
        }
        
        # Handle missing values in target
        valid_target_mask = y.notna()
        X = X[valid_target_mask]
        y = y[valid_target_mask]
        
        if len(y) == 0:
            raise ValueError("No valid target values found")
        
        # Handle categorical features (encode)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if X[col].nunique() > 100:  # High cardinality - drop
                X = X.drop(columns=[col])
                preprocessing_info["features_dropped"] += 1
            else:
                # Label encode
                le = LabelEncoder()
                # Fill missing categorical with 'missing'
                X[col] = X[col].fillna('missing').astype(str)
                X[col] = le.fit_transform(X[col])
                preprocessing_info["categorical_encoded"] += 1
        
        # Handle missing values in numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
            preprocessing_info["missing_imputed"] = len(numerical_cols)
        
        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values
        
        # Encode target if categorical
        if y.dtype in ['object', 'category']:
            le_target = LabelEncoder()
            y_array = le_target.fit_transform(y_array.astype(str))
        
        preprocessing_info["final_features"] = X_array.shape[1]
        
        return X_array, y_array, preprocessing_info
    
    def _get_baseline_performance(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        problem_type: str
    ) -> float:
        """Get baseline model performance using cross-validation"""
        
        n_samples = len(y)
        
        # Determine CV folds based on sample size
        if n_samples < 30:
            cv_folds = 3
        elif n_samples < 100:
            cv_folds = 5
        else:
            cv_folds = 5  # Standard
        
        try:
            if problem_type == "classification":
                # Check if we have multiple classes
                unique_classes = len(np.unique(y))
                if unique_classes == 1:
                    return 0.0  # No signal - only one class
                
                # Use stratified CV for classification
                cv = StratifiedKFold(n_splits=min(cv_folds, unique_classes), shuffle=True, random_state=42)
                
                # Ensure we have enough samples per class for CV
                min_class_size = np.min(np.bincount(y))
                if min_class_size < cv_folds:
                    cv_folds = max(2, min_class_size)
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                scores = cross_val_score(
                    self.classification_model, X, y, 
                    cv=cv, scoring='accuracy', n_jobs=1  # Single job for stability
                )
                
            else:  # regression
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(
                    self.regression_model, X, y, 
                    cv=cv, scoring='r2', n_jobs=1
                )
            
            # Return mean score, handle negative R² scores
            mean_score = scores.mean()
            if problem_type == "regression" and mean_score < 0:
                return 0.0  # Negative R² indicates no predictive power
                
            return max(0.0, mean_score)  # Ensure non-negative
            
        except Exception as e:
            # If CV fails, return 0 (no signal detected)
            return 0.0
    
    def _assess_signal_strength(
        self, 
        baseline_score: float, 
        problem_type: str
    ) -> Dict[str, str]:
        """Assess signal strength based on baseline performance"""
        
        if problem_type == "classification":
            # Classification thresholds (accuracy-based)
            if baseline_score >= 0.85:
                return {
                    "quality": "excellent",
                    "strength": "very_strong",
                    "interpretation": "Excellent predictive signal - high accuracy achieved with simple model"
                }
            elif baseline_score >= 0.75:
                return {
                    "quality": "good",
                    "strength": "strong",
                    "interpretation": "Good predictive signal - clear patterns detected"
                }
            elif baseline_score >= 0.65:
                return {
                    "quality": "moderate",
                    "strength": "moderate",
                    "interpretation": "Moderate predictive signal - some useful patterns present"
                }
            elif baseline_score >= 0.55:
                return {
                    "quality": "weak",
                    "strength": "weak",
                    "interpretation": "Weak predictive signal - limited useful patterns"
                }
            else:
                return {
                    "quality": "poor",
                    "strength": "very_weak",
                    "interpretation": "Very weak signal - features may not be predictive of target"
                }
                
        else:  # regression
            # Regression thresholds (R²-based)
            if baseline_score >= 0.80:
                return {
                    "quality": "excellent",
                    "strength": "very_strong",
                    "interpretation": "Excellent predictive signal - high R² achieved with simple model"
                }
            elif baseline_score >= 0.65:
                return {
                    "quality": "good",
                    "strength": "strong", 
                    "interpretation": "Good predictive signal - clear linear/non-linear relationships"
                }
            elif baseline_score >= 0.50:
                return {
                    "quality": "moderate",
                    "strength": "moderate",
                    "interpretation": "Moderate predictive signal - some useful relationships present"
                }
            elif baseline_score >= 0.35:
                return {
                    "quality": "weak",
                    "strength": "weak",
                    "interpretation": "Weak predictive signal - limited predictive relationships"
                }
            else:
                return {
                    "quality": "poor",
                    "strength": "very_weak",
                    "interpretation": "Very weak signal - features may not predict target effectively"
                }
    
    def convert_signal_to_score(self, signal_assessment: Dict[str, Any]) -> int:
        """Convert signal assessment to quality score (0-100)"""
        
        if signal_assessment.get("error"):
            return 30  # Default low score for errors
        
        quality = signal_assessment.get("quality", "poor")
        
        quality_to_score = {
            "excellent": 90,
            "good": 75,
            "moderate": 60,
            "weak": 45,
            "poor": 30,
            "insufficient_data": 20,
            "assessment_failed": 25
        }
        
        return quality_to_score.get(quality, 30)
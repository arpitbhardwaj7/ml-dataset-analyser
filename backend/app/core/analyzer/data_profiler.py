import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import warnings

# Import new analyzers
from app.core.analyzer.leakage_detector import LeakageDetector
from app.core.analyzer.consistency_checker import ConsistencyChecker

warnings.filterwarnings('ignore')

class DataProfiler:
    def __init__(self, df: pd.DataFrame, target_column: str = None, custom_leakage_patterns: List[str] = None):
        self.df = df.copy()
        self.target_column = target_column
        self.target_series = df[target_column] if target_column else None
        
        # Initialize enhanced analyzers
        self.leakage_detector = LeakageDetector(custom_patterns=custom_leakage_patterns)
        self.consistency_checker = ConsistencyChecker()
        
    def generate_profile(self) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        
        profile = {
            "basic_info": self._get_basic_info(),
            "missing_values": self._analyze_missing_values(),
            "duplicates": self._analyze_duplicates(),
            "column_types": self._analyze_column_types(),
            "statistical_summary": self._get_statistical_summary(),
            "correlations": self._analyze_correlations(),
            "outliers": self._detect_outliers(),
            "feature_analysis": self._analyze_features()
        }
        
        # Add enhanced analyses
        profile["data_leakage"] = self._analyze_data_leakage()
        profile["data_consistency"] = self._analyze_data_consistency()
        profile["quality_insights"] = self._generate_quality_insights()
        
        if self.target_column:
            profile["target_analysis"] = self._analyze_target()
            profile["feature_importance"] = self._calculate_feature_importance()
        
        return profile
    
    def _get_basic_info(self) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "shape": self.df.shape,
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / (1024 * 1024),
            "column_names": self.df.columns.tolist()
        }
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values pattern"""
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        columns_with_missing = []
        for col in missing_counts.index:
            if missing_counts[col] > 0:
                columns_with_missing.append({
                    "column": col,
                    "missing_count": int(missing_counts[col]),
                    "missing_percentage": round(float(missing_percentages[col]), 2)
                })
        
        return {
            "total_missing_values": int(missing_counts.sum()),
            "overall_missing_percentage": round(float((missing_counts.sum() / (len(self.df) * len(self.df.columns))) * 100), 2),
            "columns_with_missing": columns_with_missing,
            "complete_rows": int(len(self.df) - self.df.isnull().any(axis=1).sum())
        }
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate rows"""
        duplicate_rows = self.df.duplicated().sum()
        
        return {
            "duplicate_rows": int(duplicate_rows),
            "duplicate_percentage": round(float((duplicate_rows / len(self.df)) * 100), 2),
            "unique_rows": int(len(self.df) - duplicate_rows)
        }
    
    def _analyze_column_types(self) -> Dict[str, Any]:
        """Analyze data types and categorize columns"""
        numerical_cols = []
        categorical_cols = []
        datetime_cols = []
        boolean_cols = []
        
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                numerical_cols.append(col)
            elif self.df[col].dtype in ['object']:
                # Check if it's actually datetime
                try:
                    pd.to_datetime(self.df[col].dropna().head(10))
                    datetime_cols.append(col)
                except:
                    categorical_cols.append(col)
            elif self.df[col].dtype in ['bool', 'category']:
                boolean_cols.append(col)
            elif self.df[col].dtype.name.startswith('datetime'):
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)
        
        return {
            "numerical": len(numerical_cols),
            "categorical": len(categorical_cols),
            "datetime": len(datetime_cols),
            "boolean": len(boolean_cols),
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "boolean_columns": boolean_cols
        }
    
    def _get_statistical_summary(self) -> Dict[str, Any]:
        """Get statistical summary for numerical columns"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {"message": "No numerical columns found"}
        
        stats_summary = {}
        for col in numerical_cols:
            series = self.df[col].dropna()
            if len(series) > 0:
                stats_summary[col] = {
                    "count": int(len(series)),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "25%": float(series.quantile(0.25)),
                    "50%": float(series.quantile(0.50)),
                    "75%": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis())
                }
        
        return stats_summary
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between numerical features"""
        numerical_df = self.df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) < 2:
            return {"message": "Not enough numerical columns for correlation analysis"}
        
        corr_matrix = numerical_df.corr()
        
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": round(float(corr_val), 3)
                    })
        
        return {
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "high_correlation_pairs": high_corr_pairs,
            "max_correlation": float(abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]).max()) if len(corr_matrix) > 1 else 0.0
        }
    
    def _detect_outliers(self) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numerical_cols:
            series = self.df[col].dropna()
            if len(series) > 0:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                
                outlier_info[col] = {
                    "outlier_count": int(len(outliers)),
                    "outlier_percentage": round(float(len(outliers) / len(series) * 100), 2),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }
        
        return outlier_info
    
    def _analyze_features(self) -> Dict[str, Any]:
        """Analyze feature characteristics"""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Categorical feature analysis
        categorical_analysis = {}
        for col in categorical_cols:
            unique_values = self.df[col].nunique()
            most_frequent = self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else "N/A"
            
            categorical_analysis[col] = {
                "unique_values": int(unique_values),
                "cardinality": "high" if unique_values > 50 else "medium" if unique_values > 10 else "low",
                "most_frequent_value": str(most_frequent),
                "most_frequent_count": int(self.df[col].value_counts().iloc[0]) if len(self.df[col].value_counts()) > 0 else 0
            }
        
        # Numerical feature analysis
        numerical_analysis = {}
        for col in numerical_cols:
            series = self.df[col].dropna()
            if len(series) > 0:
                variance = series.var()
                numerical_analysis[col] = {
                    "variance": float(variance),
                    "is_low_variance": variance < 0.01,
                    "unique_values": int(series.nunique()),
                    "is_constant": series.nunique() == 1
                }
        
        return {
            "categorical_features": categorical_analysis,
            "numerical_features": numerical_analysis
        }
    
    def _analyze_target(self) -> Dict[str, Any]:
        """Analyze target variable"""
        if not self.target_column or self.target_series is None:
            return {}
        
        target_analysis = {
            "column_name": self.target_column,
            "data_type": str(self.target_series.dtype),
            "unique_values": int(self.target_series.nunique()),
            "missing_values": int(self.target_series.isnull().sum())
        }
        
        # Classification target analysis
        if self.target_series.dtype in ['object', 'category', 'bool'] or self.target_series.nunique() <= 20:
            value_counts = self.target_series.value_counts()
            target_analysis.update({
                "problem_type": "classification",
                "class_distribution": value_counts.to_dict(),
                "class_balance": {
                    "balanced": (value_counts.max() / value_counts.sum()) < 0.6,
                    "imbalance_ratio": float(value_counts.max() / value_counts.min()) if value_counts.min() > 0 else float('inf'),
                    "majority_class": str(value_counts.index[0]),
                    "minority_class": str(value_counts.index[-1])
                }
            })
        else:
            # Regression target analysis
            target_analysis.update({
                "problem_type": "regression",
                "statistics": {
                    "mean": float(self.target_series.mean()),
                    "std": float(self.target_series.std()),
                    "min": float(self.target_series.min()),
                    "max": float(self.target_series.max()),
                    "skewness": float(self.target_series.skew()),
                    "kurtosis": float(self.target_series.kurtosis())
                }
            })
        
        return target_analysis
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance using mutual information"""
        if not self.target_column or self.target_series is None:
            return {}
        
        try:
            # Prepare features
            feature_cols = [col for col in self.df.columns if col != self.target_column]
            X = self.df[feature_cols].copy()
            y = self.target_series.copy()
            
            # Handle missing values
            X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).columns.tolist() else X.mode().iloc[0])
            y = y.fillna(y.mode().iloc[0] if len(y.mode()) > 0 else 0)
            
            # Encode categorical variables
            label_encoders = {}
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            
            # Encode target if categorical
            if y.dtype in ['object', 'category']:
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
                mutual_info_func = mutual_info_classif
            else:
                mutual_info_func = mutual_info_regression
            
            # Calculate mutual information
            mi_scores = mutual_info_func(X, y, random_state=42)
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, col in enumerate(feature_cols):
                feature_importance[col] = float(mi_scores[i])
            
            return feature_importance
        
        except Exception as e:
            return {"error": f"Could not calculate feature importance: {str(e)}"}
    
    def _analyze_data_leakage(self) -> Dict[str, Any]:
        """Analyze potential data leakage using LeakageDetector"""
        try:
            leakage_results = self.leakage_detector.detect_leakage(self.df, self.target_column)
            
            # Extract key information for profiler
            return {
                "leakage_risk_level": leakage_results.get("leakage_risk_level", "low"),
                "potential_leakage_columns": leakage_results.get("potential_leakage_columns", []),
                "high_risk_columns": [item["column"] for item in leakage_results.get("high_risk_columns", [])],
                "medium_risk_columns": [item["column"] for item in leakage_results.get("medium_risk_columns", [])],
                "leakage_categories_found": leakage_results.get("leakage_categories_found", []),
                "perfect_correlations": leakage_results.get("perfect_correlations", []),
                "recommendations": leakage_results.get("recommendations", [])[:3]  # Top 3 recommendations
            }
        except Exception as e:
            return {"error": f"Leakage analysis failed: {str(e)}"}
    
    def _analyze_data_consistency(self) -> Dict[str, Any]:
        """Analyze data consistency using ConsistencyChecker"""
        try:
            consistency_results = self.consistency_checker.check_consistency(self.df)
            
            # Extract key information for profiler
            return {
                "overall_consistency_score": consistency_results.get("consistency_score", 100),
                "total_issues": consistency_results.get("overall_consistency_issues", 0),
                "mixed_type_columns": [item["column"] for item in consistency_results.get("mixed_type_columns", [])],
                "inconsistent_categorical_columns": [item["column"] for item in consistency_results.get("inconsistent_categorical_columns", [])],
                "string_noise_columns": [item["column"] for item in consistency_results.get("string_noise_columns", [])],
                "unit_inconsistency_columns": [item["column"] for item in consistency_results.get("unit_inconsistency_columns", [])],
                "format_inconsistency_columns": [item["column"] for item in consistency_results.get("format_inconsistency_columns", [])],
                "top_recommendations": consistency_results.get("recommendations", [])[:3]  # Top 3 recommendations
            }
        except Exception as e:
            return {"error": f"Consistency analysis failed: {str(e)}"}
    
    def _generate_quality_insights(self) -> Dict[str, Any]:
        """Generate high-level quality insights and flags"""
        
        insights = {
            "dataset_size_category": self._categorize_dataset_size(),
            "complexity_indicators": self._assess_complexity(),
            "ml_readiness_flags": self._generate_ml_readiness_flags(),
            "recommended_preprocessing": self._suggest_preprocessing_steps(),
            "risk_factors": self._identify_risk_factors()
        }
        
        return insights
    
    def _categorize_dataset_size(self) -> Dict[str, Any]:
        """Categorize dataset size for ML purposes"""
        rows, cols = self.df.shape
        
        if rows < 100:
            size_category = "very_small"
            adequacy = "insufficient for most ML tasks"
        elif rows < 1000:
            size_category = "small" 
            adequacy = "suitable for simple models only"
        elif rows < 10000:
            size_category = "medium"
            adequacy = "good for most ML algorithms"
        elif rows < 100000:
            size_category = "large"
            adequacy = "excellent for complex models"
        else:
            size_category = "very_large"
            adequacy = "suitable for deep learning and complex models"
        
        samples_per_feature = rows / max(1, cols - 1) if self.target_column else rows / cols
        
        return {
            "category": size_category,
            "adequacy": adequacy,
            "rows": rows,
            "columns": cols,
            "samples_per_feature": round(samples_per_feature, 1)
        }
    
    def _assess_complexity(self) -> Dict[str, Any]:
        """Assess dataset complexity indicators"""
        
        # Count different types of complexity
        missing_pct = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100
        categorical_cols = len(self.df.select_dtypes(include=['object', 'category']).columns)
        numerical_cols = len(self.df.select_dtypes(include=[np.number]).columns)
        
        # High cardinality categorical features
        high_cardinality_count = 0
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            if self.df[col].nunique() > 50:
                high_cardinality_count += 1
        
        return {
            "missing_data_complexity": "high" if missing_pct > 20 else "medium" if missing_pct > 5 else "low",
            "feature_type_diversity": "high" if categorical_cols > 0 and numerical_cols > 0 else "medium",
            "high_cardinality_features": high_cardinality_count,
            "categorical_to_numerical_ratio": round(categorical_cols / max(1, numerical_cols), 2),
            "overall_complexity": self._determine_overall_complexity(missing_pct, categorical_cols, high_cardinality_count)
        }
    
    def _determine_overall_complexity(self, missing_pct: float, categorical_cols: int, high_cardinality_count: int) -> str:
        """Determine overall preprocessing complexity"""
        complexity_score = 0
        
        if missing_pct > 20:
            complexity_score += 3
        elif missing_pct > 5:
            complexity_score += 1
        
        if categorical_cols > 5:
            complexity_score += 2
        elif categorical_cols > 0:
            complexity_score += 1
        
        if high_cardinality_count > 0:
            complexity_score += high_cardinality_count
        
        if complexity_score >= 6:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _generate_ml_readiness_flags(self) -> List[str]:
        """Generate ML readiness warning flags"""
        flags = []
        
        rows, cols = self.df.shape
        
        # Size-based flags
        if rows < 100:
            flags.append("CRITICAL: Dataset too small for reliable ML models")
        elif rows / cols < 10:
            flags.append("WARNING: High risk of overfitting (low samples per feature)")
        
        # Missing data flags
        missing_pct = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100
        if missing_pct > 30:
            flags.append("CRITICAL: Excessive missing data (>30%)")
        elif missing_pct > 15:
            flags.append("WARNING: Significant missing data requires attention")
        
        # Target-specific flags
        if self.target_column:
            target_missing = self.df[self.target_column].isnull().sum()
            if target_missing > 0:
                flags.append(f"CRITICAL: Target variable has {target_missing} missing values")
        else:
            flags.append("INFO: No target variable specified - unsupervised learning")
        
        # Data type flags
        object_cols = self.df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            flags.append("INFO: Categorical encoding required for ML compatibility")
        
        return flags
    
    def _suggest_preprocessing_steps(self) -> List[str]:
        """Suggest preprocessing steps based on data profile"""
        steps = []
        
        # Missing values
        missing_pct = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100
        if missing_pct > 0:
            steps.append(f"Handle missing values ({missing_pct:.1f}% of data)")
        
        # Categorical encoding
        categorical_cols = len(self.df.select_dtypes(include=['object', 'category']).columns)
        if categorical_cols > 0:
            steps.append(f"Encode {categorical_cols} categorical features")
        
        # Feature scaling
        numerical_cols = len(self.df.select_dtypes(include=[np.number]).columns)
        if numerical_cols > 1:
            steps.append("Consider feature scaling for numerical columns")
        
        # Duplicates
        duplicate_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        if duplicate_pct > 0:
            steps.append(f"Remove duplicate rows ({duplicate_pct:.1f}%)")
        
        # Class imbalance (if classification target)
        if self.target_column and self.target_series is not None:
            if self.target_series.dtype in ['object', 'category', 'bool'] or self.target_series.nunique() <= 20:
                value_counts = self.target_series.value_counts()
                imbalance_ratio = value_counts.max() / value_counts.min() if value_counts.min() > 0 else float('inf')
                if imbalance_ratio > 5:
                    steps.append(f"Address class imbalance (ratio: {imbalance_ratio:.1f}:1)")
        
        return steps[:5]  # Top 5 most important steps
    
    def _identify_risk_factors(self) -> List[Dict[str, str]]:
        """Identify risk factors for ML model performance"""
        risks = []
        
        rows, cols = self.df.shape
        
        # Overfitting risks
        if rows / cols < 10:
            risks.append({
                "risk": "overfitting",
                "severity": "high",
                "description": f"Only {rows/cols:.1f} samples per feature - high overfitting risk"
            })
        elif rows / cols < 20:
            risks.append({
                "risk": "overfitting", 
                "severity": "medium",
                "description": f"{rows/cols:.1f} samples per feature - moderate overfitting risk"
            })
        
        # Data quality risks
        missing_pct = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100
        if missing_pct > 20:
            risks.append({
                "risk": "poor_data_quality",
                "severity": "high", 
                "description": f"High missing data ({missing_pct:.1f}%) may bias model predictions"
            })
        
        # Curse of dimensionality
        if cols > rows:
            risks.append({
                "risk": "curse_of_dimensionality",
                "severity": "critical",
                "description": "More features than samples - requires dimensionality reduction"
            })
        
        return risks

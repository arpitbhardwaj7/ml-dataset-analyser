"""
Data Leakage Detection for ML Quality Assessment
Detects potential data leakage issues using both built-in patterns and custom rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set
import re
from datetime import datetime

class LeakageDetector:
    """
    Detects potential data leakage in ML datasets
    Uses hybrid approach: built-in patterns + optional custom patterns
    """
    
    def __init__(self, custom_patterns: Optional[List[str]] = None):
        """
        Initialize leakage detector
        
        Args:
            custom_patterns: Optional list of custom column patterns to check
        """
        self.custom_patterns = custom_patterns or []
        
        # Built-in leakage patterns (always checked)
        self.built_in_patterns = self._get_built_in_patterns()
    
    def _get_built_in_patterns(self) -> Dict[str, List[str]]:
        """Define built-in leakage patterns by category"""
        
        return {
            # Date-related leakage (future information)
            "temporal_leakage": [
                "exit_date", "termination_date", "left_date", "churn_date",
                "departure_date", "resignation_date", "quit_date",
                "purchase_date", "conversion_date", "transaction_date",
                "completion_date", "success_date", "failure_date",
                "end_date", "finish_date", "final_date"
            ],
            
            # Outcome-related leakage (direct target information)
            "outcome_leakage": [
                "is_churned", "did_churn", "churned", "will_churn",
                "did_buy", "purchased", "converted", "will_buy",
                "is_fraud", "fraud_flag", "fraudulent",
                "outcome", "result", "target", "label",
                "success", "failure", "won", "lost",
                "approved", "rejected", "accepted", "denied"
            ],
            
            # ID-based leakage (unique identifiers that could encode target)
            "identifier_leakage": [
                "customer_id", "user_id", "account_id", "session_id",
                "transaction_id", "order_id", "ticket_id", "case_id",
                "employee_id", "patient_id", "student_id"
            ],
            
            # Aggregated future information
            "aggregation_leakage": [
                "total_purchases", "lifetime_value", "total_spent",
                "num_transactions", "avg_rating", "final_score",
                "cumulative_", "total_", "sum_", "count_", "avg_"
            ]
        }
    
    def detect_leakage(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect potential data leakage in the dataset
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (if known)
            
        Returns:
            Dictionary with leakage analysis results
        """
        
        leakage_results = {
            "potential_leakage_columns": [],
            "leakage_risk_level": "low",
            "leakage_categories_found": [],
            "high_risk_columns": [],
            "medium_risk_columns": [],
            "low_risk_columns": [],
            "perfect_correlations": [],
            "suspicious_patterns": [],
            "recommendations": []
        }
        
        # Check built-in patterns
        built_in_findings = self._check_built_in_patterns(df, target_column)
        
        # Check custom patterns
        custom_findings = self._check_custom_patterns(df, target_column)
        
        # Check statistical leakage (perfect correlations)
        statistical_findings = self._check_statistical_leakage(df, target_column)
        
        # Check temporal consistency
        temporal_findings = self._check_temporal_consistency(df)
        
        # Combine all findings
        all_findings = {
            **built_in_findings,
            "custom_pattern_matches": custom_findings,
            **statistical_findings,
            **temporal_findings
        }
        
        # Assess overall risk level
        leakage_results = self._assess_leakage_risk(all_findings, df)
        
        return leakage_results
    
    def _check_built_in_patterns(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str]
    ) -> Dict[str, Any]:
        """Check for built-in leakage patterns"""
        
        column_names = [col.lower().replace('_', '').replace(' ', '') for col in df.columns]
        original_columns = df.columns.tolist()
        
        findings = {
            "temporal_leakage_found": [],
            "outcome_leakage_found": [],
            "identifier_leakage_found": [],
            "aggregation_leakage_found": []
        }
        
        for category, patterns in self.built_in_patterns.items():
            category_key = f"{category}_found"
            
            for pattern in patterns:
                pattern_clean = pattern.lower().replace('_', '').replace(' ', '')
                
                # Exact matches
                exact_matches = [original_columns[i] for i, col in enumerate(column_names) 
                               if col == pattern_clean]
                
                # Partial matches (contains pattern)
                partial_matches = [original_columns[i] for i, col in enumerate(column_names) 
                                 if pattern_clean in col and col != pattern_clean]
                
                # Add matches with metadata
                for match in exact_matches:
                    findings[category_key].append({
                        "column": match,
                        "pattern": pattern,
                        "match_type": "exact",
                        "risk_level": self._assess_pattern_risk(pattern, category)
                    })
                
                for match in partial_matches:
                    findings[category_key].append({
                        "column": match,
                        "pattern": pattern,
                        "match_type": "partial",
                        "risk_level": self._assess_pattern_risk(pattern, category)
                    })
        
        return findings
    
    def _check_custom_patterns(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Check for custom leakage patterns"""
        
        if not self.custom_patterns:
            return []
        
        custom_matches = []
        column_names = df.columns.tolist()
        
        for pattern in self.custom_patterns:
            pattern_lower = pattern.lower()
            
            # Check each column against the custom pattern
            for col in column_names:
                col_lower = col.lower()
                
                if pattern_lower in col_lower:
                    custom_matches.append({
                        "column": col,
                        "custom_pattern": pattern,
                        "match_type": "custom_pattern",
                        "risk_level": "medium"  # Default risk for custom patterns
                    })
        
        return custom_matches
    
    def _check_statistical_leakage(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str]
    ) -> Dict[str, Any]:
        """Check for statistical indicators of leakage"""
        
        findings = {
            "perfect_correlations": [],
            "near_perfect_correlations": [],
            "suspicious_unique_ratios": []
        }
        
        if not target_column or target_column not in df.columns:
            return findings
        
        target_series = df[target_column]
        
        # Skip if target is not numeric for correlation analysis
        if target_series.dtype not in ['int64', 'int32', 'float64', 'float32']:
            # For categorical targets, check for perfect associations
            return self._check_categorical_target_leakage(df, target_column)
        
        # Check correlations with numeric features
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != target_column]
        
        for col in numeric_columns:
            try:
                correlation = df[col].corr(target_series)
                
                if pd.notna(correlation):
                    if abs(correlation) >= 0.99:
                        findings["perfect_correlations"].append({
                            "column": col,
                            "correlation": round(correlation, 4),
                            "risk_level": "high"
                        })
                    elif abs(correlation) >= 0.95:
                        findings["near_perfect_correlations"].append({
                            "column": col,
                            "correlation": round(correlation, 4),
                            "risk_level": "medium"
                        })
            except:
                continue
        
        # Check for suspicious unique value ratios
        for col in df.columns:
            if col != target_column:
                unique_ratio = df[col].nunique() / len(df)
                
                # Very high uniqueness might indicate leakage (like IDs)
                if unique_ratio > 0.95:
                    findings["suspicious_unique_ratios"].append({
                        "column": col,
                        "unique_ratio": round(unique_ratio, 3),
                        "unique_values": df[col].nunique(),
                        "risk_level": "medium"
                    })
        
        return findings
    
    def _check_categorical_target_leakage(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Dict[str, Any]:
        """Check for leakage with categorical target"""
        
        findings = {
            "perfect_correlations": [],
            "near_perfect_correlations": [],
            "suspicious_unique_ratios": []
        }
        
        target_series = df[target_column]
        
        # Check each column for perfect separation
        for col in df.columns:
            if col == target_column:
                continue
            
            try:
                # Create contingency table
                crosstab = pd.crosstab(df[col], target_series)
                
                # Check if any feature value perfectly predicts target
                perfect_predictors = []
                for feature_val in crosstab.index:
                    row = crosstab.loc[feature_val]
                    # If only one target class for this feature value
                    non_zero_classes = sum(row > 0)
                    if non_zero_classes == 1:
                        perfect_predictors.append(feature_val)
                
                if perfect_predictors:
                    # Calculate what % of samples have perfect prediction
                    perfect_samples = sum(df[col].isin(perfect_predictors))
                    perfect_ratio = perfect_samples / len(df)
                    
                    if perfect_ratio >= 0.5:  # 50%+ of samples perfectly predicted
                        findings["perfect_correlations"].append({
                            "column": col,
                            "perfect_prediction_ratio": round(perfect_ratio, 3),
                            "risk_level": "high"
                        })
                    elif perfect_ratio >= 0.2:  # 20%+ of samples perfectly predicted
                        findings["near_perfect_correlations"].append({
                            "column": col,
                            "perfect_prediction_ratio": round(perfect_ratio, 3),
                            "risk_level": "medium"
                        })
            except:
                continue
        
        return findings
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for temporal inconsistencies that indicate leakage"""
        
        findings = {
            "temporal_inconsistencies": [],
            "future_dates_detected": []
        }
        
        # Identify date columns
        date_columns = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                date_columns.append(col)
        
        current_date = datetime.now()
        
        # Check for future dates (might indicate data leakage)
        for col in date_columns:
            try:
                if df[col].dtype != 'datetime64[ns]':
                    # Try to convert to datetime
                    date_series = pd.to_datetime(df[col], errors='coerce')
                else:
                    date_series = df[col]
                
                # Check for future dates
                future_dates = date_series > current_date
                if future_dates.any():
                    future_count = future_dates.sum()
                    findings["future_dates_detected"].append({
                        "column": col,
                        "future_date_count": int(future_count),
                        "future_date_percentage": round(future_count / len(df) * 100, 2),
                        "risk_level": "medium"
                    })
            except:
                continue
        
        # Check for temporal ordering issues (hire_date > termination_date, etc.)
        date_pairs_to_check = [
            ("hire_date", "termination_date"),
            ("start_date", "end_date"),
            ("birth_date", "death_date"),
            ("created_date", "modified_date"),
            ("order_date", "ship_date")
        ]
        
        for early_col_pattern, late_col_pattern in date_pairs_to_check:
            early_cols = [col for col in date_columns if early_col_pattern in col.lower()]
            late_cols = [col for col in date_columns if late_col_pattern in col.lower()]
            
            for early_col in early_cols:
                for late_col in late_cols:
                    if early_col != late_col:
                        try:
                            early_series = pd.to_datetime(df[early_col], errors='coerce')
                            late_series = pd.to_datetime(df[late_col], errors='coerce')
                            
                            # Check for temporal inconsistencies
                            inconsistencies = early_series > late_series
                            valid_comparisons = early_series.notna() & late_series.notna()
                            
                            if valid_comparisons.any():
                                inconsistent_count = inconsistencies[valid_comparisons].sum()
                                if inconsistent_count > 0:
                                    findings["temporal_inconsistencies"].append({
                                        "early_column": early_col,
                                        "late_column": late_col,
                                        "inconsistent_count": int(inconsistent_count),
                                        "inconsistency_percentage": round(inconsistent_count / valid_comparisons.sum() * 100, 2),
                                        "risk_level": "low"
                                    })
                        except:
                            continue
        
        return findings
    
    def _assess_pattern_risk(self, pattern: str, category: str) -> str:
        """Assess risk level for a specific pattern"""
        
        high_risk_patterns = {
            "exit_date", "termination_date", "churn_date", "conversion_date",
            "is_churned", "did_churn", "outcome", "result", "target"
        }
        
        if pattern in high_risk_patterns:
            return "high"
        elif category == "temporal_leakage":
            return "high"  # Most temporal patterns are high risk
        elif category == "outcome_leakage":
            return "high"  # Direct outcome information
        elif category == "identifier_leakage":
            return "medium"  # IDs can encode information
        else:
            return "medium"
    
    def _assess_leakage_risk(
        self, 
        all_findings: Dict[str, Any], 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Assess overall leakage risk and generate recommendations"""
        
        # Collect all potential leakage columns
        potential_leakage = []
        high_risk = []
        medium_risk = []
        low_risk = []
        categories_found = set()
        
        # Process built-in pattern findings
        for category, findings in all_findings.items():
            if category.endswith("_found") and isinstance(findings, list):
                category_clean = category.replace("_found", "")
                
                for finding in findings:
                    column = finding["column"]
                    risk = finding["risk_level"]
                    
                    potential_leakage.append(column)
                    categories_found.add(category_clean)
                    
                    if risk == "high":
                        high_risk.append({
                            "column": column,
                            "reason": f"Matches {category_clean} pattern: {finding['pattern']}",
                            "category": category_clean
                        })
                    elif risk == "medium":
                        medium_risk.append({
                            "column": column,
                            "reason": f"Matches {category_clean} pattern: {finding['pattern']}",
                            "category": category_clean
                        })
                    else:
                        low_risk.append({
                            "column": column,
                            "reason": f"Matches {category_clean} pattern: {finding['pattern']}",
                            "category": category_clean
                        })
        
        # Process statistical findings
        for correlation in all_findings.get("perfect_correlations", []):
            column = correlation["column"]
            if column not in potential_leakage:
                potential_leakage.append(column)
                high_risk.append({
                    "column": column,
                    "reason": f"Perfect correlation with target (r={correlation.get('correlation', 'N/A')})",
                    "category": "statistical_leakage"
                })
                categories_found.add("statistical_leakage")
        
        for correlation in all_findings.get("near_perfect_correlations", []):
            column = correlation["column"]
            if column not in potential_leakage:
                potential_leakage.append(column)
                medium_risk.append({
                    "column": column,
                    "reason": f"Very high correlation with target (r={correlation.get('correlation', 'N/A')})",
                    "category": "statistical_leakage"
                })
                categories_found.add("statistical_leakage")
        
        # Determine overall risk level
        if len(high_risk) > 0:
            overall_risk = "high"
        elif len(medium_risk) > 0:
            overall_risk = "medium"
        elif len(low_risk) > 0:
            overall_risk = "low"
        else:
            overall_risk = "low"
        
        # Generate recommendations
        recommendations = self._generate_leakage_recommendations(
            high_risk, medium_risk, low_risk, overall_risk
        )
        
        return {
            "potential_leakage_columns": list(set(potential_leakage)),
            "leakage_risk_level": overall_risk,
            "leakage_categories_found": list(categories_found),
            "high_risk_columns": high_risk,
            "medium_risk_columns": medium_risk,
            "low_risk_columns": low_risk,
            "perfect_correlations": all_findings.get("perfect_correlations", []),
            "temporal_inconsistencies": all_findings.get("temporal_inconsistencies", []),
            "recommendations": recommendations
        }
    
    def _generate_leakage_recommendations(
        self,
        high_risk: List[Dict],
        medium_risk: List[Dict], 
        low_risk: List[Dict],
        overall_risk: str
    ) -> List[str]:
        """Generate actionable recommendations for handling leakage"""
        
        recommendations = []
        
        if overall_risk == "high":
            recommendations.append("CRITICAL: Remove high-risk columns before training to prevent data leakage")
            
        if len(high_risk) > 0:
            high_risk_cols = [item["column"] for item in high_risk]
            recommendations.append(f"Remove columns: {', '.join(high_risk_cols[:5])}")  # Limit display
            
        if len(medium_risk) > 0:
            recommendations.append("Investigate medium-risk columns for potential leakage before using")
            
        if overall_risk in ["high", "medium"]:
            recommendations.append("Use time-based train/test splits if temporal data is available")
            recommendations.append("Validate model performance on truly future data")
            
        if not recommendations:
            recommendations.append("No obvious data leakage detected - proceed with standard train/test split")
            
        return recommendations
"""
LLM Finding Validator - Validates LLM-detected issues against actual data
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple


class LLMFindingValidator:
    """Validates LLM-detected issues against actual data to prevent hallucinations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.semantic_null_patterns = [
            'n/a', 'na', 'null', 'none', 'nan', 'missing', '--', '?', 
            'not available', 'not applicable', 'unknown', 'nil', 'empty'
        ]
        self.text_number_patterns = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
            'hundred': 100, 'thousand': 1000, 'million': 1000000
        }
        
    def validate_all_findings(self, llm_findings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all LLM findings and add verification flags
        
        Returns updated findings with validation results
        """
        validated_findings = llm_findings.copy()
        
        if "blocking_issues" in llm_findings:
            validated_findings["blocking_issues"] = [
                self._validate_issue(issue) 
                for issue in llm_findings["blocking_issues"]
            ]
        
        # Calculate validation summary
        total_issues = len(validated_findings.get("blocking_issues", []))
        verified_issues = sum(
            1 for issue in validated_findings.get("blocking_issues", [])
            if issue.get("validation_status") == "VERIFIED"
        )
        partial_issues = sum(
            1 for issue in validated_findings.get("blocking_issues", [])
            if issue.get("validation_status") == "PARTIAL"
        )
        
        verification_rate = verified_issues / total_issues if total_issues > 0 else 1.0
        partial_rate = partial_issues / total_issues if total_issues > 0 else 0.0
        
        # Determine overall LLM reliability
        if verification_rate > 0.8:
            reliability = "HIGH"
        elif verification_rate + partial_rate > 0.6:
            reliability = "MEDIUM"
        else:
            reliability = "LOW"
        
        validated_findings["validation_summary"] = {
            "total_issues_claimed": total_issues,
            "issues_verified": verified_issues,
            "issues_partial": partial_issues,
            "verification_rate": round(verification_rate, 3),
            "llm_reliability": reliability,
            "confidence_adjustment": self._calculate_confidence_adjustment(verification_rate, partial_rate)
        }
        
        return validated_findings
    
    def _validate_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single issue claim"""
        issue_type = issue.get("issue_type", "").lower()
        affected_columns = issue.get("affected_columns", [])
        
        # Initialize validation result
        validation_result = {
            "validation_status": "UNVERIFIED",
            "verification_details": "",
            "evidence_found": [],
            "confidence_adjustment": 0.0
        }
        
        # Validate based on issue type
        if "mixed_data_type" in issue_type or "mixed_type" in issue_type:
            validation_result = self._validate_mixed_types(affected_columns, issue)
        elif "semantic_null" in issue_type:
            validation_result = self._validate_semantic_nulls(affected_columns, issue)
        elif "format_inconsisten" in issue_type or "inconsistent_format" in issue_type:
            validation_result = self._validate_format_inconsistency(affected_columns, issue)
        elif "missing" in issue_type:
            validation_result = self._validate_missing_data(affected_columns, issue)
        elif "outlier" in issue_type:
            validation_result = self._validate_outliers(affected_columns, issue)
        elif "duplicate" in issue_type:
            validation_result = self._validate_duplicates(issue)
        elif "correlation" in issue_type or "multicollinear" in issue_type:
            validation_result = self._validate_correlations(affected_columns, issue)
        else:
            validation_result = self._validate_generic_issue(affected_columns, issue)
        
        # Merge validation result into issue
        issue.update(validation_result)
        return issue
    
    def _validate_mixed_types(self, columns: List[str], issue: Dict[str, Any]) -> Dict[str, Any]:
        """Check if columns truly have mixed data types"""
        mixed_found = []
        evidence = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            # Sample data to check for mixing
            sample = self.df[col].dropna().head(200)  # Check more rows for better validation
            
            numeric_values = []
            text_number_values = []
            other_text_values = []
            
            for val in sample:
                val_str = str(val).strip().lower()
                
                # Check if it's a pure number
                try:
                    float(val)
                    numeric_values.append(val)
                    continue
                except:
                    pass
                
                # Check if it's a text representation of number
                if any(text_num in val_str for text_num in self.text_number_patterns.keys()):
                    text_number_values.append(val)
                elif val_str and not val_str.isspace():
                    other_text_values.append(val)
            
            # Determine if truly mixed
            has_numeric = len(numeric_values) > 0
            has_text_numbers = len(text_number_values) > 0
            has_other_text = len(other_text_values) > 0
            
            total_values = len(numeric_values) + len(text_number_values) + len(other_text_values)
            
            if total_values > 0:
                # Calculate mixing percentages
                numeric_pct = len(numeric_values) / total_values
                text_num_pct = len(text_number_values) / total_values
                other_text_pct = len(other_text_values) / total_values
                
                # Mixed if we have significant amounts (>5%) of different types
                is_mixed = False
                mix_details = []
                
                if has_numeric and (has_text_numbers or has_other_text):
                    if text_num_pct > 0.05:  # 5% threshold
                        is_mixed = True
                        mix_details.append(f"numeric ({numeric_pct:.1%}) mixed with text numbers ({text_num_pct:.1%})")
                    if other_text_pct > 0.05:
                        is_mixed = True
                        mix_details.append(f"numeric ({numeric_pct:.1%}) mixed with text ({other_text_pct:.1%})")
                
                if is_mixed:
                    mixed_found.append(col)
                    evidence.append({
                        "column": col,
                        "mix_type": ", ".join(mix_details),
                        "sample_numeric": numeric_values[:3],
                        "sample_text_numbers": text_number_values[:3],
                        "sample_other_text": other_text_values[:3]
                    })
        
        if mixed_found:
            return {
                "validation_status": "VERIFIED",
                "verification_details": f"Confirmed mixed types in {len(mixed_found)} columns: {', '.join(mixed_found)}",
                "evidence_found": evidence,
                "confidence_adjustment": 0.0
            }
        elif columns:  # Columns were specified but no mixing found
            return {
                "validation_status": "NOT_FOUND",
                "verification_details": f"Could not verify mixed types in specified columns: {', '.join(columns)}",
                "evidence_found": [],
                "confidence_adjustment": -0.3  # Reduce confidence in LLM
            }
        else:
            return {
                "validation_status": "UNVERIFIED",
                "verification_details": "No columns specified for mixed type validation",
                "evidence_found": [],
                "confidence_adjustment": 0.0
            }
    
    def _validate_semantic_nulls(self, columns: List[str], issue: Dict[str, Any]) -> Dict[str, Any]:
        """Check for semantic null values like 'N/A', 'null' as strings"""
        found_evidence = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            # Convert to string and check for semantic nulls
            sample = self.df[col].astype(str).str.lower().str.strip()
            
            semantic_null_counts = {}
            for pattern in self.semantic_null_patterns:
                count = (sample == pattern).sum()
                if count > 0:
                    semantic_null_counts[pattern] = count
            
            if semantic_null_counts:
                total_nulls = sum(semantic_null_counts.values())
                null_pct = (total_nulls / len(self.df)) * 100
                
                found_evidence.append({
                    "column": col,
                    "semantic_null_patterns": semantic_null_counts,
                    "total_semantic_nulls": total_nulls,
                    "percentage": round(null_pct, 2)
                })
        
        if found_evidence:
            total_semantic_nulls = sum(ev["total_semantic_nulls"] for ev in found_evidence)
            return {
                "validation_status": "VERIFIED",
                "verification_details": f"Found {total_semantic_nulls} semantic nulls across {len(found_evidence)} columns",
                "evidence_found": found_evidence,
                "confidence_adjustment": 0.0
            }
        else:
            return {
                "validation_status": "NOT_FOUND",
                "verification_details": "No semantic nulls found in specified columns",
                "evidence_found": [],
                "confidence_adjustment": -0.2
            }
    
    def _validate_format_inconsistency(self, columns: List[str], issue: Dict[str, Any]) -> Dict[str, Any]:
        """Check for format inconsistencies in columns"""
        inconsistent_evidence = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            sample = self.df[col].dropna().astype(str).head(200)
            
            # Check for date format variations
            date_formats_found = {"slash": 0, "dash": 0, "dot": 0}
            date_samples = {"slash": [], "dash": [], "dot": []}
            
            for val in sample:
                val_clean = val.strip()
                if len(val_clean) > 5 and any(c.isdigit() for c in val_clean):
                    if '/' in val_clean:
                        date_formats_found["slash"] += 1
                        if len(date_samples["slash"]) < 3:
                            date_samples["slash"].append(val_clean)
                    elif '-' in val_clean:
                        date_formats_found["dash"] += 1
                        if len(date_samples["dash"]) < 3:
                            date_samples["dash"].append(val_clean)
                    elif '.' in val_clean:
                        date_formats_found["dot"] += 1
                        if len(date_samples["dot"]) < 3:
                            date_samples["dot"].append(val_clean)
            
            # Count significant format variations (>5% each)
            total_potential_dates = sum(date_formats_found.values())
            if total_potential_dates > 10:  # Only check if we have enough samples
                significant_formats = []
                for fmt, count in date_formats_found.items():
                    if count / total_potential_dates > 0.05:  # >5% threshold
                        significant_formats.append(fmt)
                
                if len(significant_formats) > 1:
                    inconsistent_evidence.append({
                        "column": col,
                        "inconsistency_type": "date_formats",
                        "formats_found": {fmt: date_formats_found[fmt] for fmt in significant_formats},
                        "sample_values": {fmt: date_samples[fmt] for fmt in significant_formats},
                        "total_values_checked": total_potential_dates
                    })
            
            # Check for number format inconsistencies
            number_formats = {"currency": 0, "plain": 0, "scientific": 0, "percentage": 0}
            number_samples = {"currency": [], "plain": [], "scientific": [], "percentage": []}
            
            for val in sample:
                val_clean = val.strip()
                if re.search(r'[$€£¥]', val_clean):
                    number_formats["currency"] += 1
                    if len(number_samples["currency"]) < 3:
                        number_samples["currency"].append(val_clean)
                elif re.match(r'^\d+\.?\d*$', val_clean):
                    number_formats["plain"] += 1
                    if len(number_samples["plain"]) < 3:
                        number_samples["plain"].append(val_clean)
                elif re.search(r'\d+[eE][+-]?\d+', val_clean):
                    number_formats["scientific"] += 1
                    if len(number_samples["scientific"]) < 3:
                        number_samples["scientific"].append(val_clean)
                elif '%' in val_clean and any(c.isdigit() for c in val_clean):
                    number_formats["percentage"] += 1
                    if len(number_samples["percentage"]) < 3:
                        number_samples["percentage"].append(val_clean)
            
            total_potential_numbers = sum(number_formats.values())
            if total_potential_numbers > 10:
                significant_number_formats = []
                for fmt, count in number_formats.items():
                    if count / total_potential_numbers > 0.1:  # >10% threshold for numbers
                        significant_number_formats.append(fmt)
                
                if len(significant_number_formats) > 1:
                    inconsistent_evidence.append({
                        "column": col,
                        "inconsistency_type": "number_formats",
                        "formats_found": {fmt: number_formats[fmt] for fmt in significant_number_formats},
                        "sample_values": {fmt: number_samples[fmt] for fmt in significant_number_formats},
                        "total_values_checked": total_potential_numbers
                    })
        
        if inconsistent_evidence:
            return {
                "validation_status": "VERIFIED",
                "verification_details": f"Format inconsistencies confirmed in {len(inconsistent_evidence)} columns",
                "evidence_found": inconsistent_evidence,
                "confidence_adjustment": 0.0
            }
        else:
            return {
                "validation_status": "PARTIAL",
                "verification_details": "Some format variations detected but within acceptable ranges",
                "evidence_found": [],
                "confidence_adjustment": -0.1
            }
    
    def _validate_missing_data(self, columns: List[str], issue: Dict[str, Any]) -> Dict[str, Any]:
        """Validate missing data claims"""
        missing_evidence = []
        
        for col in columns:
            if col in self.df.columns:
                missing_count = self.df[col].isna().sum()
                missing_pct = (missing_count / len(self.df)) * 100
                
                missing_evidence.append({
                    "column": col,
                    "missing_count": missing_count,
                    "missing_percentage": round(missing_pct, 2),
                    "total_rows": len(self.df)
                })
        
        if missing_evidence:
            avg_missing = sum(ev["missing_percentage"] for ev in missing_evidence) / len(missing_evidence)
            return {
                "validation_status": "VERIFIED",
                "verification_details": f"Confirmed {avg_missing:.1f}% average missing data across {len(missing_evidence)} columns",
                "evidence_found": missing_evidence,
                "confidence_adjustment": 0.0
            }
        else:
            return {
                "validation_status": "UNVERIFIED",
                "verification_details": "Could not verify missing data claim - columns not found",
                "evidence_found": [],
                "confidence_adjustment": 0.0
            }
    
    def _validate_outliers(self, columns: List[str], issue: Dict[str, Any]) -> Dict[str, Any]:
        """Validate outlier claims using IQR method"""
        outlier_evidence = []
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                series = self.df[col].dropna()
                if len(series) > 0:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    outlier_pct = (len(outliers) / len(series)) * 100
                    
                    outlier_evidence.append({
                        "column": col,
                        "outlier_count": len(outliers),
                        "outlier_percentage": round(outlier_pct, 2),
                        "bounds": [float(lower_bound), float(upper_bound)],
                        "extreme_values": [float(x) for x in outliers.head(5).tolist()]
                    })
        
        if outlier_evidence:
            avg_outlier_pct = sum(ev["outlier_percentage"] for ev in outlier_evidence) / len(outlier_evidence)
            return {
                "validation_status": "VERIFIED",
                "verification_details": f"Confirmed {avg_outlier_pct:.1f}% average outliers across {len(outlier_evidence)} columns",
                "evidence_found": outlier_evidence,
                "confidence_adjustment": 0.0
            }
        else:
            return {
                "validation_status": "NOT_FOUND",
                "verification_details": "No significant outliers found in specified columns",
                "evidence_found": [],
                "confidence_adjustment": -0.15
            }
    
    def _validate_duplicates(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Validate duplicate row claims"""
        duplicate_count = self.df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(self.df)) * 100
        
        if duplicate_count > 0:
            return {
                "validation_status": "VERIFIED",
                "verification_details": f"Confirmed {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)",
                "evidence_found": [{
                    "duplicate_count": duplicate_count,
                    "duplicate_percentage": round(duplicate_pct, 2),
                    "total_rows": len(self.df)
                }],
                "confidence_adjustment": 0.0
            }
        else:
            return {
                "validation_status": "NOT_FOUND",
                "verification_details": "No duplicate rows found",
                "evidence_found": [],
                "confidence_adjustment": -0.2
            }
    
    def _validate_correlations(self, columns: List[str], issue: Dict[str, Any]) -> Dict[str, Any]:
        """Validate high correlation claims"""
        if len(columns) < 2:
            return {
                "validation_status": "UNVERIFIED",
                "verification_details": "Need at least 2 columns to check correlations",
                "evidence_found": [],
                "confidence_adjustment": 0.0
            }
        
        # Check correlations between specified columns
        numeric_cols = [col for col in columns if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])]
        
        if len(numeric_cols) < 2:
            return {
                "validation_status": "NOT_FOUND",
                "verification_details": "Insufficient numeric columns for correlation analysis",
                "evidence_found": [],
                "confidence_adjustment": -0.1
            }
        
        correlation_evidence = []
        corr_matrix = self.df[numeric_cols].corr()
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    correlation_evidence.append({
                        "feature1": numeric_cols[i],
                        "feature2": numeric_cols[j],
                        "correlation": round(float(corr_val), 3),
                        "strength": "very high" if abs(corr_val) > 0.9 else "high"
                    })
        
        if correlation_evidence:
            return {
                "validation_status": "VERIFIED",
                "verification_details": f"Confirmed {len(correlation_evidence)} high correlation pairs",
                "evidence_found": correlation_evidence,
                "confidence_adjustment": 0.0
            }
        else:
            return {
                "validation_status": "NOT_FOUND",
                "verification_details": "No high correlations found between specified columns",
                "evidence_found": [],
                "confidence_adjustment": -0.2
            }
    
    def _validate_generic_issue(self, columns: List[str], issue: Dict[str, Any]) -> Dict[str, Any]:
        """Generic validation for unrecognized issue types"""
        return {
            "validation_status": "UNVERIFIED",
            "verification_details": f"Cannot validate issue type: {issue.get('issue_type', 'unknown')}",
            "evidence_found": [],
            "confidence_adjustment": 0.0
        }
    
    def _calculate_confidence_adjustment(self, verification_rate: float, partial_rate: float) -> float:
        """Calculate overall confidence adjustment based on validation results"""
        if verification_rate > 0.8:
            return 0.1  # Increase confidence
        elif verification_rate + partial_rate > 0.6:
            return 0.0  # Neutral
        elif verification_rate + partial_rate > 0.3:
            return -0.2  # Decrease confidence moderately  
        else:
            return -0.4  # Decrease confidence significantly
    
    def get_validation_summary_for_display(self, validation_summary: Dict[str, Any]) -> Dict[str, str]:
        """Generate user-friendly validation summary"""
        reliability = validation_summary.get("llm_reliability", "UNKNOWN")
        verification_rate = validation_summary.get("verification_rate", 0)
        total_issues = validation_summary.get("total_issues_claimed", 0)
        verified_issues = validation_summary.get("issues_verified", 0)
        
        if reliability == "HIGH":
            summary_text = f"High reliability: {verified_issues}/{total_issues} LLM findings verified ({verification_rate:.1%})"
            recommendation = "LLM analysis appears accurate and trustworthy"
            color = "green"
        elif reliability == "MEDIUM":
            summary_text = f"Medium reliability: {verified_issues}/{total_issues} LLM findings verified ({verification_rate:.1%})"
            recommendation = "LLM analysis partially validated - some findings may need manual review"
            color = "orange"
        else:
            summary_text = f"Low reliability: {verified_issues}/{total_issues} LLM findings verified ({verification_rate:.1%})"
            recommendation = "LLM analysis not well validated - recommend manual data inspection"
            color = "red"
        
        return {
            "summary": summary_text,
            "recommendation": recommendation,
            "color": color,
            "reliability": reliability
        }
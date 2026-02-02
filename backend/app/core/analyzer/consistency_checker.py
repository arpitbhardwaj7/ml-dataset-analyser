"""
Data Consistency Checker for ML Quality Assessment
Detects mixed data types, categorical inconsistencies, and string noise patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
import re
from collections import Counter

class ConsistencyChecker:
    """
    Detects data consistency issues that affect ML model training
    Focuses on mixed types, categorical inconsistencies, and string noise
    """
    
    def __init__(self):
        # Common string noise patterns
        self.noise_patterns = {
            "null_representations": [
                "n/a", "na", "null", "none", "nil", "missing", "unknown", 
                "--", "-", ".", "..", "...", "?", "??", "???",
                "not available", "not applicable", "no data", "no value",
                "empty", "blank", "undefined", "void", "nan", "n.a.",
                "#n/a", "#null!", "#value!", "#ref!", "#div/0!", "#num!",
                "<null>", "<na>", "<none>", "<missing>", "<blank>"
            ],
            
            "inconsistent_boolean": [
                ("true", "false"), ("yes", "no"), ("y", "n"), ("1", "0"),
                ("on", "off"), ("active", "inactive"), ("enabled", "disabled"),
                ("valid", "invalid"), ("success", "failure"), ("pass", "fail")
            ],
            
            "numeric_text_mixing": [
                r"\d+", r"zero|one|two|three|four|five|six|seven|eight|nine|ten",
                r"first|second|third|fourth|fifth", r"dozen|hundred|thousand"
            ]
        }
        
        # Unit inconsistency patterns
        self.unit_patterns = {
            "currency": [r"\$\d+", r"\d+\s*dollars?", r"\d+\s*usd", r"\d+k", r"\d+m"],
            "percentage": [r"\d+%", r"\d+\s*percent", r"0\.\d+", r"\d+\s*pct"],
            "phone": [r"\d{3}-\d{3}-\d{4}", r"\(\d{3}\)\s*\d{3}-\d{4}", r"\d{10}", r"\+1-\d{3}-\d{3}-\d{4}"],
            "date": [r"\d{2}/\d{2}/\d{4}", r"\d{4}-\d{2}-\d{2}", r"\w+\s+\d{1,2},\s+\d{4}"]
        }
    
    def check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive consistency analysis
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with consistency analysis results
        """
        
        consistency_results = {
            "mixed_type_columns": [],
            "inconsistent_categorical_columns": [],
            "string_noise_columns": [],
            "unit_inconsistency_columns": [],
            "format_inconsistency_columns": [],
            "overall_consistency_issues": 0,
            "consistency_score": 100,
            "recommendations": []
        }
        
        # Check each column for consistency issues
        for column in df.columns:
            series = df[column]
            
            # Skip completely empty columns
            if series.isna().all():
                continue
            
            # Check for mixed data types
            mixed_type_issues = self._check_mixed_types(series, column)
            if mixed_type_issues:
                consistency_results["mixed_type_columns"].append(mixed_type_issues)
            
            # Check for categorical inconsistencies
            categorical_issues = self._check_categorical_consistency(series, column)
            if categorical_issues:
                consistency_results["inconsistent_categorical_columns"].append(categorical_issues)
            
            # Check for string noise
            string_noise_issues = self._check_string_noise(series, column)
            if string_noise_issues:
                consistency_results["string_noise_columns"].append(string_noise_issues)
            
            # Check for unit inconsistencies
            unit_issues = self._check_unit_inconsistencies(series, column)
            if unit_issues:
                consistency_results["unit_inconsistency_columns"].append(unit_issues)
            
            # Check for format inconsistencies
            format_issues = self._check_format_inconsistencies(series, column)
            if format_issues:
                consistency_results["format_inconsistency_columns"].append(format_issues)
        
        # Calculate overall consistency metrics
        consistency_results = self._calculate_consistency_metrics(consistency_results)
        
        # Generate recommendations
        consistency_results["recommendations"] = self._generate_consistency_recommendations(consistency_results)
        
        return consistency_results
    
    def _check_mixed_types(self, series: pd.Series, column_name: str) -> Optional[Dict[str, Any]]:
        """Check for mixed data types within a single column"""
        
        # Skip if all values are null
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return None
        
        # Analyze value types
        type_analysis = self._analyze_value_types(non_null_series)
        
        # If we have multiple fundamental types, it's a mixed type column
        fundamental_types = len([t for t in type_analysis["type_counts"].keys() 
                               if type_analysis["type_counts"][t] > 0])
        
        if fundamental_types > 1:
            # Check if it's a semantic mixing (like "25" and 25)
            semantic_issue = self._detect_semantic_mixing(non_null_series)
            
            return {
                "column": column_name,
                "issue_type": "mixed_data_types",
                "severity": "high" if semantic_issue else "medium",
                "type_analysis": type_analysis,
                "semantic_mixing": semantic_issue,
                "examples": self._get_mixed_type_examples(non_null_series),
                "impact": "Will cause type errors during preprocessing and model training"
            }
        
        return None
    
    def _analyze_value_types(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the types of values in a series"""
        
        type_counts = {
            "numeric": 0,
            "text_numeric": 0,  # Numbers represented as text
            "boolean_like": 0,
            "categorical": 0,
            "datetime_like": 0,
            "mixed": 0
        }
        
        sample_values = {}
        
        for value in series.head(100):  # Sample first 100 values
            str_value = str(value).strip().lower()
            
            # Check if it's a number
            try:
                float(value)
                type_counts["numeric"] += 1
                if "numeric" not in sample_values:
                    sample_values["numeric"] = [value]
                elif len(sample_values["numeric"]) < 3:
                    sample_values["numeric"].append(value)
            except (ValueError, TypeError):
                # Check if it's a text representation of a number
                if self._is_text_numeric(str_value):
                    type_counts["text_numeric"] += 1
                    if "text_numeric" not in sample_values:
                        sample_values["text_numeric"] = [str_value]
                    elif len(sample_values["text_numeric"]) < 3:
                        sample_values["text_numeric"].append(str_value)
                
                # Check if it's boolean-like
                elif self._is_boolean_like(str_value):
                    type_counts["boolean_like"] += 1
                    if "boolean_like" not in sample_values:
                        sample_values["boolean_like"] = [str_value]
                    elif len(sample_values["boolean_like"]) < 3:
                        sample_values["boolean_like"].append(str_value)
                
                # Check if it's datetime-like
                elif self._is_datetime_like(str_value):
                    type_counts["datetime_like"] += 1
                    if "datetime_like" not in sample_values:
                        sample_values["datetime_like"] = [str_value]
                    elif len(sample_values["datetime_like"]) < 3:
                        sample_values["datetime_like"].append(str_value)
                
                # Otherwise, it's categorical/text
                else:
                    type_counts["categorical"] += 1
                    if "categorical" not in sample_values:
                        sample_values["categorical"] = [str_value]
                    elif len(sample_values["categorical"]) < 3:
                        sample_values["categorical"].append(str_value)
        
        return {
            "type_counts": type_counts,
            "sample_values": sample_values,
            "total_analyzed": min(100, len(series))
        }
    
    def _is_text_numeric(self, value: str) -> bool:
        """Check if a string represents a number in text form"""
        
        text_numbers = {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
            "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million"
        }
        
        # Check for text number representations
        if value in text_numbers:
            return True
        
        # Check for ordinals
        ordinal_pattern = r"(first|second|third|\d+(st|nd|rd|th))"
        if re.match(ordinal_pattern, value):
            return True
        
        return False
    
    def _is_boolean_like(self, value: str) -> bool:
        """Check if a string represents a boolean value"""
        
        boolean_values = {
            "true", "false", "yes", "no", "y", "n", "1", "0",
            "on", "off", "active", "inactive", "enabled", "disabled",
            "valid", "invalid", "success", "failure", "pass", "fail"
        }
        
        return value in boolean_values
    
    def _is_datetime_like(self, value: str) -> bool:
        """Check if a string looks like a date"""
        
        date_patterns = [
            r"\d{1,2}/\d{1,2}/\d{4}",      # MM/DD/YYYY
            r"\d{4}-\d{2}-\d{2}",          # YYYY-MM-DD
            r"\w+\s+\d{1,2},\s+\d{4}",     # Month DD, YYYY
            r"\d{1,2}-\w+-\d{4}",          # DD-Mon-YYYY
        ]
        
        return any(re.match(pattern, value) for pattern in date_patterns)
    
    def _detect_semantic_mixing(self, series: pd.Series) -> Dict[str, Any]:
        """Detect if mixed types represent the same semantic concept"""
        
        # Check for numeric/text numeric mixing
        numeric_count = 0
        text_numeric_count = 0
        examples = {"numeric": [], "text_numeric": []}
        
        for value in series.head(50):
            str_value = str(value).strip().lower()
            
            try:
                float(value)
                numeric_count += 1
                if len(examples["numeric"]) < 3:
                    examples["numeric"].append(value)
            except (ValueError, TypeError):
                if self._is_text_numeric(str_value):
                    text_numeric_count += 1
                    if len(examples["text_numeric"]) < 3:
                        examples["text_numeric"].append(str_value)
        
        if numeric_count > 0 and text_numeric_count > 0:
            return {
                "type": "numeric_text_mixing",
                "numeric_count": numeric_count,
                "text_numeric_count": text_numeric_count,
                "examples": examples,
                "severity": "high"
            }
        
        return {}
    
    def _check_categorical_consistency(self, series: pd.Series, column_name: str) -> Optional[Dict[str, Any]]:
        """Check for inconsistencies in categorical data"""
        
        # Only check object/categorical columns
        if series.dtype not in ['object', 'category']:
            return None
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return None
        
        # Convert to lowercase for comparison
        lower_values = non_null_series.astype(str).str.lower().str.strip()
        value_counts = lower_values.value_counts()
        
        # Find potential inconsistencies
        inconsistencies = self._find_categorical_inconsistencies(value_counts.index.tolist())
        
        if inconsistencies:
            return {
                "column": column_name,
                "issue_type": "categorical_inconsistency",
                "severity": "medium",
                "inconsistencies": inconsistencies,
                "unique_values": len(value_counts),
                "examples": self._get_inconsistency_examples(series, inconsistencies),
                "impact": "Increases feature dimensionality and introduces noise"
            }
        
        return None
    
    def _find_categorical_inconsistencies(self, values: List[str]) -> List[Dict[str, Any]]:
        """Find groups of values that represent the same concept"""
        
        inconsistencies = []
        
        # Check for case variations
        case_groups = {}
        for value in values:
            lower_val = value.lower()
            if lower_val not in case_groups:
                case_groups[lower_val] = []
            case_groups[lower_val].append(value)
        
        for lower_val, variants in case_groups.items():
            if len(variants) > 1:
                inconsistencies.append({
                    "type": "case_variation",
                    "concept": lower_val,
                    "variants": variants,
                    "severity": "low"
                })
        
        # Check for whitespace variations
        stripped_groups = {}
        for value in values:
            stripped_val = value.strip()
            if stripped_val not in stripped_groups:
                stripped_groups[stripped_val] = []
            stripped_groups[stripped_val].append(value)
        
        for stripped_val, variants in stripped_groups.items():
            if len(variants) > 1:
                inconsistencies.append({
                    "type": "whitespace_variation",
                    "concept": stripped_val,
                    "variants": variants,
                    "severity": "low"
                })
        
        # Check for abbreviation/expansion patterns
        abbreviation_patterns = [
            (r"^m$", ["male", "man"]),
            (r"^f$", ["female", "woman"]),
            (r"^y$", ["yes", "true"]),
            (r"^n$", ["no", "false"]),
            (r"^st$", ["street"]),
            (r"^ave$", ["avenue"]),
            (r"^dr$", ["doctor", "drive"])
        ]
        
        for pattern, expansions in abbreviation_patterns:
            abbrev_matches = [v for v in values if re.match(pattern, v.lower())]
            expansion_matches = [v for v in values if any(exp in v.lower() for exp in expansions)]
            
            if abbrev_matches and expansion_matches:
                inconsistencies.append({
                    "type": "abbreviation_expansion",
                    "abbreviations": abbrev_matches,
                    "expansions": expansion_matches,
                    "severity": "medium"
                })
        
        return inconsistencies
    
    def _check_string_noise(self, series: pd.Series, column_name: str) -> Optional[Dict[str, Any]]:
        """Check for string noise patterns"""
        
        if series.dtype not in ['object', 'category']:
            return None
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return None
        
        # Check for null representations
        null_noise = self._find_null_representations(non_null_series)
        
        # Check for placeholder values
        placeholder_noise = self._find_placeholder_values(non_null_series)
        
        # Check for encoding issues
        encoding_noise = self._find_encoding_issues(non_null_series)
        
        total_noise_count = len(null_noise) + len(placeholder_noise) + len(encoding_noise)
        
        if total_noise_count > 0:
            noise_percentage = (total_noise_count / len(non_null_series)) * 100
            
            return {
                "column": column_name,
                "issue_type": "string_noise",
                "severity": "high" if noise_percentage > 10 else "medium",
                "null_representations": null_noise,
                "placeholder_values": placeholder_noise,
                "encoding_issues": encoding_noise,
                "noise_percentage": round(noise_percentage, 2),
                "impact": "Introduces noise and may be treated as valid categories"
            }
        
        return None
    
    def _find_null_representations(self, series: pd.Series) -> List[str]:
        """Find values that represent null/missing data"""
        
        null_noise = []
        series_lower = series.astype(str).str.lower().str.strip()
        
        for null_pattern in self.noise_patterns["null_representations"]:
            matches = series_lower[series_lower == null_pattern]
            if len(matches) > 0:
                null_noise.extend(matches.tolist())
        
        return list(set(null_noise))
    
    def _find_placeholder_values(self, series: pd.Series) -> List[str]:
        """Find placeholder or test values"""
        
        placeholder_patterns = [
            r"test\d*", r"dummy\d*", r"placeholder\d*", r"temp\d*",
            r"sample\d*", r"example\d*", r"default\d*", r"tbd",
            r"todo", r"fixme", r"xxx+", r"aaa+", r"zzz+"
        ]
        
        placeholders = []
        series_lower = series.astype(str).str.lower().str.strip()
        
        for pattern in placeholder_patterns:
            matches = series_lower[series_lower.str.match(pattern)]
            if len(matches) > 0:
                placeholders.extend(matches.tolist())
        
        return list(set(placeholders))
    
    def _find_encoding_issues(self, series: pd.Series) -> List[str]:
        """Find potential character encoding issues"""
        
        encoding_issues = []
        
        for value in series.astype(str):
            # Check for common encoding issue patterns
            if any(char in value for char in ['�', '\\x', '\\u']):
                encoding_issues.append(value)
            
            # Check for HTML entities
            if re.search(r'&[a-zA-Z]+;|&#\d+;', value):
                encoding_issues.append(value)
        
        return list(set(encoding_issues))
    
    def _check_unit_inconsistencies(self, series: pd.Series, column_name: str) -> Optional[Dict[str, Any]]:
        """Check for inconsistent units or formats"""
        
        if series.dtype not in ['object', 'category']:
            return None
        
        non_null_series = series.dropna().astype(str)
        if len(non_null_series) == 0:
            return None
        
        unit_findings = {}
        
        for unit_type, patterns in self.unit_patterns.items():
            matches_per_pattern = []
            
            for pattern in patterns:
                matches = non_null_series[non_null_series.str.contains(pattern, regex=True, case=False)]
                if len(matches) > 0:
                    matches_per_pattern.append({
                        "pattern": pattern,
                        "matches": matches.tolist()[:5],  # Sample matches
                        "count": len(matches)
                    })
            
            if len(matches_per_pattern) > 1:  # Multiple patterns for same unit type
                unit_findings[unit_type] = matches_per_pattern
        
        if unit_findings:
            return {
                "column": column_name,
                "issue_type": "unit_inconsistency",
                "severity": "medium",
                "unit_inconsistencies": unit_findings,
                "impact": "Same values represented in different formats"
            }
        
        return None
    
    def _check_format_inconsistencies(self, series: pd.Series, column_name: str) -> Optional[Dict[str, Any]]:
        """Check for format inconsistencies within the same data type"""
        
        if series.dtype not in ['object', 'category']:
            return None
        
        non_null_series = series.dropna().astype(str)
        if len(non_null_series) == 0:
            return None
        
        # Check for date format inconsistencies
        date_formats = self._detect_date_formats(non_null_series)
        if len(date_formats) > 1:
            return {
                "column": column_name,
                "issue_type": "format_inconsistency",
                "data_type": "date",
                "severity": "medium",
                "formats_found": date_formats,
                "impact": "Requires format standardization before processing"
            }
        
        # Check for phone format inconsistencies  
        phone_formats = self._detect_phone_formats(non_null_series)
        if len(phone_formats) > 1:
            return {
                "column": column_name,
                "issue_type": "format_inconsistency",
                "data_type": "phone",
                "severity": "low",
                "formats_found": phone_formats,
                "impact": "Multiple phone number formats detected"
            }
        
        return None
    
    def _detect_date_formats(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Detect different date formats in a series"""
        
        date_format_patterns = [
            (r"\d{1,2}/\d{1,2}/\d{4}", "MM/DD/YYYY"),
            (r"\d{4}-\d{2}-\d{2}", "YYYY-MM-DD"),
            (r"\d{2}-\d{2}-\d{4}", "DD-MM-YYYY"),
            (r"\w+\s+\d{1,2},\s+\d{4}", "Month DD, YYYY"),
            (r"\d{1,2}\s+\w+\s+\d{4}", "DD Month YYYY")
        ]
        
        formats_found = []
        
        for pattern, format_name in date_format_patterns:
            matches = series[series.str.match(pattern, case=False)]
            if len(matches) > 0:
                formats_found.append({
                    "format": format_name,
                    "pattern": pattern,
                    "count": len(matches),
                    "examples": matches.head(3).tolist()
                })
        
        return formats_found
    
    def _detect_phone_formats(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Detect different phone number formats"""
        
        phone_patterns = [
            (r"\d{3}-\d{3}-\d{4}", "XXX-XXX-XXXX"),
            (r"\(\d{3}\)\s*\d{3}-\d{4}", "(XXX) XXX-XXXX"),
            (r"\d{10}", "XXXXXXXXXX"),
            (r"\+1-\d{3}-\d{3}-\d{4}", "+1-XXX-XXX-XXXX"),
            (r"\d{3}\.\d{3}\.\d{4}", "XXX.XXX.XXXX")
        ]
        
        formats_found = []
        
        for pattern, format_name in phone_patterns:
            matches = series[series.str.match(pattern)]
            if len(matches) > 0:
                formats_found.append({
                    "format": format_name,
                    "pattern": pattern,
                    "count": len(matches),
                    "examples": matches.head(3).tolist()
                })
        
        return formats_found
    
    def _get_mixed_type_examples(self, series: pd.Series) -> List[Any]:
        """Get examples of mixed types"""
        
        examples = []
        type_examples = {}
        
        for value in series.head(20):
            try:
                float(value)
                if "numeric" not in type_examples:
                    type_examples["numeric"] = []
                if len(type_examples["numeric"]) < 2:
                    type_examples["numeric"].append(value)
            except (ValueError, TypeError):
                if "text" not in type_examples:
                    type_examples["text"] = []
                if len(type_examples["text"]) < 2:
                    type_examples["text"].append(value)
        
        for type_name, values in type_examples.items():
            examples.extend(values)
        
        return examples
    
    def _get_inconsistency_examples(self, series: pd.Series, inconsistencies: List[Dict]) -> List[str]:
        """Get examples of categorical inconsistencies"""
        
        examples = []
        for inconsistency in inconsistencies[:3]:  # Top 3 inconsistencies
            if "variants" in inconsistency:
                examples.extend(inconsistency["variants"][:2])
        
        return examples
    
    def _calculate_consistency_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall consistency score and metrics"""
        
        # Count total issues
        total_issues = (
            len(results["mixed_type_columns"]) +
            len(results["inconsistent_categorical_columns"]) +
            len(results["string_noise_columns"]) +
            len(results["unit_inconsistency_columns"]) +
            len(results["format_inconsistency_columns"])
        )
        
        results["overall_consistency_issues"] = total_issues
        
        # Calculate penalty based on issue severity
        penalty = 0
        
        # Mixed types are critical
        for issue in results["mixed_type_columns"]:
            penalty += 20 if issue["severity"] == "high" else 10
        
        # String noise is significant
        for issue in results["string_noise_columns"]:
            penalty += 15 if issue["severity"] == "high" else 8
        
        # Categorical inconsistencies
        for issue in results["inconsistent_categorical_columns"]:
            penalty += 8
        
        # Unit and format issues
        for issue in results["unit_inconsistency_columns"]:
            penalty += 5
        
        for issue in results["format_inconsistency_columns"]:
            penalty += 3
        
        # Calculate final score
        results["consistency_score"] = max(0, 100 - penalty)
        
        return results
    
    def _generate_consistency_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations for consistency issues"""
        
        recommendations = []
        
        if results["mixed_type_columns"]:
            recommendations.append("CRITICAL: Fix mixed data types before model training - will cause type errors")
            for issue in results["mixed_type_columns"][:3]:
                recommendations.append(f"Column '{issue['column']}': Convert all values to consistent type")
        
        if results["string_noise_columns"]:
            recommendations.append("Clean string noise patterns (null representations, placeholders)")
            for issue in results["string_noise_columns"][:2]:
                recommendations.append(f"Column '{issue['column']}': Replace noise with proper null values")
        
        if results["inconsistent_categorical_columns"]:
            recommendations.append("Standardize categorical values to reduce feature dimensionality")
            
        if results["unit_inconsistency_columns"]:
            recommendations.append("Normalize unit representations (currency, percentages, etc.)")
            
        if results["format_inconsistency_columns"]:
            recommendations.append("Standardize format consistency for dates, phones, etc.")
        
        if not recommendations:
            recommendations.append("Data consistency looks good - no major issues detected")
        
        return recommendations
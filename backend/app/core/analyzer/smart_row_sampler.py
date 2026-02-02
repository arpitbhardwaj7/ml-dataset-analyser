import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import random


class SmartRowSampler:
    """Intelligent row selection for LLM analysis to identify data quality issues"""
    
    def __init__(self, max_samples: int = 50):
        self.max_samples = max_samples
    
    def select_for_llm_analysis(self, df: pd.DataFrame, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to select most problematic and representative rows for LLM analysis
        
        Returns: Dictionary with categorized samples for LLM prompt
        """
        # Use enhanced stratified sampling
        return self.get_stratified_samples(df, profile_data, self.max_samples)
    
    def get_stratified_samples(self, df: pd.DataFrame, profile_data: Dict[str, Any], max_samples: int = 50) -> Dict[str, Any]:
        """Enhanced sampling with temporal and distribution stratification"""
        
        samples = []
        
        # Allocate sample budget
        critical_budget = int(max_samples * 0.30)     # 30% for critical issues
        high_budget = int(max_samples * 0.25)         # 25% for high severity
        temporal_budget = int(max_samples * 0.20)     # 20% for temporal/distribution coverage
        medium_budget = int(max_samples * 0.15)       # 15% for medium severity
        baseline_budget = max_samples - critical_budget - high_budget - temporal_budget - medium_budget  # Remainder for baseline
        
        # 1. Get critical issues (blocking ML training)
        critical_samples = self._get_critical_samples(df, critical_budget)
        samples.extend(critical_samples)
        
        # 2. Get high severity issues
        high_samples = self._get_high_severity_samples(df, profile_data, high_budget)
        samples.extend(high_samples)
        
        # 3. Get temporal/distribution stratified samples
        temporal_samples = self._get_temporal_samples(df, temporal_budget)
        samples.extend(temporal_samples)
        
        # 4. Get medium severity issues
        medium_samples = self._get_medium_severity_samples(df, profile_data, medium_budget)
        samples.extend(medium_samples)
        
        # 5. Get baseline samples for comparison
        baseline_samples = self._get_baseline_samples(df, baseline_budget)
        samples.extend(baseline_samples)
        
        # 6. Deduplicate and format
        unique_samples = self.deduplicate_samples(samples)
        return self.format_for_llm(unique_samples)
    
    def _get_critical_samples(self, df: pd.DataFrame, budget: int) -> List[Dict[str, Any]]:
        """Get samples with critical issues that completely block ML training"""
        critical_samples = self.get_critical_issue_rows(df)
        return critical_samples[:budget]
    
    def _get_high_severity_samples(self, df: pd.DataFrame, profile_data: Dict[str, Any], budget: int) -> List[Dict[str, Any]]:
        """Get samples with high severity issues"""
        high_samples = []
        high_samples.extend(self.get_type_inconsistency_rows(df))
        high_samples.extend(self.get_missing_pattern_rows(df))
        
        # Prioritize by severity and return top samples
        high_samples_prioritized = []
        for sample in high_samples:
            if sample.get('severity') in ['HIGH', 'CRITICAL']:
                high_samples_prioritized.append(sample)
        
        return high_samples_prioritized[:budget]
    
    def _get_medium_severity_samples(self, df: pd.DataFrame, profile_data: Dict[str, Any], budget: int) -> List[Dict[str, Any]]:
        """Get samples with medium severity issues"""
        medium_samples = []
        medium_samples.extend(self.get_outlier_rows(df, profile_data))
        
        # Add other medium severity issues
        for sample in medium_samples:
            sample['severity'] = 'MEDIUM'
        
        return medium_samples[:budget]
    
    def _get_baseline_samples(self, df: pd.DataFrame, budget: int) -> List[Dict[str, Any]]:
        """Get clean baseline samples"""
        return self.get_baseline_rows(df)[:budget]
    
    def _get_temporal_samples(self, df: pd.DataFrame, budget: int) -> List[Dict[str, Any]]:
        """Sample from different time periods to catch temporal issues"""
        temporal_samples = []
        
        if budget <= 0:
            return temporal_samples
        
        # Try to find date columns
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'created' in col.lower() or 'updated' in col.lower():
                date_columns.append(col)
        
        # Also check for columns that might contain dates
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(10)
                date_like_count = 0
                for val in sample_values:
                    val_str = str(val).strip()
                    # Check if it looks like a date (has numbers and separators)
                    if len(val_str) > 6 and any(c.isdigit() for c in val_str) and any(sep in val_str for sep in ['/', '-', '.']):
                        # Try to parse as date
                        try:
                            pd.to_datetime(val_str, errors='raise')
                            date_like_count += 1
                        except:
                            pass
                
                # If >50% of samples look like dates, consider this a date column
                if date_like_count > len(sample_values) * 0.5:
                    date_columns.append(col)
        
        if date_columns:
            # Use the first date column for stratification
            date_col = date_columns[0]
            try:
                # Try to convert to datetime
                date_series = pd.to_datetime(df[date_col], errors='coerce')
                valid_dates = date_series.dropna()
                
                if len(valid_dates) > 10:  # Need reasonable amount of valid dates
                    df_with_dates = df[date_series.notna()].copy()
                    df_with_dates['_parsed_date'] = valid_dates
                    df_sorted = df_with_dates.sort_values(by='_parsed_date')
                    
                    # Sample from beginning, middle, and end
                    periods = ['early', 'middle', 'late']
                    samples_per_period = budget // 3
                    
                    for i, period in enumerate(periods):
                        if samples_per_period <= 0:
                            break
                            
                        if period == 'early':
                            period_df = df_sorted.head(len(df_sorted) // 3)
                        elif period == 'middle':
                            third = len(df_sorted) // 3
                            period_df = df_sorted.iloc[third:2*third]
                        else:  # late
                            period_df = df_sorted.tail(len(df_sorted) // 3)
                        
                        if len(period_df) > 0:
                            sample_size = min(samples_per_period, len(period_df))
                            if sample_size > 0:
                                sampled = period_df.sample(n=sample_size, random_state=42+i)
                                for idx, row in sampled.iterrows():
                                    # Remove the helper column before adding to samples
                                    row_dict = row.drop('_parsed_date').to_dict()
                                    temporal_samples.append({
                                        'row': row_dict,
                                        'issue': f'Temporal sample from {period} time period',
                                        'severity': 'TEMPORAL',
                                        'issue_type': 'temporal_coverage',
                                        'context': f'Temporal coverage - {period} data from {date_col}',
                                        'temporal_period': period,
                                        'date_column_used': date_col
                                    })
            except Exception as e:
                # If date parsing fails, fall back to positional sampling
                pass
        
        # If no temporal samples yet, do positional distribution sampling
        if not temporal_samples and budget > 0:
            # Sample evenly across the dataset rows
            total_rows = len(df)
            if total_rows > budget:
                # Calculate step size for even distribution
                step = total_rows // budget
                indices = list(range(0, total_rows, step))[:budget]
            else:
                indices = list(range(total_rows))
            
            for i, idx in enumerate(indices):
                if idx < len(df):
                    row = df.iloc[idx]
                    position_type = 'early' if i < len(indices) // 3 else 'middle' if i < 2 * len(indices) // 3 else 'late'
                    temporal_samples.append({
                        'row': row.to_dict(),
                        'issue': f'Positional sample from {position_type} part of dataset (row {idx+1})',
                        'severity': 'TEMPORAL',
                        'issue_type': 'positional_coverage',
                        'context': f'Even distribution - {position_type} section',
                        'temporal_period': position_type,
                        'row_position': idx + 1
                    })
        
        return temporal_samples[:budget]
    
    def get_critical_issue_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find rows with blocking issues that prevent ML"""
        critical_rows = []
        
        # 1. Mixed data types in same column
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            # Check for numeric and string mix
            has_numeric = series.apply(lambda x: isinstance(x, (int, float, np.integer, np.floating))).any()
            has_string_numbers = series.apply(lambda x: isinstance(x, str) and any(word in x.lower() 
                                                                                  for word in ['thirty', 'twenty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'])).any()
            has_other_strings = series.apply(lambda x: isinstance(x, str) and x not in ['', ' '] and not str(x).replace('.', '').replace('-', '').isdigit()).any()
            
            if has_numeric and (has_string_numbers or has_other_strings):
                # Get examples of each type using proper indexing
                numeric_indices = []
                string_indices = []
                
                for idx, value in series.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        numeric_indices.append(idx)
                    elif isinstance(value, str) and value not in ['', ' ']:
                        string_indices.append(idx)
                
                if numeric_indices:
                    numeric_samples = df.loc[numeric_indices[:2]]  # Get first 2
                    for _, row in numeric_samples.iterrows():
                        critical_rows.append({
                            'row': row.to_dict(),
                            'issue': f'Mixed data types in {col}: numeric value {row[col]}',
                            'severity': 'CRITICAL',
                            'column': col,
                            'issue_type': 'mixed_types'
                        })
                
                if string_indices:
                    string_samples = df.loc[string_indices[:2]]  # Get first 2
                    for _, row in string_samples.iterrows():
                        critical_rows.append({
                            'row': row.to_dict(),
                            'issue': f'Mixed data types in {col}: string value "{row[col]}"',
                            'severity': 'CRITICAL',
                            'column': col,
                            'issue_type': 'mixed_types'
                        })
        
        # 2. Non-standard null representations
        null_patterns = ['NAN', ' NAN ', 'nan', 'null', 'NULL', 'N/A', '#N/A', 'n/a', 'Na', 'NA']
        for col in df.columns:
            for pattern in null_patterns:
                mask = df[col].astype(str).str.contains(pattern, case=False, na=False, regex=False)
                if mask.any():
                    samples = df[mask].head(2)
                    for _, row in samples.iterrows():
                        critical_rows.append({
                            'row': row.to_dict(),
                            'issue': f'Non-standard null representation "{row[col]}" in {col}',
                            'severity': 'HIGH',
                            'column': col,
                            'issue_type': 'semantic_nulls'
                        })
        
        # 3. Suspicious repeated values that might be placeholders
        for col in df.columns:
            if df[col].dtype == 'object':
                value_counts = df[col].value_counts()
                total_rows = len(df[col].dropna())
                
                # Check for values that appear in >50% of rows (suspicious)
                for value, count in value_counts.head(3).items():
                    if count > total_rows * 0.5 and str(value) not in ['', ' ', 'nan']:
                        samples = df[df[col] == value].head(1)
                        for _, row in samples.iterrows():
                            critical_rows.append({
                                'row': row.to_dict(),
                                'issue': f'Suspicious repeated value "{value}" appears {count}/{total_rows} times in {col}',
                                'severity': 'MEDIUM',
                                'column': col,
                                'issue_type': 'repeated_placeholders'
                            })
        
        return critical_rows[:15]  # Limit to 15 critical samples
    
    def get_type_inconsistency_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find format inconsistencies within columns"""
        inconsistent_rows = []
        
        # Date format inconsistencies
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(50)
                date_formats = {'slash': [], 'dash': [], 'dot': [], 'other': []}
                
                for idx, val in sample_values.items():
                    val_str = str(val).strip()
                    if len(val_str) > 5:  # Reasonable date length
                        if '/' in val_str and any(c.isdigit() for c in val_str):
                            date_formats['slash'].append(idx)
                        elif '-' in val_str and any(c.isdigit() for c in val_str):
                            date_formats['dash'].append(idx)
                        elif '.' in val_str and any(c.isdigit() for c in val_str):
                            date_formats['dot'].append(idx)
                
                # If multiple date formats detected
                format_types = [fmt for fmt, indices in date_formats.items() if len(indices) > 0]
                if len(format_types) > 1:
                    for format_type in format_types[:2]:  # Max 2 format types
                        indices = date_formats[format_type][:2]  # Max 2 examples per format
                        for idx in indices:
                            row = df.loc[idx]
                            inconsistent_rows.append({
                                'row': row.to_dict(),
                                'issue': f'Inconsistent date format in {col}: "{row[col]}" ({format_type} format)',
                                'severity': 'HIGH',
                                'column': col,
                                'format': format_type,
                                'issue_type': 'inconsistent_formats'
                            })
        
        # Number format inconsistencies (currency, thousands separators, etc.)
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(30)
                
                # Only flag as inconsistent if DIFFERENT formatting patterns exist
                currency_count = sample_values.str.contains(r'[$€£¥]', regex=True).sum()
                plain_number_count = sample_values.str.match(r'^\d+\.?\d*$').sum()
                text_number_count = sample_values.str.contains(r'\b(thousand|million|billion)\b', case=False, regex=True).sum()
                
                total_valid = len(sample_values)
                
                # Only report as mixed if we have significant amounts of different formats
                # (not just consistent formatting with multiple elements like "$1,234.56")
                format_diversity = 0
                if currency_count > 0 and currency_count < total_valid * 0.8:
                    format_diversity += 1
                if plain_number_count > 0 and plain_number_count < total_valid * 0.8:
                    format_diversity += 1
                if text_number_count > 0 and text_number_count < total_valid * 0.8:
                    format_diversity += 1
                
                # Only flag if we have 2+ different formats, each appearing in significant amounts
                if format_diversity >= 2:
                    # Get examples of truly different formats
                    if currency_count > 0 and currency_count < total_valid * 0.9:
                        currency_mask = df[col].astype(str).str.contains(r'[$€£¥]', regex=True, na=False)
                        samples = df[currency_mask].head(1)
                        for _, row in samples.iterrows():
                            inconsistent_rows.append({
                                'row': row.to_dict(),
                                'issue': f'Mixed number formats in {col}: currency format "{row[col]}" (appears in {currency_count}/{total_valid} rows)',
                                'severity': 'MEDIUM',
                                'column': col,
                                'issue_type': 'number_format_mix'
                            })
                    
                    if plain_number_count > 0 and plain_number_count < total_valid * 0.9:
                        plain_mask = df[col].astype(str).str.match(r'^\d+\.?\d*$', na=False)
                        samples = df[plain_mask].head(1)
                        for _, row in samples.iterrows():
                            inconsistent_rows.append({
                                'row': row.to_dict(),
                                'issue': f'Mixed number formats in {col}: plain number "{row[col]}" (appears in {plain_number_count}/{total_valid} rows)',
                                'severity': 'MEDIUM',
                                'column': col,
                                'issue_type': 'number_format_mix'
                            })
                    
                    if text_number_count > 0 and text_number_count < total_valid * 0.9:
                        text_mask = df[col].astype(str).str.contains(r'\b(thousand|million|billion)\b', case=False, regex=True, na=False)
                        samples = df[text_mask].head(1)
                        for _, row in samples.iterrows():
                            inconsistent_rows.append({
                                'row': row.to_dict(),
                                'issue': f'Mixed number formats in {col}: text format "{row[col]}" (appears in {text_number_count}/{total_valid} rows)',
                                'severity': 'MEDIUM',
                                'column': col,
                                'issue_type': 'number_format_mix'
                            })
        
        return inconsistent_rows[:15]
    
    def get_missing_pattern_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find informative missing value patterns"""
        pattern_rows = []
        
        # 1. Rows with multiple missing values
        missing_count = df.isnull().sum(axis=1)
        high_missing_threshold = max(2, len(df.columns) * 0.3)
        high_missing = df[missing_count >= high_missing_threshold]
        
        if not high_missing.empty:
            samples = high_missing.head(3)
            for idx, row in samples.iterrows():
                pattern_rows.append({
                    'row': row.to_dict(),
                    'issue': f'Row has {missing_count[idx]} out of {len(df.columns)} columns missing',
                    'severity': 'MEDIUM',
                    'missing_count': int(missing_count[idx]),
                    'issue_type': 'high_missing_row'
                })
        
        # 2. Correlated missing patterns
        for col1 in df.columns[:10]:  # Limit to avoid too many combinations
            for col2 in df.columns[:10]:
                if col1 != col2:
                    both_null = df[df[col1].isnull() & df[col2].isnull()]
                    if len(both_null) > len(df) * 0.2:  # >20% have both null
                        if not both_null.empty:
                            sample_row = both_null.head(1).iloc[0]
                            pattern_rows.append({
                                'row': sample_row.to_dict(),
                                'issue': f'{col1} and {col2} are both missing in {len(both_null)}/{len(df)} rows',
                                'severity': 'MEDIUM',
                                'pattern': 'correlated_missing',
                                'columns': [col1, col2],
                                'issue_type': 'correlated_missing'
                            })
        
        # 3. Blank/empty strings that should be nulls
        for col in df.columns:
            if df[col].dtype == 'object':
                blank_mask = df[col].astype(str).str.strip().isin(['', ' ', '  ', '   '])
                if blank_mask.any():
                    samples = df[blank_mask].head(2)
                    for _, row in samples.iterrows():
                        pattern_rows.append({
                            'row': row.to_dict(),
                            'issue': f'Blank/empty string in {col} should be null',
                            'severity': 'MEDIUM',
                            'column': col,
                            'issue_type': 'blank_strings'
                        })
        
        return pattern_rows[:10]
    
    def get_outlier_rows(self, df: pd.DataFrame, profile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find statistical outliers"""
        outlier_rows = []
        
        outlier_info = profile_data.get('outliers', {})
        
        for col, info in outlier_info.items():
            if info.get('outlier_percentage', 0) > 5:  # More than 5% are outliers
                if col in df.columns:
                    series = df[col].dropna()
                    if len(series) == 0:
                        continue
                        
                    lower_bound = info.get('lower_bound')
                    upper_bound = info.get('upper_bound')
                    
                    # Get extreme outliers - only for numeric columns
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if numeric_series.notna().any():  # Has numeric values
                        # Create a temporary dataframe with numeric values for sorting
                        df_with_numeric = df.copy()
                        df_with_numeric[col + '_numeric'] = numeric_series
                        
                        if lower_bound is not None:
                            outlier_mask = numeric_series < lower_bound
                            if outlier_mask.any():
                                low_outliers = df_with_numeric[outlier_mask].nsmallest(1, col + '_numeric')
                                for idx, row in low_outliers.iterrows():
                                    outlier_rows.append({
                                        'row': {k: v for k, v in row.to_dict().items() if not k.endswith('_numeric')},
                                        'issue': f'{col} extreme low outlier: {row[col]} (normal range: [{lower_bound:.2f}, {upper_bound:.2f}])',
                                        'severity': 'LOW',
                                        'column': col,
                                        'value': float(numeric_series.loc[idx]) if pd.notna(numeric_series.loc[idx]) else None,
                                        'bounds': [float(lower_bound), float(upper_bound)],
                                        'issue_type': 'statistical_outlier'
                                    })
                        
                        if upper_bound is not None:
                            outlier_mask = numeric_series > upper_bound
                            if outlier_mask.any():
                                high_outliers = df_with_numeric[outlier_mask].nlargest(1, col + '_numeric')
                                for idx, row in high_outliers.iterrows():
                                    outlier_rows.append({
                                        'row': {k: v for k, v in row.to_dict().items() if not k.endswith('_numeric')},
                                        'issue': f'{col} extreme high outlier: {row[col]} (normal range: [{lower_bound:.2f}, {upper_bound:.2f}])',
                                        'severity': 'LOW',
                                        'column': col,
                                        'value': float(numeric_series.loc[idx]) if pd.notna(numeric_series.loc[idx]) else None,
                                        'bounds': [float(lower_bound), float(upper_bound)],
                                        'issue_type': 'statistical_outlier'
                                    })
        
        return outlier_rows[:10]
    
    def get_baseline_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Select clean rows for comparison"""
        # Find rows with no obvious issues
        complete_rows = df.dropna()
        
        if complete_rows.empty:
            # If no complete rows, get least problematic ones
            missing_count = df.isnull().sum(axis=1)
            best_rows = df[missing_count == missing_count.min()]
            sample_size = min(3, len(best_rows))
            if sample_size > 0:
                baseline = best_rows.sample(n=sample_size, random_state=42)
            else:
                return []
        else:
            sample_size = min(5, len(complete_rows))
            baseline = complete_rows.sample(n=sample_size, random_state=42)
        
        return [
            {
                'row': row.to_dict(),
                'issue': 'BASELINE (relatively clean row for comparison)',
                'severity': 'NONE',
                'issue_type': 'baseline'
            }
            for _, row in baseline.iterrows()
        ]
    
    def deduplicate_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove redundant samples showing the same issue"""
        seen_patterns = set()
        unique_samples = []
        
        for sample in samples:
            # Create fingerprint based on issue type and affected columns
            issue_type = sample.get('issue_type', 'unknown')
            column = sample.get('column', '')
            severity = sample.get('severity', 'unknown')
            
            pattern = f"{issue_type}_{column}_{severity}"
            
            if pattern not in seen_patterns:
                seen_patterns.add(pattern)
                unique_samples.append(sample)
            elif len([s for s in unique_samples if s.get('issue_type') == issue_type]) < 3:
                # Allow up to 3 examples per issue type
                unique_samples.append(sample)
        
        return unique_samples
    
    def balance_by_severity(self, samples: List[Dict[str, Any]], max_total: int) -> List[Dict[str, Any]]:
        """Ensure representation across severity levels"""
        severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NONE']
        
        balanced = []
        remaining = max_total
        
        for severity in severity_order:
            severity_samples = [s for s in samples if s.get('severity') == severity]
            
            if severity == 'CRITICAL':
                take = min(len(severity_samples), max(1, remaining // 2))  # At least 50% for critical
            elif severity == 'HIGH':
                take = min(len(severity_samples), max(1, remaining // 3))  # 33% for high
            elif severity == 'MEDIUM':
                take = min(len(severity_samples), max(1, remaining // 4))  # 25% for medium
            else:
                take = min(len(severity_samples), remaining)  # Remainder
            
            balanced.extend(severity_samples[:take])
            remaining -= take
            
            if remaining <= 0:
                break
        
        return balanced[:max_total]
    
    def format_for_llm(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format samples for LLM prompt"""
        formatted = {
            'total_samples': len(samples),
            'by_severity': {},
            'summary': {
                'critical_count': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0,
                'baseline_count': 0
            }
        }
        
        for sample in samples:
            severity = sample.get('severity', 'UNKNOWN')
            if severity not in formatted['by_severity']:
                formatted['by_severity'][severity] = []
            
            formatted['by_severity'][severity].append({
                'row_data': sample['row'],
                'issue_description': sample['issue'],
                'additional_info': {k: v for k, v in sample.items() 
                                   if k not in ['row', 'issue', 'severity']}
            })
            
            # Update summary counts
            if severity == 'CRITICAL':
                formatted['summary']['critical_count'] += 1
            elif severity == 'HIGH':
                formatted['summary']['high_count'] += 1
            elif severity == 'MEDIUM':
                formatted['summary']['medium_count'] += 1
            elif severity == 'LOW':
                formatted['summary']['low_count'] += 1
            elif severity == 'NONE':
                formatted['summary']['baseline_count'] += 1
        
        return formatted
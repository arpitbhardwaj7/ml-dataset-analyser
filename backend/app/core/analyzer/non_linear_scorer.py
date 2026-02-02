"""
Non-linear quality scoring with impact-based penalties
Applies graduated penalties based on ML impact and issue combinations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import math


class NonLinearQualityScorer:
    """Applies non-linear penalties based on ML impact and issue combinations"""
    
    def __init__(self):
        self.base_score = 100
        
        # Issue impact weights (how much this affects ML models)
        self.impact_weights = {
            'mixed_data_types': 0.95,        # Extremely high impact - blocks training
            'semantic_nulls': 0.85,          # High impact - data leakage/bias
            'inconsistent_formats': 0.75,    # High impact - parsing errors
            'high_missing_percentage': 0.80, # High impact - reduces model quality
            'correlated_missing': 0.70,      # Medium-high impact - bias patterns
            'outliers_extreme': 0.65,        # Medium impact - model robustness
            'duplicates': 0.60,              # Medium impact - overfitting
            'low_cardinality': 0.50,         # Lower impact - reduces information
            'encoding_issues': 0.40,         # Lower impact - preprocessing needed
            'minor_inconsistencies': 0.30    # Low impact - cosmetic issues
        }
        
        # Severity multipliers
        self.severity_multipliers = {
            'CRITICAL': 1.0,
            'HIGH': 0.8,
            'MEDIUM': 0.6,
            'LOW': 0.4,
            'MINOR': 0.2
        }
        
        # Coverage thresholds (what percentage of data affected triggers penalty)
        self.coverage_thresholds = {
            'widespread': 0.20,   # >20% of data affected
            'moderate': 0.10,     # 10-20% affected
            'localized': 0.05     # 5-10% affected
        }
    
    def calculate_non_linear_score(self, 
                                   deterministic_score: float,
                                   llm_findings: Dict[str, Any],
                                   validation_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Apply non-linear penalties to the deterministic score
        
        Args:
            deterministic_score: Base score from deterministic analysis
            llm_findings: LLM-identified issues
            validation_results: Validated findings from LLM validator
            
        Returns:
            Tuple of (final_score, penalty_metadata)
        """
        
        penalties = []
        total_penalty = 0.0
        
        # Extract verified issues
        verified_issues = self._extract_verified_issues(llm_findings, validation_results)
        
        # Apply individual issue penalties
        for issue in verified_issues:
            penalty_info = self._calculate_issue_penalty(issue)
            penalties.append(penalty_info)
            total_penalty += penalty_info['penalty_points']
        
        # Apply combination penalties (issues that compound each other)
        combination_penalty = self._calculate_combination_penalties(verified_issues)
        if combination_penalty > 0:
            penalties.append({
                'type': 'combination_penalty',
                'penalty_points': combination_penalty,
                'description': f'Issues compound each other (additional {combination_penalty:.1f} points)'
            })
            total_penalty += combination_penalty
        
        # Apply coverage penalties (widespread issues are worse)
        coverage_penalty = self._calculate_coverage_penalties(verified_issues)
        if coverage_penalty > 0:
            penalties.append({
                'type': 'coverage_penalty', 
                'penalty_points': coverage_penalty,
                'description': f'Widespread data quality issues (additional {coverage_penalty:.1f} points)'
            })
            total_penalty += coverage_penalty
        
        # Apply non-linear scaling (more issues = exponentially worse)
        if len(verified_issues) > 3:
            scaling_penalty = self._calculate_scaling_penalty(len(verified_issues), total_penalty)
            penalties.append({
                'type': 'scaling_penalty',
                'penalty_points': scaling_penalty,
                'description': f'Multiple quality issues compound exponentially (+{scaling_penalty:.1f} points)'
            })
            total_penalty += scaling_penalty
        
        # Calculate final score with floor
        final_score = max(10, deterministic_score - total_penalty)  # Floor at 10
        
        penalty_metadata = {
            'original_score': deterministic_score,
            'total_penalty': total_penalty,
            'final_score': final_score,
            'penalty_breakdown': penalties,
            'verified_issues_count': len(verified_issues),
            'penalty_method': 'non_linear',
            'grade': self._calculate_grade(final_score)
        }
        
        return final_score, penalty_metadata
    
    def _extract_verified_issues(self, llm_findings: Dict[str, Any], validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract only verified issues from LLM findings"""
        verified_issues = []
        
        # Get blocking issues that were verified
        blocking_issues = validation_results.get('blocking_issues', [])
        for issue in blocking_issues:
            if issue.get('validation_status') == 'VERIFIED':
                verified_issues.append(issue)
        
        # Add any other verified issue categories
        for category in ['data_quality_issues', 'consistency_issues', 'format_issues']:
            issues = validation_results.get(category, [])
            for issue in issues:
                if issue.get('validation_status') == 'VERIFIED':
                    verified_issues.append(issue)
        
        return verified_issues
    
    def _calculate_issue_penalty(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate penalty for a single issue"""
        
        issue_type = issue.get('issue_type', 'unknown')
        severity = issue.get('severity', 'MEDIUM')
        affected_columns = issue.get('affected_columns', [])
        
        # Get base impact weight
        base_impact = self.impact_weights.get(issue_type, 0.5)
        
        # Apply severity multiplier
        severity_mult = self.severity_multipliers.get(severity, 0.6)
        
        # Column count multiplier (more columns affected = worse)
        column_mult = min(1.0 + (len(affected_columns) - 1) * 0.2, 2.0)  # Cap at 2x
        
        # Calculate penalty points
        penalty_points = base_impact * severity_mult * column_mult * 20  # Scale to reasonable range
        
        return {
            'type': 'individual_issue',
            'issue_type': issue_type,
            'severity': severity,
            'affected_columns': len(affected_columns),
            'penalty_points': penalty_points,
            'calculation': f"{base_impact:.2f} × {severity_mult:.2f} × {column_mult:.2f} × 20 = {penalty_points:.1f}",
            'description': f"{issue_type} ({severity}) affecting {len(affected_columns)} column(s): -{penalty_points:.1f} points"
        }
    
    def _calculate_combination_penalties(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate penalties for problematic issue combinations"""
        
        issue_types = [issue.get('issue_type', '') for issue in issues]
        combination_penalty = 0.0
        
        # Critical combinations that make datasets nearly unusable
        critical_combinations = [
            (['mixed_data_types', 'semantic_nulls'], 8.0),  # Both block training
            (['mixed_data_types', 'inconsistent_formats'], 6.0),  # Parsing nightmare
            (['high_missing_percentage', 'correlated_missing'], 5.0),  # Missing data patterns
        ]
        
        # Check for critical combinations
        for combo_types, penalty in critical_combinations:
            if all(issue_type in issue_types for issue_type in combo_types):
                combination_penalty += penalty
        
        # General combination penalty (3+ issues)
        if len(issues) >= 3:
            combination_penalty += 2.0  # Base combination penalty
        
        return combination_penalty
    
    def _calculate_coverage_penalties(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate penalties based on how widespread issues are"""
        
        coverage_penalty = 0.0
        
        for issue in issues:
            # Try to estimate coverage from verification details
            verification_details = issue.get('verification_details', '')
            
            # Look for coverage indicators in verification details
            if 'widespread' in verification_details.lower() or 'most' in verification_details.lower():
                coverage_penalty += 3.0
            elif 'many' in verification_details.lower() or 'several' in verification_details.lower():
                coverage_penalty += 2.0
            elif 'some' in verification_details.lower():
                coverage_penalty += 1.0
            
            # Check affected columns count as coverage proxy
            affected_columns = issue.get('affected_columns', [])
            if len(affected_columns) > 3:
                coverage_penalty += 2.0  # Many columns affected
            elif len(affected_columns) > 1:
                coverage_penalty += 1.0  # Multiple columns
        
        return min(coverage_penalty, 10.0)  # Cap coverage penalty
    
    def _calculate_scaling_penalty(self, issue_count: int, current_penalty: float) -> float:
        """Apply exponential scaling for multiple issues"""
        
        if issue_count <= 3:
            return 0.0
        
        # Exponential scaling: each additional issue beyond 3 increases penalty
        excess_issues = issue_count - 3
        scaling_factor = 1.2 ** excess_issues  # 20% compound increase per issue
        
        # Apply to a portion of current penalty 
        base_scaling_penalty = current_penalty * 0.15  # 15% of current penalty
        scaled_penalty = base_scaling_penalty * (scaling_factor - 1)
        
        return min(scaled_penalty, 15.0)  # Cap scaling penalty at 15 points
    
    def _calculate_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def get_penalty_summary(self, penalty_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable summary of penalties applied"""
        
        breakdown = penalty_metadata.get('penalty_breakdown', [])
        
        # Group penalties by type
        individual_penalties = [p for p in breakdown if p['type'] == 'individual_issue']
        combination_penalties = [p for p in breakdown if p['type'] == 'combination_penalty']
        coverage_penalties = [p for p in breakdown if p['type'] == 'coverage_penalty']  
        scaling_penalties = [p for p in breakdown if p['type'] == 'scaling_penalty']
        
        summary = {
            'total_penalty': penalty_metadata.get('total_penalty', 0),
            'final_score': penalty_metadata.get('final_score', 0),
            'grade': penalty_metadata.get('grade', 'F'),
            'penalty_categories': {
                'individual_issues': {
                    'count': len(individual_penalties),
                    'total_penalty': sum(p['penalty_points'] for p in individual_penalties),
                    'details': individual_penalties
                },
                'combination_effects': {
                    'count': len(combination_penalties),
                    'total_penalty': sum(p['penalty_points'] for p in combination_penalties),
                    'details': combination_penalties
                },
                'coverage_effects': {
                    'count': len(coverage_penalties),
                    'total_penalty': sum(p['penalty_points'] for p in coverage_penalties),
                    'details': coverage_penalties
                },
                'scaling_effects': {
                    'count': len(scaling_penalties), 
                    'total_penalty': sum(p['penalty_points'] for p in scaling_penalties),
                    'details': scaling_penalties
                }
            },
            'explanation': self._generate_explanation(penalty_metadata)
        }
        
        return summary
    
    def _generate_explanation(self, penalty_metadata: Dict[str, Any]) -> str:
        """Generate human-readable explanation of the scoring"""
        
        original = penalty_metadata.get('original_score', 100)
        final = penalty_metadata.get('final_score', 0)
        total_penalty = penalty_metadata.get('total_penalty', 0)
        issue_count = penalty_metadata.get('verified_issues_count', 0)
        
        explanation = f"Started with deterministic score of {original:.1f}. "
        
        if total_penalty > 0:
            explanation += f"Applied {total_penalty:.1f} penalty points for {issue_count} verified quality issues. "
            
            if total_penalty > 20:
                explanation += "Large penalty due to multiple critical issues compounding. "
            elif total_penalty > 10:
                explanation += "Moderate penalty due to significant quality issues. "
            else:
                explanation += "Small penalty for minor quality issues. "
        
        explanation += f"Final score: {final:.1f} (Grade: {penalty_metadata.get('grade', 'F')})"
        
        return explanation
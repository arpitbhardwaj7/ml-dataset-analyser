"""
Robust JSON parser for LLM responses with error handling and recovery
Handles partial JSON, malformed responses, and provides fallback parsing
"""

import json
import re
from typing import Dict, Any, Optional, Union, List
import logging


class RobustJSONParser:
    """Enhanced JSON parser with fallback strategies for LLM responses"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common JSON repair patterns
        self.repair_patterns = [
            # Fix trailing commas
            (r',(\s*[}\]])', r'\1'),
            # Fix missing quotes on keys
            (r'(\w+)(\s*:\s*)', r'"\1"\2'),
            # Fix single quotes to double quotes
            (r"'([^']*)'", r'"\1"'),
            # Fix unescaped quotes inside strings
            (r':"([^"]*)"([^"]*)"([^"]*)"', r':"\1\\"2\\"\3"'),
            # Fix boolean values
            (r'\bTrue\b', 'true'),
            (r'\bFalse\b', 'false'),
            (r'\bNone\b', 'null'),
        ]
    
    def parse_llm_response(self, response_text: str, expected_fields: List[str] = None) -> Dict[str, Any]:
        """
        Parse LLM JSON response with multiple fallback strategies
        
        Args:
            response_text: Raw text response from LLM
            expected_fields: List of required fields for validation
            
        Returns:
            Parsed JSON dict with fallback values if parsing fails
        """
        
        if not response_text or not response_text.strip():
            return self._create_fallback_response("Empty response", expected_fields)
        
        # Strategy 1: Direct JSON parsing
        try:
            parsed = json.loads(response_text.strip())
            if self._validate_response(parsed, expected_fields):
                return parsed
        except json.JSONDecodeError as e:
            self.logger.warning(f"Direct JSON parsing failed: {e}")
        
        # Strategy 2: Extract JSON from markdown or text
        extracted_json = self._extract_json_from_text(response_text)
        if extracted_json:
            try:
                parsed = json.loads(extracted_json)
                if self._validate_response(parsed, expected_fields):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Repair common JSON issues
        repaired_json = self._repair_json(response_text)
        try:
            parsed = json.loads(repaired_json)
            if self._validate_response(parsed, expected_fields):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Partial parsing with regex
        partial_data = self._parse_partial_json(response_text, expected_fields)
        if partial_data and len(partial_data) > 0:
            return self._fill_missing_fields(partial_data, expected_fields)
        
        # Strategy 5: Text-based extraction
        text_extracted = self._extract_from_text(response_text, expected_fields)
        if text_extracted:
            return text_extracted
        
        # Final fallback
        return self._create_fallback_response(f"All parsing strategies failed for: {response_text[:100]}...", expected_fields)
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from markdown code blocks or text"""
        
        # Look for JSON in code blocks
        json_block_patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'`([^`]*)`',
        ]
        
        for pattern in json_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                cleaned = match.strip()
                if cleaned.startswith('{') and cleaned.endswith('}'):
                    return cleaned
        
        # Look for JSON-like structures in plain text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            if len(match) > 20:  # Reasonable size for a JSON response
                return match
        
        return None
    
    def _repair_json(self, json_text: str) -> str:
        """Apply common JSON repair patterns"""
        
        repaired = json_text.strip()
        
        # Apply repair patterns
        for pattern, replacement in self.repair_patterns:
            repaired = re.sub(pattern, replacement, repaired)
        
        # Ensure it starts and ends with braces
        if not repaired.startswith('{'):
            # Look for first opening brace
            start_idx = repaired.find('{')
            if start_idx >= 0:
                repaired = repaired[start_idx:]
        
        if not repaired.endswith('}'):
            # Look for last closing brace
            end_idx = repaired.rfind('}')
            if end_idx >= 0:
                repaired = repaired[:end_idx + 1]
        
        return repaired
    
    def _parse_partial_json(self, text: str, expected_fields: List[str] = None) -> Dict[str, Any]:
        """Extract values using regex when JSON parsing fails"""
        
        partial_data = {}
        
        if not expected_fields:
            expected_fields = [
                'confidence_level', 'adjusted_quality_score', 'blocking_issues',
                'recommended_fixes', 'cleanup_effort_hours', 'ml_readiness_assessment'
            ]
        
        # Patterns for different field types
        field_patterns = {
            'string_fields': [
                'confidence_level', 'adjusted_grade', 'score_adjustment_reasoning',
                'ml_readiness_assessment'
            ],
            'numeric_fields': [
                'adjusted_quality_score', 'cleanup_effort_hours', 'total_cleanup_effort_hours'
            ],
            'array_fields': [
                'blocking_issues', 'recommended_fixes', 'biggest_concerns'
            ]
        }
        
        # Extract string fields
        for field in field_patterns['string_fields']:
            if field in expected_fields:
                value = self._extract_string_field(text, field)
                if value:
                    partial_data[field] = value
        
        # Extract numeric fields
        for field in field_patterns['numeric_fields']:
            if field in expected_fields:
                value = self._extract_numeric_field(text, field)
                if value is not None:
                    partial_data[field] = value
        
        # Extract array fields (simplified)
        for field in field_patterns['array_fields']:
            if field in expected_fields:
                value = self._extract_array_field(text, field)
                if value:
                    partial_data[field] = value
        
        return partial_data
    
    def _extract_string_field(self, text: str, field_name: str) -> Optional[str]:
        """Extract a string field value from text"""
        
        patterns = [
            rf'"{field_name}":\s*"([^"]*)"',
            rf'"{field_name}":\s*\'([^\']*)\'',
            rf'{field_name}:\s*"([^"]*)"',
            rf'{field_name}:\s*\'([^\']*)\'',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_numeric_field(self, text: str, field_name: str) -> Optional[Union[int, float]]:
        """Extract a numeric field value from text"""
        
        patterns = [
            rf'"{field_name}":\s*(\d+\.?\d*)',
            rf'{field_name}:\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                try:
                    return float(value_str) if '.' in value_str else int(value_str)
                except ValueError:
                    continue
        
        return None
    
    def _extract_array_field(self, text: str, field_name: str) -> Optional[List[Any]]:
        """Extract an array field (simplified - return empty list for now)"""
        
        # For arrays, we'll return an empty list as a fallback
        # In a real implementation, you'd parse the array content
        patterns = [
            rf'"{field_name}":\s*\[[^\]]*\]',
            rf'{field_name}:\s*\[[^\]]*\]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return empty list - could be enhanced to actually parse array content
                return []
        
        return []
    
    def _extract_from_text(self, text: str, expected_fields: List[str] = None) -> Optional[Dict[str, Any]]:
        """Extract structured data from natural language text"""
        
        extracted = {}
        
        # Look for quality score mentions
        score_patterns = [
            r'quality score.*?(\d+)',
            r'score.*?(\d+)',
            r'rating.*?(\d+)',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 100:
                    extracted['adjusted_quality_score'] = score
                    break
        
        # Look for confidence mentions
        confidence_patterns = [
            r'confidence.*?(high|medium|low)',
            r'(high|medium|low).*?confidence',
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['confidence_level'] = match.group(1).upper()
                break
        
        # Look for readiness assessment
        readiness_patterns = [
            r'(ready|not ready|needs.*?work|needs.*?cleanup)',
        ]
        
        for pattern in readiness_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                readiness_text = match.group(1).lower()
                if 'ready' in readiness_text and 'not' not in readiness_text:
                    extracted['ml_readiness_assessment'] = 'Ready'
                elif 'minor' in readiness_text:
                    extracted['ml_readiness_assessment'] = 'Needs Minor Cleanup'
                elif 'major' in readiness_text:
                    extracted['ml_readiness_assessment'] = 'Needs Major Work'
                else:
                    extracted['ml_readiness_assessment'] = 'Not Ready'
                break
        
        return extracted if extracted else None
    
    def _validate_response(self, parsed_data: Dict[str, Any], expected_fields: List[str] = None) -> bool:
        """Validate that the parsed response has required structure"""
        
        if not isinstance(parsed_data, dict):
            return False
        
        if not expected_fields:
            return True
        
        # Check for required fields
        missing_fields = []
        for field in expected_fields:
            if field not in parsed_data or parsed_data[field] is None:
                missing_fields.append(field)
        
        # Allow some missing fields but not all
        if len(missing_fields) > len(expected_fields) * 0.5:
            return False
        
        return True
    
    def _fill_missing_fields(self, partial_data: Dict[str, Any], expected_fields: List[str] = None) -> Dict[str, Any]:
        """Fill in missing fields with reasonable defaults"""
        
        if not expected_fields:
            return partial_data
        
        defaults = {
            'confidence_level': 'LOW',
            'adjusted_quality_score': 50,
            'adjusted_grade': 'D',
            'score_adjustment_reasoning': 'Unable to parse detailed reasoning from response',
            'blocking_issues': [],
            'recommended_fixes': [],
            'total_cleanup_effort_hours': 'Unknown',
            'ml_readiness_assessment': 'Needs Review',
            'biggest_concerns': [],
        }
        
        filled_data = partial_data.copy()
        
        for field in expected_fields:
            if field not in filled_data or filled_data[field] is None:
                filled_data[field] = defaults.get(field, 'Unknown')
        
        return filled_data
    
    def _create_fallback_response(self, error_message: str, expected_fields: List[str] = None) -> Dict[str, Any]:
        """Create a fallback response when all parsing fails"""
        
        fallback = {
            'error': f'JSON parsing failed: {error_message}',
            'confidence_level': 'VERY_LOW',
            'adjusted_quality_score': 50,
            'adjusted_grade': 'Unknown',
            'score_adjustment_reasoning': 'Could not parse LLM response - using fallback values',
            'blocking_issues': [],
            'recommended_fixes': [],
            'total_cleanup_effort_hours': 0,
            'ml_readiness_assessment': 'Needs Manual Review',
            'biggest_concerns': ['LLM response parsing failed'],
            'parsing_status': 'FAILED',
            'fallback_used': True
        }
        
        if expected_fields:
            # Only include expected fields plus error info
            filtered_fallback = {key: fallback[key] for key in expected_fields if key in fallback}
            filtered_fallback.update({
                'error': fallback['error'],
                'parsing_status': fallback['parsing_status'],
                'fallback_used': fallback['fallback_used']
            })
            return filtered_fallback
        
        return fallback
    
    def validate_and_clean_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Final validation and cleaning of parsed response"""
        
        cleaned = response.copy()
        
        # Ensure numeric fields are numeric
        numeric_fields = ['adjusted_quality_score', 'cleanup_effort_hours', 'total_cleanup_effort_hours']
        for field in numeric_fields:
            if field in cleaned:
                try:
                    if isinstance(cleaned[field], str):
                        cleaned[field] = float(cleaned[field])
                    # Clamp quality scores to 0-100
                    if field == 'adjusted_quality_score':
                        cleaned[field] = max(0, min(100, cleaned[field]))
                except (ValueError, TypeError):
                    cleaned[field] = 50 if field == 'adjusted_quality_score' else 0
        
        # Ensure array fields are arrays
        array_fields = ['blocking_issues', 'recommended_fixes', 'biggest_concerns']
        for field in array_fields:
            if field in cleaned and not isinstance(cleaned[field], list):
                cleaned[field] = []
        
        # Validate confidence level
        if 'confidence_level' in cleaned:
            valid_levels = ['HIGH', 'MEDIUM', 'LOW', 'VERY_HIGH', 'VERY_LOW']
            if cleaned['confidence_level'] not in valid_levels:
                cleaned['confidence_level'] = 'LOW'
        
        return cleaned
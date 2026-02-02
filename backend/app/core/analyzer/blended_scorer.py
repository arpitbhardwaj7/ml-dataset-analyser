"""
Blended Scoring System - Combines deterministic heuristics with LLM insights
"""

from typing import Dict, Any, Optional, Tuple


class BlendedScorer:
    """
    Intelligently blends deterministic and LLM scores to provide more reliable quality assessments
    """
    
    def __init__(self):
        self.max_adjustment = 15  # Cap LLM adjustments to ±15 points
        
    def calculate_final_score(
        self, 
        deterministic_score: float, 
        llm_score: Optional[float] = None, 
        llm_confidence: Optional[str] = None,
        validation_reliability: Optional[str] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Blend deterministic and LLM scores intelligently
        
        Args:
            deterministic_score: Score from heuristic analysis (0-100)
            llm_score: Optional LLM-provided score (0-100)
            llm_confidence: Optional confidence level ("HIGH", "MEDIUM", "LOW")
            validation_reliability: Optional validation result ("HIGH", "MEDIUM", "LOW")
        
        Returns:
            Tuple of (final_score, scoring_metadata)
        """
        if llm_score is None:
            return deterministic_score, {
                "method": "deterministic_only",
                "final_score": round(deterministic_score, 2),
                "deterministic_score": deterministic_score,
                "llm_score": None,
                "blend_applied": False
            }
        
        # Calculate raw adjustment
        adjustment = llm_score - deterministic_score
        capped_adjustment = max(min(adjustment, self.max_adjustment), -self.max_adjustment)
        
        # Determine blend weight based on confidence and validation
        blend_weight = self._calculate_blend_weight(llm_confidence, validation_reliability)
        
        # Apply blended scoring
        blended_score = (
            deterministic_score * (1 - blend_weight) + 
            (deterministic_score + capped_adjustment) * blend_weight
        )
        
        final_score = round(max(0, min(100, blended_score)), 2)
        
        metadata = {
            "method": "blended",
            "deterministic_score": deterministic_score,
            "llm_score": llm_score,
            "llm_confidence": llm_confidence,
            "validation_reliability": validation_reliability,
            "raw_adjustment": round(adjustment, 2),
            "capped_adjustment": round(capped_adjustment, 2),
            "blend_weight": blend_weight,
            "blend_applied": True,
            "final_score": final_score,
            "adjustment_reason": self._get_adjustment_reason(
                adjustment, capped_adjustment, llm_confidence, validation_reliability
            )
        }
        
        return final_score, metadata
    
    def _calculate_blend_weight(
        self, 
        llm_confidence: Optional[str], 
        validation_reliability: Optional[str]
    ) -> float:
        """Calculate how much weight to give LLM score vs deterministic score"""
        
        # Base weights by confidence level
        confidence_weights = {
            "HIGH": 0.35,    # 65% deterministic, 35% LLM  
            "MEDIUM": 0.25,  # 75% deterministic, 25% LLM
            "LOW": 0.15,     # 85% deterministic, 15% LLM
            None: 0.20       # Default when no confidence provided
        }
        
        base_weight = confidence_weights.get(llm_confidence, 0.20)
        
        # Adjust based on validation reliability
        if validation_reliability == "HIGH":
            # If LLM findings are well-validated, increase trust
            weight_adjustment = 0.10
        elif validation_reliability == "LOW":
            # If LLM findings aren't validated, decrease trust
            weight_adjustment = -0.15
        else:
            # MEDIUM or None
            weight_adjustment = 0.0
        
        final_weight = max(0.05, min(0.45, base_weight + weight_adjustment))
        return round(final_weight, 3)
    
    def _get_adjustment_reason(
        self, 
        raw_adjustment: float, 
        capped_adjustment: float, 
        confidence: Optional[str],
        validation: Optional[str]
    ) -> str:
        """Generate human-readable explanation for the adjustment"""
        
        if abs(raw_adjustment) < 1:
            return "LLM analysis confirmed deterministic assessment"
        
        direction = "increased" if capped_adjustment > 0 else "decreased"
        magnitude = "significantly" if abs(capped_adjustment) > 10 else "moderately" if abs(capped_adjustment) > 5 else "slightly"
        
        confidence_text = f" (LLM confidence: {confidence})" if confidence else ""
        validation_text = f" with {validation.lower()} validation reliability" if validation else ""
        
        capped_note = f" (capped from {raw_adjustment:+.1f})" if abs(raw_adjustment) > self.max_adjustment else ""
        
        return f"LLM analysis {magnitude} {direction} score by {capped_adjustment:+.1f} points{capped_note}{confidence_text}{validation_text}"
    
    def calculate_grade_from_score(self, score: float) -> str:
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
    
    def get_scoring_summary(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the scoring process for display"""
        
        if not metadata.get("blend_applied", False):
            return {
                "primary_method": "Deterministic Analysis",
                "score_source": "Rule-based heuristics",
                "reliability": "High",
                "explanation": "Score based entirely on statistical analysis and data profiling"
            }
        
        primary_weight = 1 - metadata.get("blend_weight", 0)
        llm_weight = metadata.get("blend_weight", 0)
        
        return {
            "primary_method": "Blended Analysis",
            "score_source": f"{primary_weight*100:.0f}% deterministic, {llm_weight*100:.0f}% AI-enhanced",
            "reliability": self._assess_reliability(metadata),
            "explanation": metadata.get("adjustment_reason", "Blended scoring applied"),
            "component_scores": {
                "deterministic": metadata.get("deterministic_score"),
                "llm_suggested": metadata.get("llm_score"),
                "final_blended": metadata.get("final_score")
            }
        }
    
    def _assess_reliability(self, metadata: Dict[str, Any]) -> str:
        """Assess overall reliability of the blended score"""
        
        validation = metadata.get("validation_reliability")
        confidence = metadata.get("llm_confidence")
        adjustment_size = abs(metadata.get("capped_adjustment", 0))
        
        # High reliability conditions
        if validation == "HIGH" and confidence == "HIGH" and adjustment_size < 5:
            return "Very High"
        elif validation in ["HIGH", "MEDIUM"] and confidence in ["HIGH", "MEDIUM"]:
            return "High"
        elif validation == "LOW" or confidence == "LOW" or adjustment_size > 10:
            return "Medium"
        else:
            return "High"
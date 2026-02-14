from enum import Enum
from typing import List, Dict, Optional

class BiasType(Enum):
    GENDER = "gender"
    SOCIOECONOMIC = "socioeconomic"

class EthicalAuditor:
    def __init__(self):
        # In a real production system, these would be more extensive or model-based
        self.gender_markers = ["he", "she", "man", "woman", "guy", "girl", "his", "her"]
        self.wealth_markers = ["expensive", "buy", "invest", "rich", "cheap", "costly", "pay for", "macbook", "iphone"]
        
    def audit_advice(self, text: str) -> Dict:
        """
        Analyzes the provided text for potential biases.
        Returns a dictionary with a trust score and a list of flags.
        """
        if not text:
            return {"trust_score": 100, "is_ethically_safe": True, "flags": []}

        text_lower = text.lower()
        flags = []
        score = 100
        
        # Check Gender Neutrality
        # Simple token matching for MVP
        tokens = text_lower.replace(".", "").replace(",", "").split()
        if any(marker in tokens for marker in self.gender_markers):
            flags.append({
                "type": BiasType.GENDER.value, 
                "message": "Advice may contain gendered pronouns on assumptions. Use neutral language (they/them/student)."
            })
            score -= 10
            
        # Check Socioeconomic Assumptions (e.g., assuming user can buy expensive items)
        if any(marker in text_lower for marker in self.wealth_markers):
            flags.append({
                "type": BiasType.SOCIOECONOMIC.value,
                "message": "Advice mentions financial transactions or expensive items. Ensure free alternatives are suggested."
            })
            score -= 15
            
        return {
            # Clamp score between 0 and 100
            "trust_score": max(0, score),
            "is_ethically_safe": score >= 80,
            "flags": flags
        }

# Singleton instance for easy import
auditor = EthicalAuditor()

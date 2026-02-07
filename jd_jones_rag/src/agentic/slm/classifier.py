"""
Fast Classifier
Lightweight classifier for instant categorization tasks.
Runs locally for low latency on website/customer portal.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from collections import Counter


class ClassificationType(str, Enum):
    """Types of classification tasks."""
    INTENT = "intent"
    SENTIMENT = "sentiment"
    URGENCY = "urgency"
    CATEGORY = "category"
    LANGUAGE = "language"


@dataclass
class ClassificationResult:
    """Result of classification."""
    classification_type: ClassificationType
    predicted_class: str
    confidence: float
    all_scores: Dict[str, float]
    reasoning: str


class FastClassifier:
    """
    Fast classification for common categorization tasks.
    
    Use cases:
    - Intent classification for chatbot
    - Urgency detection for enquiries
    - Sentiment analysis for feedback
    - Category tagging for documents
    
    In production, would use a fine-tuned model like:
    - DistilBERT for classification
    - Phi-4 for general classification
    
    Currently implements keyword-based classification.
    """
    
    # Intent keywords
    INTENT_KEYWORDS = {
        "product_inquiry": [
            "product", "recommend", "suggest", "what", "which", "looking for",
            "need", "suitable", "best", "gasket", "packing", "seal"
        ],
        "pricing_request": [
            "price", "cost", "quote", "quotation", "how much", "budget",
            "discount", "pricing", "estimate"
        ],
        "technical_support": [
            "problem", "issue", "help", "not working", "failed", "error",
            "troubleshoot", "support", "fix"
        ],
        "order_status": [
            "order", "track", "tracking", "shipment", "delivery", "where",
            "status", "shipped", "arrived"
        ],
        "complaint": [
            "complaint", "unhappy", "dissatisfied", "wrong", "damaged",
            "late", "missing", "refund", "return"
        ],
        "general_inquiry": [
            "information", "about", "tell me", "explain", "how does",
            "what is", "general"
        ]
    }
    
    # Urgency indicators
    URGENCY_KEYWORDS = {
        "critical": [
            "emergency", "urgent", "immediately", "asap", "critical",
            "shutdown", "down", "failed", "safety", "hazard"
        ],
        "high": [
            "soon", "quickly", "fast", "important", "priority",
            "deadline", "time-sensitive"
        ],
        "medium": [
            "when possible", "appreciate", "would like", "planning"
        ],
        "low": [
            "no rush", "whenever", "future", "planning ahead", "just curious"
        ]
    }
    
    # Sentiment indicators
    SENTIMENT_KEYWORDS = {
        "positive": [
            "great", "excellent", "amazing", "wonderful", "thank",
            "appreciate", "happy", "satisfied", "good", "love", "perfect"
        ],
        "negative": [
            "bad", "terrible", "awful", "disappointed", "frustrated",
            "angry", "upset", "poor", "worst", "hate", "unacceptable"
        ],
        "neutral": [
            "okay", "fine", "alright", "average", "acceptable"
        ]
    }
    
    # Category keywords for JD Jones
    CATEGORY_KEYWORDS = {
        "gaskets": [
            "gasket", "sheet", "spiral wound", "ring joint", "flange"
        ],
        "packings": [
            "packing", "gland", "stem", "valve packing", "pump packing"
        ],
        "expansion_joints": [
            "expansion", "joint", "bellows", "compensator", "flexible"
        ],
        "seals": [
            "seal", "o-ring", "mechanical seal", "lip seal"
        ],
        "insulation": [
            "insulation", "thermal", "heat", "cold", "blanket"
        ]
    }
    
    def __init__(self):
        """Initialize classifier."""
        pass
    
    def classify_intent(self, text: str) -> ClassificationResult:
        """
        Classify user intent.
        
        Args:
            text: Text to classify
            
        Returns:
            ClassificationResult with intent
        """
        return self._keyword_classify(
            text=text,
            keywords=self.INTENT_KEYWORDS,
            classification_type=ClassificationType.INTENT,
            default_class="general_inquiry"
        )
    
    def classify_urgency(self, text: str) -> ClassificationResult:
        """
        Classify urgency level.
        
        Args:
            text: Text to classify
            
        Returns:
            ClassificationResult with urgency
        """
        return self._keyword_classify(
            text=text,
            keywords=self.URGENCY_KEYWORDS,
            classification_type=ClassificationType.URGENCY,
            default_class="medium"
        )
    
    def classify_sentiment(self, text: str) -> ClassificationResult:
        """
        Classify sentiment.
        
        Args:
            text: Text to classify
            
        Returns:
            ClassificationResult with sentiment
        """
        return self._keyword_classify(
            text=text,
            keywords=self.SENTIMENT_KEYWORDS,
            classification_type=ClassificationType.SENTIMENT,
            default_class="neutral"
        )
    
    def classify_category(self, text: str) -> ClassificationResult:
        """
        Classify product category.
        
        Args:
            text: Text to classify
            
        Returns:
            ClassificationResult with category
        """
        return self._keyword_classify(
            text=text,
            keywords=self.CATEGORY_KEYWORDS,
            classification_type=ClassificationType.CATEGORY,
            default_class="general"
        )
    
    def _keyword_classify(
        self,
        text: str,
        keywords: Dict[str, List[str]],
        classification_type: ClassificationType,
        default_class: str
    ) -> ClassificationResult:
        """Perform keyword-based classification."""
        text_lower = text.lower()
        scores = {}
        
        for category, kw_list in keywords.items():
            score = 0
            matched_keywords = []
            
            for kw in kw_list:
                if kw.lower() in text_lower:
                    score += 1
                    matched_keywords.append(kw)
            
            # Normalize by keyword count
            scores[category] = score / len(kw_list) if kw_list else 0
        
        # Find best match
        if not scores or max(scores.values()) == 0:
            return ClassificationResult(
                classification_type=classification_type,
                predicted_class=default_class,
                confidence=0.3,
                all_scores=scores,
                reasoning=f"No keywords matched, defaulting to {default_class}"
            )
        
        best_class = max(scores, key=scores.get)
        best_score = scores[best_class]
        
        # Calculate confidence
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5
        
        # Boost confidence if score is significantly higher
        second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        if best_score > second_best * 1.5:
            confidence = min(confidence + 0.2, 1.0)
        
        return ClassificationResult(
            classification_type=classification_type,
            predicted_class=best_class,
            confidence=confidence,
            all_scores=scores,
            reasoning=f"Matched keywords for {best_class}"
        )
    
    def multi_classify(self, text: str) -> Dict[str, ClassificationResult]:
        """
        Perform all classification types on text.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary of classification results
        """
        return {
            "intent": self.classify_intent(text),
            "urgency": self.classify_urgency(text),
            "sentiment": self.classify_sentiment(text),
            "category": self.classify_category(text)
        }
    
    def quick_route(self, text: str) -> Dict[str, Any]:
        """
        Quick routing decision for incoming messages.
        
        Args:
            text: Incoming message
            
        Returns:
            Routing recommendation
        """
        intent = self.classify_intent(text)
        urgency = self.classify_urgency(text)
        sentiment = self.classify_sentiment(text)
        
        # Determine routing
        routing = {
            "queue": "general",
            "priority": "normal",
            "auto_respond": False,
            "escalate": False
        }
        
        # Set queue based on intent
        queue_mapping = {
            "product_inquiry": "sales",
            "pricing_request": "sales",
            "technical_support": "technical",
            "order_status": "customer_service",
            "complaint": "customer_service"
        }
        routing["queue"] = queue_mapping.get(intent.predicted_class, "general")
        
        # Set priority based on urgency
        priority_mapping = {
            "critical": "urgent",
            "high": "high",
            "medium": "normal",
            "low": "low"
        }
        routing["priority"] = priority_mapping.get(urgency.predicted_class, "normal")
        
        # Check for escalation triggers
        if urgency.predicted_class == "critical":
            routing["escalate"] = True
        if sentiment.predicted_class == "negative" and intent.predicted_class == "complaint":
            routing["escalate"] = True
        
        # Check for auto-respond eligibility
        if intent.predicted_class in ["order_status", "general_inquiry"] and urgency.predicted_class in ["low", "medium"]:
            routing["auto_respond"] = True
        
        return {
            "routing": routing,
            "classifications": {
                "intent": intent.predicted_class,
                "urgency": urgency.predicted_class,
                "sentiment": sentiment.predicted_class
            },
            "confidence": min(intent.confidence, urgency.confidence)
        }

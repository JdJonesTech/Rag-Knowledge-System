"""
Intent Classifier for External Customer Portal.
Classifies customer queries into predefined intents for decision tree navigation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings


class CustomerIntent(str, Enum):
    """Customer intent categories."""
    PRODUCT_INFO = "product_info"
    PRICING_QUOTE = "pricing_quote"
    ORDER_STATUS = "order_status"
    TECHNICAL_SUPPORT = "technical_support"
    RETURNS_WARRANTY = "returns_warranty"
    GENERAL_INQUIRY = "general_inquiry"
    COMPLAINT = "complaint"
    PARTNERSHIP = "partnership"
    CONTACT_SALES = "contact_sales"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of intent classification."""
    primary_intent: CustomerIntent
    confidence: float
    secondary_intents: List[Tuple[CustomerIntent, float]]
    entities: Dict[str, Any]
    raw_query: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_intent": self.primary_intent.value,
            "confidence": self.confidence,
            "secondary_intents": [
                {"intent": i.value, "confidence": c} 
                for i, c in self.secondary_intents
            ],
            "entities": self.entities,
            "raw_query": self.raw_query
        }


class IntentClassifier:
    """
    Classifies customer queries into intents.
    Uses LLM for sophisticated understanding with fallback patterns.
    """
    
    # Keyword patterns for rule-based fallback
    INTENT_KEYWORDS = {
        CustomerIntent.PRODUCT_INFO: [
            "product", "specification", "spec", "features", "catalog",
            "dimensions", "material", "what is", "tell me about"
        ],
        CustomerIntent.PRICING_QUOTE: [
            "price", "pricing", "quote", "cost", "how much", "discount",
            "bulk", "wholesale", "estimate"
        ],
        CustomerIntent.ORDER_STATUS: [
            "order", "tracking", "shipment", "delivery", "shipped",
            "where is my", "status", "when will"
        ],
        CustomerIntent.TECHNICAL_SUPPORT: [
            "help", "support", "issue", "problem", "not working",
            "broken", "error", "troubleshoot", "fix"
        ],
        CustomerIntent.RETURNS_WARRANTY: [
            "return", "warranty", "refund", "exchange", "replace",
            "defective", "damaged", "guarantee"
        ],
        CustomerIntent.COMPLAINT: [
            "complaint", "unhappy", "disappointed", "terrible",
            "worst", "unacceptable", "frustrated", "angry"
        ],
        CustomerIntent.PARTNERSHIP: [
            "partner", "partnership", "distributor", "reseller",
            "collaboration", "business opportunity"
        ],
        CustomerIntent.CONTACT_SALES: [
            "speak to", "talk to", "contact", "representative",
            "sales team", "call me", "reach out"
        ],
    }
    
    CLASSIFICATION_PROMPT = """You are a customer intent classifier for JD Jones, a manufacturing company.

Classify the customer's query into one of these intents:
- product_info: Questions about products, specifications, features
- pricing_quote: Price inquiries, quote requests, bulk pricing
- order_status: Order tracking, delivery status, shipment inquiries
- technical_support: Technical issues, troubleshooting, product problems
- returns_warranty: Returns, refunds, warranty claims, exchanges
- general_inquiry: General questions not fitting other categories
- complaint: Customer complaints, dissatisfaction
- partnership: Business partnerships, distributor inquiries
- contact_sales: Requests to speak with sales representatives
- unknown: Cannot determine intent

Also extract any relevant entities (product names, order numbers, etc.)

Respond in JSON format:
{
    "primary_intent": "intent_name",
    "confidence": 0.0-1.0,
    "secondary_intents": [{"intent": "name", "confidence": 0.0-1.0}],
    "entities": {"product": "...", "order_number": "...", etc}
}

Customer Query: """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize intent classifier.
        
        Args:
            use_llm: Whether to use LLM for classification (vs rule-based)
        """
        self.use_llm = use_llm
        
        if use_llm:
            from src.config.settings import get_llm
            self.llm = get_llm(temperature=0)
    
    async def classify(self, query: str) -> ClassificationResult:
        """
        Classify a customer query.
        
        Args:
            query: Customer's input text
            
        Returns:
            ClassificationResult with intent and entities
        """
        if self.use_llm:
            try:
                return await self._classify_with_llm(query)
            except Exception as e:
                logger.error(f"LLM classification failed: {e}, falling back to rules")
                return self._classify_with_rules(query)
        else:
            return self._classify_with_rules(query)
    
    async def _classify_with_llm(self, query: str) -> ClassificationResult:
        """Classify using LLM."""
        messages = [
            SystemMessage(content=self.CLASSIFICATION_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Parse JSON response
        import json
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            primary_intent = CustomerIntent(result.get("primary_intent", "unknown"))
            confidence = float(result.get("confidence", 0.8))
            
            secondary = []
            for sec in result.get("secondary_intents", []):
                try:
                    intent = CustomerIntent(sec.get("intent"))
                    conf = float(sec.get("confidence", 0.5))
                    secondary.append((intent, conf))
                except ValueError:
                    continue
            
            entities = result.get("entities", {})
            
            return ClassificationResult(
                primary_intent=primary_intent,
                confidence=confidence,
                secondary_intents=secondary,
                entities=entities,
                raw_query=query
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback to rule-based on parse error
            return self._classify_with_rules(query)
    
    def _classify_with_rules(self, query: str) -> ClassificationResult:
        """Classify using keyword matching rules."""
        query_lower = query.lower()
        
        # Score each intent
        scores: Dict[CustomerIntent, float] = {}
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            scores[intent] = score
        
        # Find best match
        if max(scores.values()) > 0:
            sorted_intents = sorted(
                scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            primary_intent = sorted_intents[0][0]
            max_score = sorted_intents[0][1]
            confidence = min(0.9, 0.5 + (max_score * 0.1))
            
            secondary = [
                (intent, min(0.8, 0.3 + (score * 0.1)))
                for intent, score in sorted_intents[1:4]
                if score > 0
            ]
        else:
            primary_intent = CustomerIntent.GENERAL_INQUIRY
            confidence = 0.5
            secondary = []
        
        # Extract basic entities
        entities = self._extract_entities(query)
        
        return ClassificationResult(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary,
            entities=entities,
            raw_query=query
        )
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract basic entities from query."""
        entities = {}
        
        # Order number pattern (e.g., ORD-12345, #12345)
        import re
        order_pattern = r'(?:ORD-?|#)(\d{4,10})'
        order_match = re.search(order_pattern, query, re.IGNORECASE)
        if order_match:
            entities["order_number"] = order_match.group(0)
        
        # Email pattern
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        email_match = re.search(email_pattern, query)
        if email_match:
            entities["email"] = email_match.group(0)
        
        # Phone pattern
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_match = re.search(phone_pattern, query)
        if phone_match:
            entities["phone"] = phone_match.group(0)
        
        return entities
    
    def get_intent_description(self, intent: CustomerIntent) -> str:
        """Get human-readable description of an intent."""
        descriptions = {
            CustomerIntent.PRODUCT_INFO: "Product Information",
            CustomerIntent.PRICING_QUOTE: "Pricing & Quote Request",
            CustomerIntent.ORDER_STATUS: "Order Status Tracking",
            CustomerIntent.TECHNICAL_SUPPORT: "Technical Support",
            CustomerIntent.RETURNS_WARRANTY: "Returns & Warranty",
            CustomerIntent.GENERAL_INQUIRY: "General Inquiry",
            CustomerIntent.COMPLAINT: "Customer Complaint",
            CustomerIntent.PARTNERSHIP: "Partnership Inquiry",
            CustomerIntent.CONTACT_SALES: "Contact Sales Team",
            CustomerIntent.UNKNOWN: "Unknown Intent",
        }
        return descriptions.get(intent, "Unknown")

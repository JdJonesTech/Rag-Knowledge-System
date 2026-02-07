"""
FAQ Prompt Cache
Caches frequently asked question responses for instant retrieval.
Reduces LLM API calls and improves response time for common queries.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class FAQEntry:
    """A cached FAQ entry."""
    question: str
    question_normalized: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    category: str = "general"
    hit_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "category": self.category,
            "hit_count": self.hit_count,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class FAQPromptCache:
    """
    Prompt cache for frequently asked questions.
    
    Features:
    - Pre-loaded FAQ responses
    - Fuzzy matching for similar questions
    - Category-based organization
    - Auto-learning from frequent queries
    - TTL-based refresh
    """
    
    # Common JD Jones FAQ patterns
    FAQ_PATTERNS = {
        # Product-related FAQs
        "what_is_product": [
            r"what\s+is\s+(NA\s*\d+|NJ\s*\d+|PACMAAN|FLEXSEAL)",
            r"tell\s+me\s+about\s+(NA\s*\d+|NJ\s*\d+)",
            r"describe\s+(NA\s*\d+|NJ\s*\d+)"
        ],
        "product_specs": [
            r"(specifications?|specs)\s+(for|of)\s+(NA\s*\d+|NJ\s*\d+)",
            r"(temperature|pressure)\s+(rating|range)\s+(for|of)\s+(NA\s*\d+|NJ\s*\d+)"
        ],
        "product_applications": [
            r"(applications?|uses?)\s+(for|of)\s+(NA\s*\d+|NJ\s*\d+)",
            r"(what|where)\s+(can|is|are)\s+(NA\s*\d+|NJ\s*\d+)\s+used"
        ],
        # Certification FAQs
        "certifications": [
            r"(API\s*622|API\s*624|API\s*6A|FDA|ATEX)\s+(certified|certification|compliant)",
            r"which\s+products?\s+(are|have)\s+(API|FDA|ATEX)"
        ],
        # Company FAQs
        "company_info": [
            r"(contact|reach|call)\s+(jd\s*jones|you)",
            r"(where|location)\s+(is|are)\s+jd\s*jones",
            r"(phone|email|address)\s+(of\s+)?jd\s*jones"
        ],
        # Order/Quote FAQs
        "ordering": [
            r"(how|can|do|i)\s+(to\s+)?(order|purchase|buy)",
            r"(get|request)\s+(a\s+)?(quote|quotation|pricing)"
        ]
    }
    
    # Pre-defined FAQ responses (can be extended via config)
    PREDEFINED_FAQS = {
        "company_contact": {
            "patterns": ["contact jd jones", "jd jones phone", "jd jones email", "how to reach jd jones"],
            "answer": """**JD Jones Manufacturing Contact Information:**

**Address:** JD Jones Manufacturing, Industrial Area, UK
**Email:** info@jdjones.com
**Phone:** Contact our sales team for immediate assistance
**Website:** www.jdjones.com

For technical inquiries, please use our online contact form or reach out to your regional sales representative.""",
            "category": "company_info"
        },
        "api_622_explained": {
            "patterns": ["what is api 622", "api 622 standard", "api 622 certification"],
            "answer": """**API 622 - Stem Packing Testing Standard:**

API 622 is a testing standard for valve stem packing systems that evaluates:
- Fugitive emission performance
- Thermal cycling resistance (500+ cycles)
- Live loading capability
- Fire-safe performance

JD Jones products that are API 622 certified include our PACMAAN series and other high-performance packing solutions. These products are tested and certified to meet the strictest fugitive emission requirements.""",
            "category": "certifications"
        },
        "api_624_explained": {
            "patterns": ["what is api 624", "api 624 standard", "api 624 certification"],
            "answer": """**API 624 - Rising Stem Valve Testing Standard:**

API 624 is a testing standard for type testing rising stem ball, gate, and globe valves that covers:
- Type testing per API 641
- Mechanical endurance testing
- Temperature cycling
- Emission measurement

JD Jones provides packing solutions specifically designed to help valves meet API 624 certification requirements.""",
            "category": "certifications"
        },
        "product_categories": {
            "patterns": ["what products does jd jones offer", "jd jones product range", "types of products"],
            "answer": """**JD Jones Manufacturing Product Range:**

**Valve Packing Systems:**
- PACMAAN braided packings (graphite, PTFE, aramid)
- Die-formed rings
- Live loading systems

**Sealing Solutions:**
- FLEXSEAL gaskets
- Flange gaskets
- Spiral wound gaskets

**Specialty Products:**
- High-temperature seals
- Chemical-resistant packings
- Food-grade sealing solutions (FDA compliant)

Contact our team for product recommendations based on your specific application requirements.""",
            "category": "products"
        },
        "quote_request": {
            "patterns": ["get a quote", "request quote", "pricing", "how much does", "price of"],
            "answer": """**Request a Quote from JD Jones:**

To receive a quotation, please provide:
1. **Product code(s)** you're interested in
2. **Quantity** required
3. **Application details** (temperature, pressure, media)
4. **Delivery location**

You can request a quote by:
- Email: sales@jdjones.com
- Online: Visit our website quote request form
- Phone: Contact your regional sales representative

Our team typically responds within 24-48 business hours.""",
            "category": "ordering"
        }
    }
    
    def __init__(
        self,
        default_ttl_hours: int = 168,  # 1 week default
        max_cache_size: int = 500,
        fuzzy_threshold: float = 0.85
    ):
        """
        Initialize FAQ Prompt Cache.
        
        Args:
            default_ttl_hours: Default TTL for cached entries
            max_cache_size: Maximum number of cached entries
            fuzzy_threshold: Threshold for fuzzy matching (0-1)
        """
        self.default_ttl_hours = default_ttl_hours
        self.max_cache_size = max_cache_size
        self.fuzzy_threshold = fuzzy_threshold
        
        # Cache storage
        self.faq_cache: Dict[str, FAQEntry] = {}
        self.pattern_cache: Dict[str, str] = {}  # pattern_hash -> faq_key
        
        # Statistics
        self.total_hits = 0
        self.total_misses = 0
        
        # Initialize with predefined FAQs
        self._load_predefined_faqs()
    
    def _load_predefined_faqs(self) -> None:
        """Load predefined FAQ entries."""
        for faq_key, faq_data in self.PREDEFINED_FAQS.items():
            entry = FAQEntry(
                question=faq_data["patterns"][0],
                question_normalized=self._normalize_question(faq_data["patterns"][0]),
                answer=faq_data["answer"],
                category=faq_data["category"],
                confidence_score=1.0
            )
            
            self.faq_cache[faq_key] = entry
            
            # Index all patterns
            for pattern in faq_data["patterns"]:
                pattern_hash = self._hash_question(pattern)
                self.pattern_cache[pattern_hash] = faq_key
        
        logger.info(f"Loaded {len(self.PREDEFINED_FAQS)} predefined FAQs")
    
    def _normalize_question(self, question: str) -> str:
        """Normalize question for matching."""
        # Lowercase
        normalized = question.lower().strip()
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _hash_question(self, question: str) -> str:
        """Create hash for question lookup."""
        normalized = self._normalize_question(question)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _calculate_similarity(self, q1: str, q2: str) -> float:
        """Calculate similarity between two questions."""
        # Simple word overlap similarity
        words1 = set(self._normalize_question(q1).split())
        words2 = set(self._normalize_question(q2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _match_pattern(self, question: str) -> Optional[str]:
        """Match question against FAQ patterns."""
        question_lower = question.lower()
        
        for category, patterns in self.FAQ_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    return category
        
        return None
    
    def get(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Get cached FAQ response.
        
        Args:
            question: User question
            
        Returns:
            Cached response or None
        """
        # Try exact match first
        question_hash = self._hash_question(question)
        
        if question_hash in self.pattern_cache:
            faq_key = self.pattern_cache[question_hash]
            if faq_key in self.faq_cache:
                entry = self.faq_cache[faq_key]
                
                # Check expiration
                if entry.expires_at and datetime.now() > entry.expires_at:
                    logger.debug(f"FAQ entry expired: {faq_key}")
                    self.total_misses += 1
                    return None
                
                entry.hit_count += 1
                self.total_hits += 1
                
                return {
                    "answer": entry.answer,
                    "sources": entry.sources,
                    "category": entry.category,
                    "match_type": "exact",
                    "confidence": 1.0,
                    "cached": True
                }
        
        # Try fuzzy matching
        best_match = None
        best_similarity = 0.0
        
        for faq_key, entry in self.faq_cache.items():
            similarity = self._calculate_similarity(question, entry.question)
            if similarity > best_similarity and similarity >= self.fuzzy_threshold:
                best_similarity = similarity
                best_match = faq_key
        
        if best_match:
            entry = self.faq_cache[best_match]
            entry.hit_count += 1
            self.total_hits += 1
            
            return {
                "answer": entry.answer,
                "sources": entry.sources,
                "category": entry.category,
                "match_type": "fuzzy",
                "confidence": best_similarity,
                "cached": True
            }
        
        # Try pattern matching
        matched_category = self._match_pattern(question)
        if matched_category:
            # Find best FAQ in category
            for faq_key, entry in self.faq_cache.items():
                if entry.category == matched_category:
                    entry.hit_count += 1
                    self.total_hits += 1
                    
                    return {
                        "answer": entry.answer,
                        "sources": entry.sources,
                        "category": entry.category,
                        "match_type": "pattern",
                        "confidence": 0.8,
                        "cached": True
                    }
        
        self.total_misses += 1
        return None
    
    def set(
        self,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        category: str = "general",
        ttl_hours: Optional[int] = None
    ) -> bool:
        """
        Cache a FAQ response.
        
        Args:
            question: Question text
            answer: Answer text
            sources: Source documents
            category: FAQ category
            ttl_hours: Custom TTL
            
        Returns:
            True if cached successfully
        """
        # Check cache size
        if len(self.faq_cache) >= self.max_cache_size:
            self._evict_least_used()
        
        ttl = ttl_hours or self.default_ttl_hours
        question_hash = self._hash_question(question)
        faq_key = f"faq_{question_hash}"
        
        entry = FAQEntry(
            question=question,
            question_normalized=self._normalize_question(question),
            answer=answer,
            sources=sources or [],
            category=category,
            expires_at=datetime.now() + timedelta(hours=ttl)
        )
        
        self.faq_cache[faq_key] = entry
        self.pattern_cache[question_hash] = faq_key
        
        logger.debug(f"Cached FAQ: {question[:50]}...")
        return True
    
    def _evict_least_used(self) -> None:
        """Evict least used entries."""
        if not self.faq_cache:
            return
        
        # Sort by hit count (ascending)
        sorted_entries = sorted(
            [(k, v) for k, v in self.faq_cache.items() if not k.startswith("predefined")],
            key=lambda x: x[1].hit_count
        )
        
        # Remove bottom 10%
        to_remove = max(1, len(sorted_entries) // 10)
        for i in range(to_remove):
            faq_key = sorted_entries[i][0]
            entry = self.faq_cache.pop(faq_key, None)
            if entry:
                # Remove from pattern cache too
                pattern_hash = self._hash_question(entry.question)
                self.pattern_cache.pop(pattern_hash, None)
        
        logger.debug(f"Evicted {to_remove} FAQ entries")
    
    def add_pattern(self, faq_key: str, pattern: str) -> bool:
        """Add a pattern for an existing FAQ."""
        if faq_key not in self.faq_cache:
            return False
        
        pattern_hash = self._hash_question(pattern)
        self.pattern_cache[pattern_hash] = faq_key
        return True
    
    def invalidate(self, question: str) -> bool:
        """Invalidate a cached FAQ."""
        question_hash = self._hash_question(question)
        
        if question_hash in self.pattern_cache:
            faq_key = self.pattern_cache.pop(question_hash)
            if faq_key in self.faq_cache:
                del self.faq_cache[faq_key]
                return True
        
        return False
    
    def invalidate_category(self, category: str) -> int:
        """Invalidate all FAQs in a category."""
        to_remove = [
            k for k, v in self.faq_cache.items() 
            if v.category == category and not k.startswith("predefined")
        ]
        
        for faq_key in to_remove:
            entry = self.faq_cache.pop(faq_key)
            pattern_hash = self._hash_question(entry.question)
            self.pattern_cache.pop(pattern_hash, None)
        
        return len(to_remove)
    
    def clear(self) -> None:
        """Clear all cached FAQs (keeps predefined)."""
        # Keep only predefined
        self.faq_cache = {k: v for k, v in self.faq_cache.items() if k in self.PREDEFINED_FAQS}
        self.pattern_cache.clear()
        self._load_predefined_faqs()
        
        self.total_hits = 0
        self.total_misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.total_hits + self.total_misses
        hit_rate = self.total_hits / total if total > 0 else 0.0
        
        category_counts = {}
        for entry in self.faq_cache.values():
            category = entry.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_entries": len(self.faq_cache),
            "predefined_entries": len(self.PREDEFINED_FAQS),
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": hit_rate,
            "categories": category_counts,
            "max_size": self.max_cache_size
        }
    
    def export_faqs(self) -> List[Dict[str, Any]]:
        """Export all FAQs as list."""
        return [
            {
                "key": k,
                **v.to_dict()
            }
            for k, v in self.faq_cache.items()
        ]
    
    def import_faqs(self, faqs: List[Dict[str, Any]]) -> int:
        """Import FAQs from list."""
        imported = 0
        for faq in faqs:
            try:
                self.set(
                    question=faq["question"],
                    answer=faq["answer"],
                    sources=faq.get("sources", []),
                    category=faq.get("category", "general")
                )
                imported += 1
            except Exception as e:
                logger.warning(f"Failed to import FAQ: {e}")
        
        return imported


# Global instance
_faq_cache = None

def get_faq_cache() -> FAQPromptCache:
    """Get global FAQ cache instance."""
    global _faq_cache
    if _faq_cache is None:
        _faq_cache = FAQPromptCache()
    return _faq_cache

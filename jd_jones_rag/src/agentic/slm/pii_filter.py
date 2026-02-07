"""
PII Filter
Filters personally identifiable information from text.
Runs locally to ensure sensitive data never leaves the system.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import re
from enum import Enum


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    IP_ADDRESS = "ip_address"
    BANK_ACCOUNT = "bank_account"
    PASSPORT = "passport"


@dataclass
class PIIMatch:
    """A detected PII match."""
    pii_type: PIIType
    original_value: str
    start_position: int
    end_position: int
    confidence: float
    masked_value: str


@dataclass
class FilterResult:
    """Result of PII filtering."""
    original_text: str
    filtered_text: str
    pii_found: List[PIIMatch]
    pii_count: int
    risk_level: str  # low, medium, high


class PIIFilter:
    """
    Filters PII from text for privacy protection.
    
    Use cases:
    - Anonymize customer enquiries before sending to cloud LLM
    - Redact sensitive data from logs
    - Compliance with data protection regulations
    
    In production, would use a specialized model.
    Currently implements regex-based detection.
    """
    
    # Regex patterns for PII detection
    PATTERNS = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PIIType.PHONE: r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        PIIType.SSN: r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        PIIType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        PIIType.DATE_OF_BIRTH: r'\b(?:0?[1-9]|[12][0-9]|3[01])[\/\-](?:0?[1-9]|1[012])[\/\-](?:19|20)\d{2}\b',
    }
    
    # Common name patterns (simplified)
    NAME_PATTERNS = [
        r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
    ]
    
    # Masking characters
    MASK_CHARS = {
        PIIType.EMAIL: "[EMAIL]",
        PIIType.PHONE: "[PHONE]",
        PIIType.SSN: "[SSN]",
        PIIType.CREDIT_CARD: "[CARD]",
        PIIType.ADDRESS: "[ADDRESS]",
        PIIType.NAME: "[NAME]",
        PIIType.DATE_OF_BIRTH: "[DOB]",
        PIIType.IP_ADDRESS: "[IP]",
        PIIType.BANK_ACCOUNT: "[BANK]",
        PIIType.PASSPORT: "[PASSPORT]"
    }
    
    def __init__(
        self,
        detect_types: Optional[Set[PIIType]] = None,
        mask_partial: bool = False
    ):
        """
        Initialize PII filter.
        
        Args:
            detect_types: Types of PII to detect (None = all)
            mask_partial: Whether to partially mask (keep some chars)
        """
        self.detect_types = detect_types or set(PIIType)
        self.mask_partial = mask_partial
    
    def filter(self, text: str) -> FilterResult:
        """
        Filter PII from text.
        
        Args:
            text: Text to filter
            
        Returns:
            FilterResult with filtered text and detected PII
        """
        pii_matches = []
        filtered_text = text
        
        # Detect each PII type
        for pii_type in self.detect_types:
            if pii_type in self.PATTERNS:
                pattern = self.PATTERNS[pii_type]
                matches = list(re.finditer(pattern, text))
                
                for match in matches:
                    masked = self._mask_value(match.group(), pii_type)
                    pii_matches.append(PIIMatch(
                        pii_type=pii_type,
                        original_value=match.group(),
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=self._calculate_confidence(pii_type, match.group()),
                        masked_value=masked
                    ))
        
        # Detect names
        if PIIType.NAME in self.detect_types:
            for pattern in self.NAME_PATTERNS:
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    masked = self._mask_value(match.group(), PIIType.NAME)
                    pii_matches.append(PIIMatch(
                        pii_type=PIIType.NAME,
                        original_value=match.group(),
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=0.7,  # Names have lower confidence
                        masked_value=masked
                    ))
        
        # Sort by position (reverse to replace from end)
        pii_matches.sort(key=lambda x: x.start_position, reverse=True)
        
        # Apply masks
        for match in pii_matches:
            filtered_text = (
                filtered_text[:match.start_position] +
                match.masked_value +
                filtered_text[match.end_position:]
            )
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(pii_matches)
        
        # Re-sort for output
        pii_matches.sort(key=lambda x: x.start_position)
        
        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            pii_found=pii_matches,
            pii_count=len(pii_matches),
            risk_level=risk_level
        )
    
    def _mask_value(self, value: str, pii_type: PIIType) -> str:
        """Mask a PII value."""
        if self.mask_partial:
            # Keep first and last characters
            if len(value) > 4:
                return value[0] + "*" * (len(value) - 2) + value[-1]
            return "*" * len(value)
        return self.MASK_CHARS.get(pii_type, "[REDACTED]")
    
    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """Calculate confidence of PII detection."""
        confidence = 0.9  # Default high confidence for regex matches
        
        # Adjust based on type-specific validation
        if pii_type == PIIType.EMAIL:
            # Valid email structure
            if "@" in value and "." in value.split("@")[1]:
                confidence = 0.95
        
        elif pii_type == PIIType.PHONE:
            # Check digit count
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 10:
                confidence = 0.9
            else:
                confidence = 0.6
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Luhn algorithm check (simplified)
            digits = re.sub(r'\D', '', value)
            if len(digits) == 16 and self._luhn_check(digits):
                confidence = 0.95
            else:
                confidence = 0.7
        
        elif pii_type == PIIType.SSN:
            digits = re.sub(r'\D', '', value)
            if len(digits) == 9:
                confidence = 0.85
            else:
                confidence = 0.5
        
        return confidence
    
    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        try:
            digits = [int(d) for d in card_number]
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(divmod(d * 2, 10))
            
            return checksum % 10 == 0
        except Exception:
            return False
    
    def _calculate_risk_level(self, matches: List[PIIMatch]) -> str:
        """Calculate overall risk level."""
        if not matches:
            return "low"
        
        high_risk_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT}
        medium_risk_types = {PIIType.EMAIL, PIIType.PHONE, PIIType.DATE_OF_BIRTH}
        
        has_high_risk = any(m.pii_type in high_risk_types for m in matches)
        has_medium_risk = any(m.pii_type in medium_risk_types for m in matches)
        
        if has_high_risk or len(matches) >= 5:
            return "high"
        elif has_medium_risk or len(matches) >= 2:
            return "medium"
        return "low"
    
    def scan(self, text: str) -> Dict[str, Any]:
        """
        Scan text for PII without filtering.
        
        Args:
            text: Text to scan
            
        Returns:
            Scan results
        """
        result = self.filter(text)
        
        return {
            "pii_detected": result.pii_count > 0,
            "pii_count": result.pii_count,
            "pii_types": list(set(m.pii_type.value for m in result.pii_found)),
            "risk_level": result.risk_level,
            "details": [
                {
                    "type": m.pii_type.value,
                    "confidence": m.confidence,
                    "position": {"start": m.start_position, "end": m.end_position}
                }
                for m in result.pii_found
            ]
        }
    
    def anonymize_for_llm(
        self,
        text: str,
        preserve_context: bool = True
    ) -> Tuple[str, Dict[str, str]]:
        """
        Anonymize text for sending to external LLM.
        
        Args:
            text: Text to anonymize
            preserve_context: Whether to use meaningful placeholders
            
        Returns:
            Tuple of (anonymized_text, mapping_dict)
        """
        result = self.filter(text)
        
        mapping = {}
        anonymized = result.filtered_text
        
        if preserve_context:
            # Create meaningful placeholders
            counters = {}
            for match in result.pii_found:
                pii_type = match.pii_type.value
                counters[pii_type] = counters.get(pii_type, 0) + 1
                placeholder = f"[{pii_type.upper()}_{counters[pii_type]}]"
                mapping[placeholder] = match.original_value
                
                # Replace in text
                anonymized = anonymized.replace(
                    match.masked_value,
                    placeholder,
                    1
                )
        
        return anonymized, mapping
    
    def deanonymize(self, text: str, mapping: Dict[str, str]) -> str:
        """
        Restore original values from anonymized text.
        
        Args:
            text: Anonymized text
            mapping: Placeholder to original value mapping
            
        Returns:
            Text with original values restored
        """
        result = text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result

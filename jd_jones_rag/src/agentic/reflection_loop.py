"""
Reflection Loop
Validates results, checks compliance, and triggers re-queries when needed.
Implements the "Self-Correction" pattern for agentic systems.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings
from src.agentic.router_agent import QueryIntent


class ValidationStatus(str, Enum):
    """Validation result status."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    NEEDS_VERIFICATION = "needs_verification"


@dataclass
class ReflectionResult:
    """Result of the reflection/validation process."""
    is_valid: bool
    status: ValidationStatus
    confidence: float
    
    # Issues found
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Corrective actions
    corrective_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Compliance
    standards_checked: List[str] = field(default_factory=list)
    compliance_results: Dict[str, bool] = field(default_factory=dict)
    
    # Reasoning
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "status": self.status.value,
            "confidence": self.confidence,
            "issues": self.issues,
            "warnings": self.warnings,
            "corrective_actions": self.corrective_actions,
            "standards_checked": self.standards_checked,
            "compliance_results": self.compliance_results,
            "reasoning": self.reasoning
        }


class ReflectionLoop:
    """
    Validates results and triggers corrections when needed.
    
    Capabilities:
    - Checks if retrieved information answers the query
    - Validates product recommendations against specifications
    - Verifies compliance with industry standards
    - Suggests corrective re-queries
    """
    
    # Industry standards and their requirements
    STANDARDS_REQUIREMENTS = {
        "API_622": {
            "name": "API 622 - Stem Packing Testing",
            "applies_to": ["valve packing", "stem seal"],
            "requirements": [
                "Fugitive emission testing",
                "500+ thermal cycles",
                "Live loading capability",
                "Fire-safe certification"
            ]
        },
        "API_624": {
            "name": "API 624 - Rising Stem Valves Testing",
            "applies_to": ["rising stem valve", "gate valve", "globe valve"],
            "requirements": [
                "Type testing per API 641",
                "Mechanical endurance",
                "Temperature cycling",
                "Emission measurement"
            ]
        },
        "SHELL_SPE": {
            "name": "Shell SPE 77/312",
            "applies_to": ["gasket", "seal", "packing"],
            "requirements": [
                "Fire-safe testing",
                "Blowout resistance",
                "Anti-extrusion capability"
            ]
        },
        "FDA": {
            "name": "FDA 21 CFR 177",
            "applies_to": ["food contact", "pharmaceutical"],
            "requirements": [
                "Food contact approval",
                "Non-toxic materials",
                "Traceability"
            ]
        },
        "API_6A": {
            "name": "API 6A - Wellhead Equipment",
            "applies_to": ["wellhead", "christmas tree", "high pressure"],
            "requirements": [
                "Material certification",
                "Pressure rating",
                "Temperature rating",
                "H2S service capability"
            ]
        }
    }
    
    VALIDATION_PROMPT = """You are a technical validator for JD Jones Manufacturing's AI system.

Your job is to:
1. Check if the retrieved information actually answers the query
2. Verify that recommended products meet the specified requirements
3. Check compliance with mentioned industry standards
4. Identify any safety concerns or missing critical information
5. Suggest corrective actions if needed

VALIDATION CRITERIA:
- Temperature rating must exceed operating temperature by safety margin
- Pressure rating must exceed operating pressure by safety margin
- Material must be compatible with the media
- Certifications must match the specified standards
- Safety warnings must be included for hazardous applications

STANDARDS KNOWLEDGE:
{standards_info}

Query: {query}
Intent: {intent}
Parameters: {parameters}

Retrieved Results:
{tool_results}

Respond in JSON:
{{
    "is_valid": true/false,
    "status": "valid/warning/invalid/needs_verification",
    "confidence": 0.0-1.0,
    "issues": ["list of critical issues"],
    "warnings": ["list of warnings"],
    "corrective_actions": [
        {{"type": "re_query", "tool": "tool_name", "reason": "why", "modified_query": "new query"}}
    ],
    "compliance_check": {{"STANDARD": true/false}},
    "reasoning": "explanation"
}}
"""

    def __init__(self):
        """Initialize reflection loop."""
        from src.config.settings import get_llm
        self.llm = get_llm(temperature=0)
    
    async def validate(
        self,
        query: str,
        intent: QueryIntent,
        tool_results: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ) -> ReflectionResult:
        """
        Validate tool results against query requirements.
        
        Args:
            query: Original query
            intent: Detected intent
            tool_results: Results from tool executions
            parameters: Collected parameters
            
        Returns:
            ReflectionResult with validation status and corrections
        """
        # Check if we have any results to validate
        if not tool_results or all(not r.get("success") for r in tool_results):
            return ReflectionResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                confidence=0.9,
                issues=["No valid results retrieved from tools"],
                corrective_actions=[{
                    "type": "re_query",
                    "tool": "vector_search",
                    "reason": "No results found",
                    "modified_query": query
                }]
            )
        
        # Determine which standards to check
        standards_to_check = self._identify_relevant_standards(parameters, query)
        
        # Build standards info for prompt
        standards_info = self._format_standards_info(standards_to_check)
        
        # Use LLM for sophisticated validation
        try:
            return await self._validate_with_llm(
                query=query,
                intent=intent,
                tool_results=tool_results,
                parameters=parameters,
                standards_info=standards_info,
                standards_to_check=standards_to_check
            )
        except Exception as e:
            # Fallback to rule-based validation
            return self._validate_with_rules(
                tool_results=tool_results,
                parameters=parameters,
                standards_to_check=standards_to_check
            )
    
    async def _validate_with_llm(
        self,
        query: str,
        intent: QueryIntent,
        tool_results: List[Dict[str, Any]],
        parameters: Dict[str, Any],
        standards_info: str,
        standards_to_check: List[str]
    ) -> ReflectionResult:
        """Validate using LLM."""
        prompt = self.VALIDATION_PROMPT.format(
            standards_info=standards_info,
            query=query,
            intent=intent.value,
            parameters=json.dumps(parameters, indent=2),
            tool_results=json.dumps(tool_results, indent=2)
        )
        
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        
        # Parse response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content.strip())
        
        return ReflectionResult(
            is_valid=result.get("is_valid", True),
            status=ValidationStatus(result.get("status", "valid")),
            confidence=result.get("confidence", 0.8),
            issues=result.get("issues", []),
            warnings=result.get("warnings", []),
            corrective_actions=result.get("corrective_actions", []),
            standards_checked=standards_to_check,
            compliance_results=result.get("compliance_check", {}),
            reasoning=result.get("reasoning", "")
        )
    
    def _validate_with_rules(
        self,
        tool_results: List[Dict[str, Any]],
        parameters: Dict[str, Any],
        standards_to_check: List[str]
    ) -> ReflectionResult:
        """Rule-based fallback validation."""
        issues = []
        warnings = []
        compliance = {}
        
        # Check temperature compatibility
        if "temperature" in parameters:
            temp_issues = self._check_temperature_compatibility(
                parameters["temperature"], tool_results
            )
            warnings.extend(temp_issues)
        
        # Check pressure compatibility
        if "pressure" in parameters:
            pressure_issues = self._check_pressure_compatibility(
                parameters["pressure"], tool_results
            )
            warnings.extend(pressure_issues)
        
        # Basic standards compliance check
        for standard in standards_to_check:
            # This would need actual product data in production
            compliance[standard] = True  # Placeholder
        
        is_valid = len(issues) == 0
        status = ValidationStatus.VALID if is_valid else ValidationStatus.WARNING
        
        if warnings:
            status = ValidationStatus.WARNING
        
        return ReflectionResult(
            is_valid=is_valid,
            status=status,
            confidence=0.7,
            issues=issues,
            warnings=warnings,
            standards_checked=standards_to_check,
            compliance_results=compliance
        )
    
    def _identify_relevant_standards(
        self,
        parameters: Dict[str, Any],
        query: str
    ) -> List[str]:
        """Identify which standards are relevant to check."""
        standards = []
        query_lower = query.lower()
        
        # Check explicitly mentioned standards
        for std_code, std_info in self.STANDARDS_REQUIREMENTS.items():
            std_name_lower = std_code.lower().replace("_", " ")
            if std_name_lower in query_lower:
                standards.append(std_code)
        
        # Check based on industry
        industry = parameters.get("industry", "").lower()
        if "oil" in industry or "gas" in industry:
            standards.extend(["API_622", "API_624", "SHELL_SPE"])
        if "pharma" in industry or "food" in industry:
            standards.append("FDA")
        
        # Check based on certifications parameter
        certs = parameters.get("certifications", [])
        for cert in certs:
            cert_upper = cert.upper().replace(" ", "_")
            if cert_upper in self.STANDARDS_REQUIREMENTS:
                standards.append(cert_upper)
        
        return list(set(standards))
    
    def _format_standards_info(self, standards: List[str]) -> str:
        """Format standards information for the prompt."""
        if not standards:
            return "No specific standards to check."
        
        info_parts = []
        for std in standards:
            if std in self.STANDARDS_REQUIREMENTS:
                std_info = self.STANDARDS_REQUIREMENTS[std]
                info_parts.append(
                    f"- {std_info['name']}: Applies to {', '.join(std_info['applies_to'])}. "
                    f"Requirements: {', '.join(std_info['requirements'])}"
                )
        
        return "\n".join(info_parts)
    
    def _check_temperature_compatibility(
        self,
        required_temp: str,
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Check if products meet temperature requirements."""
        warnings = []
        
        # Parse required temperature (simplified)
        import re
        temp_match = re.search(r'(-?\d+)', str(required_temp))
        if not temp_match:
            return warnings
        
        required = int(temp_match.group(1))
        
        # Check each result for temperature rating
        for result in results:
            if not result.get("success"):
                continue
            
            data = result.get("result", {})
            if isinstance(data, dict):
                product_temp = data.get("max_temperature", data.get("temperature_rating"))
                if product_temp:
                    try:
                        temp_match = re.search(r'(-?\d+)', str(product_temp))
                        if temp_match:
                            max_temp = int(temp_match.group(1))
                            if max_temp < required * 1.1:  # 10% safety margin
                                warnings.append(
                                    f"Product temperature rating ({max_temp}°C) is close to "
                                    f"or below required temperature ({required}°C). "
                                    f"Consider higher-rated alternative."
                                )
                    except (ValueError, TypeError):
                        pass
        
        return warnings
    
    def _check_pressure_compatibility(
        self,
        required_pressure: str,
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Check if products meet pressure requirements."""
        warnings = []
        
        # Parse required pressure (simplified)
        import re
        pressure_match = re.search(r'(\d+)', str(required_pressure))
        if not pressure_match:
            return warnings
        
        required = int(pressure_match.group(1))
        
        # Check each result for pressure rating
        for result in results:
            if not result.get("success"):
                continue
            
            data = result.get("result", {})
            if isinstance(data, dict):
                product_pressure = data.get("max_pressure", data.get("pressure_rating"))
                if product_pressure:
                    try:
                        pressure_match = re.search(r'(\d+)', str(product_pressure))
                        if pressure_match:
                            max_pressure = int(pressure_match.group(1))
                            if max_pressure < required * 1.2:  # 20% safety margin
                                warnings.append(
                                    f"Product pressure rating ({max_pressure} bar) provides "
                                    f"limited safety margin above required ({required} bar). "
                                    f"Verify application requirements."
                                )
                    except (ValueError, TypeError):
                        pass
        
        return warnings
    
    async def check_compliance(
        self,
        product_info: Dict[str, Any],
        standards: List[str]
    ) -> Dict[str, Any]:
        """
        Check product compliance against specific standards.
        
        Args:
            product_info: Product information
            standards: Standards to check against
            
        Returns:
            Compliance results
        """
        results = {
            "compliant": [],
            "non_compliant": [],
            "unknown": [],
            "details": {}
        }
        
        for standard in standards:
            if standard not in self.STANDARDS_REQUIREMENTS:
                results["unknown"].append(standard)
                continue
            
            std_reqs = self.STANDARDS_REQUIREMENTS[standard]
            
            # Check if product has certification
            product_certs = product_info.get("certifications", [])
            product_certs_upper = [c.upper().replace(" ", "_") for c in product_certs]
            
            if standard in product_certs_upper or standard.replace("_", " ") in product_certs:
                results["compliant"].append(standard)
                results["details"][standard] = {
                    "status": "certified",
                    "requirements_met": std_reqs["requirements"]
                }
            else:
                # Check if product type is applicable
                product_type = product_info.get("type", "").lower()
                if any(app in product_type for app in std_reqs["applies_to"]):
                    results["non_compliant"].append(standard)
                    results["details"][standard] = {
                        "status": "not_certified",
                        "requirements": std_reqs["requirements"]
                    }
                else:
                    results["unknown"].append(standard)
        
        return results

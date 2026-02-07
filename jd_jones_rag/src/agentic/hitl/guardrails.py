"""
Guardrails
Safety constraints and validation for agent actions.
Ensures agents operate within defined boundaries.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re


class GuardrailType(str, Enum):
    """Types of guardrails."""
    INPUT_FILTER = "input_filter"          # Filter user input
    OUTPUT_FILTER = "output_filter"        # Filter agent output
    ACTION_CONSTRAINT = "action_constraint" # Limit what actions can be taken
    DATA_ACCESS = "data_access"            # Control data access
    RATE_LIMIT = "rate_limit"              # Rate limiting
    CONTENT_POLICY = "content_policy"      # Content restrictions


class GuardrailSeverity(str, Enum):
    """Severity of guardrail violations."""
    BLOCK = "block"       # Block the action entirely
    WARN = "warn"         # Allow but warn
    LOG = "log"           # Allow and log only
    MODIFY = "modify"     # Modify the action/content


@dataclass
class GuardrailResult:
    """Result of guardrail evaluation."""
    passed: bool
    guardrail_name: str
    guardrail_type: GuardrailType
    severity: GuardrailSeverity
    message: str = ""
    violations: List[str] = field(default_factory=list)
    modified_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "guardrail_name": self.guardrail_name,
            "guardrail_type": self.guardrail_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "violations": self.violations,
            "modified_content": self.modified_content,
            "metadata": self.metadata,
            "warnings": self.warnings
        }


@dataclass
class AggregatedGuardrailResult:
    """Aggregated result from multiple guardrails."""
    passed: bool
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    individual_results: List[GuardrailResult] = field(default_factory=list)
    
    @classmethod
    def from_results(cls, results: List[GuardrailResult]) -> "AggregatedGuardrailResult":
        """Create aggregated result from individual results."""
        passed = all(r.passed for r in results)
        warnings = []
        violations = []
        messages = []
        
        for r in results:
            if not r.passed:
                if r.severity == GuardrailSeverity.BLOCK:
                    messages.append(r.message)
                    violations.extend(r.violations)
                else:
                    warnings.append(r.message)
            elif r.severity == GuardrailSeverity.WARN:
                warnings.append(r.message)
        
        return cls(
            passed=passed,
            message="; ".join(messages) if messages else "",
            warnings=warnings,
            violations=violations,
            individual_results=results
        )


@dataclass
class Guardrail:
    """A single guardrail definition."""
    name: str
    guardrail_type: GuardrailType
    severity: GuardrailSeverity
    check_function: Callable
    enabled: bool = True
    description: str = ""
    
    async def check(self, content: Any, context: Dict[str, Any] = None) -> GuardrailResult:
        """Run the guardrail check."""
        try:
            result = self.check_function(content, context or {})
            if isinstance(result, GuardrailResult):
                return result
            elif isinstance(result, bool):
                return GuardrailResult(
                    passed=result,
                    guardrail_name=self.name,
                    guardrail_type=self.guardrail_type,
                    severity=self.severity
                )
            else:
                return GuardrailResult(
                    passed=True,
                    guardrail_name=self.name,
                    guardrail_type=self.guardrail_type,
                    severity=self.severity
                )
        except Exception as e:
            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                guardrail_type=self.guardrail_type,
                severity=GuardrailSeverity.BLOCK,
                message=f"Guardrail error: {str(e)}"
            )


class Guardrails:
    """
    Manages safety guardrails for the agentic system.
    
    Features:
    - PII detection and filtering
    - Prompt injection prevention
    - Content policy enforcement
    - Action constraints
    - Rate limiting
    """
    
    # Patterns for PII detection
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r'ignore\s+(?:all\s+)?(?:previous\s+)?instructions',
        r'disregard\s+(?:all\s+)?(?:previous\s+)?(?:instructions|prompts)',
        r'forget\s+(?:everything|all)',
        r'new\s+instructions?\s*:',
        r'system\s*:\s*you\s+are',
        r'</?(system|user|assistant)>',
        r'jailbreak',
        r'bypass\s+(?:safety|security|restrictions)',
    ]
    
    # Restricted topics/content
    RESTRICTED_CONTENT = [
        "competitor pricing",
        "salary information",
        "personal employee data",
        "trade secrets",
        "unreleased products",
    ]
    
    def __init__(self):
        """Initialize guardrails."""
        self.guardrails: Dict[str, Guardrail] = {}
        self.violation_log: List[Dict[str, Any]] = []
        
        # Register default guardrails
        self._register_default_guardrails()
    
    def _register_default_guardrails(self):
        """Register built-in guardrails."""
        # PII Detection
        self.register(Guardrail(
            name="pii_detector",
            guardrail_type=GuardrailType.OUTPUT_FILTER,
            severity=GuardrailSeverity.MODIFY,
            check_function=self._check_pii,
            description="Detects and masks PII in outputs"
        ))
        
        # Prompt Injection Prevention
        self.register(Guardrail(
            name="injection_prevention",
            guardrail_type=GuardrailType.INPUT_FILTER,
            severity=GuardrailSeverity.BLOCK,
            check_function=self._check_injection,
            description="Prevents prompt injection attacks"
        ))
        
        # Content Policy
        self.register(Guardrail(
            name="content_policy",
            guardrail_type=GuardrailType.CONTENT_POLICY,
            severity=GuardrailSeverity.BLOCK,
            check_function=self._check_content_policy,
            description="Enforces content restrictions"
        ))
        
        # Output Length
        self.register(Guardrail(
            name="output_length",
            guardrail_type=GuardrailType.OUTPUT_FILTER,
            severity=GuardrailSeverity.MODIFY,
            check_function=self._check_output_length,
            description="Limits output length"
        ))
        
        # SQL Injection Prevention
        self.register(Guardrail(
            name="sql_injection",
            guardrail_type=GuardrailType.INPUT_FILTER,
            severity=GuardrailSeverity.BLOCK,
            check_function=self._check_sql_injection,
            description="Prevents SQL injection in inputs"
        ))
    
    def register(self, guardrail: Guardrail):
        """Register a guardrail."""
        self.guardrails[guardrail.name] = guardrail
    
    def unregister(self, name: str) -> bool:
        """Unregister a guardrail."""
        if name in self.guardrails:
            del self.guardrails[name]
            return True
        return False
    
    def enable(self, name: str) -> bool:
        """Enable a guardrail."""
        if name in self.guardrails:
            self.guardrails[name].enabled = True
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """Disable a guardrail."""
        if name in self.guardrails:
            self.guardrails[name].enabled = False
            return True
        return False
    
    async def check_input_async(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[GuardrailResult]:
        """
        Check input against all input guardrails (async version).
        
        Args:
            content: Input content to check
            context: Additional context
            
        Returns:
            List of guardrail results
        """
        results = []
        
        for guardrail in self.guardrails.values():
            if not guardrail.enabled:
                continue
            if guardrail.guardrail_type != GuardrailType.INPUT_FILTER:
                continue
            
            result = await guardrail.check(content, context)
            results.append(result)
            
            if not result.passed:
                self._log_violation(guardrail, content, result)
        
        return results
    
    def check_input(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AggregatedGuardrailResult:
        """
        Check input against all input guardrails (synchronous wrapper).
        Returns aggregated result for easy use in orchestrator.
        
        For async contexts (FastAPI endpoints), use check_input_async() instead.
        
        Args:
            content: Input content to check
            context: Additional context
            
        Returns:
            AggregatedGuardrailResult with combined pass/fail status
        """
        import asyncio
        
        # Run the async check in a way that's safe for both sync and async contexts
        try:
            # Try to get the running loop - this will fail in sync context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - use thread pool to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Run in a new event loop in a separate thread
                    future = executor.submit(
                        lambda: asyncio.run(self._async_check_input(content, context))
                    )
                    results = future.result(timeout=30)
            except RuntimeError:
                # No running loop - we're in a sync context, safe to create one
                results = asyncio.run(self._async_check_input(content, context))
        except Exception as e:
            # Fallback: return empty result with error logged
            import logging
            logging.error(f"Guardrails check_input failed: {e}")
            return AggregatedGuardrailResult(
                passed=True,  # Fail-open to not block on guardrail errors
                message=f"Guardrail check error: {str(e)}",
                warnings=[f"Guardrail system error: {str(e)}"]
            )
        
        return AggregatedGuardrailResult.from_results(results)
    
    async def _async_check_input(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[GuardrailResult]:
        """Async implementation of check_input."""
        results = []
        
        for guardrail in self.guardrails.values():
            if not guardrail.enabled:
                continue
            if guardrail.guardrail_type != GuardrailType.INPUT_FILTER:
                continue
            
            result = await guardrail.check(content, context)
            results.append(result)
            
            if not result.passed:
                self._log_violation(guardrail, content, result)
        
        return results
    
    async def check_output(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, List[GuardrailResult]]:
        """
        Check and potentially modify output.
        
        Args:
            content: Output content to check
            context: Additional context
            
        Returns:
            Tuple of (potentially modified content, results)
        """
        results = []
        modified_content = content
        
        for guardrail in self.guardrails.values():
            if not guardrail.enabled:
                continue
            if guardrail.guardrail_type != GuardrailType.OUTPUT_FILTER:
                continue
            
            result = await guardrail.check(modified_content, context)
            results.append(result)
            
            if not result.passed:
                self._log_violation(guardrail, content, result)
                
                if result.severity == GuardrailSeverity.MODIFY and result.modified_content:
                    modified_content = result.modified_content
        
        return modified_content, results
    
    async def check_action(
        self,
        action_type: str,
        action_payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[GuardrailResult]:
        """
        Check if an action is allowed.
        
        Args:
            action_type: Type of action
            action_payload: Action parameters
            context: Additional context
            
        Returns:
            List of guardrail results
        """
        results = []
        
        for guardrail in self.guardrails.values():
            if not guardrail.enabled:
                continue
            if guardrail.guardrail_type != GuardrailType.ACTION_CONSTRAINT:
                continue
            
            check_context = {
                "action_type": action_type,
                "payload": action_payload,
                **(context or {})
            }
            
            result = await guardrail.check(action_payload, check_context)
            results.append(result)
            
            if not result.passed:
                self._log_violation(guardrail, str(action_payload), result)
        
        return results
    
    def _check_pii(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Check for PII and mask if found."""
        violations = []
        modified = content
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations.append(f"Found {len(matches)} {pii_type} pattern(s)")
                # Mask the PII
                modified = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", modified, flags=re.IGNORECASE)
        
        return GuardrailResult(
            passed=len(violations) == 0,
            guardrail_name="pii_detector",
            guardrail_type=GuardrailType.OUTPUT_FILTER,
            severity=GuardrailSeverity.MODIFY,
            message="PII detected and masked" if violations else "",
            violations=violations,
            modified_content=modified if violations else None
        )
    
    def _check_injection(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Check for prompt injection attempts."""
        violations = []
        content_lower = content.lower()
        
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, content_lower):
                violations.append(f"Injection pattern detected: {pattern[:30]}...")
        
        return GuardrailResult(
            passed=len(violations) == 0,
            guardrail_name="injection_prevention",
            guardrail_type=GuardrailType.INPUT_FILTER,
            severity=GuardrailSeverity.BLOCK,
            message="Potential prompt injection detected" if violations else "",
            violations=violations
        )
    
    def _check_content_policy(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Check content against policy restrictions."""
        violations = []
        content_lower = content.lower()
        
        for restricted in self.RESTRICTED_CONTENT:
            if restricted.lower() in content_lower:
                violations.append(f"Restricted content: {restricted}")
        
        return GuardrailResult(
            passed=len(violations) == 0,
            guardrail_name="content_policy",
            guardrail_type=GuardrailType.CONTENT_POLICY,
            severity=GuardrailSeverity.BLOCK,
            message="Content policy violation" if violations else "",
            violations=violations
        )
    
    def _check_output_length(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Check and limit output length."""
        max_length = context.get("max_output_length", 10000)
        
        if len(content) > max_length:
            truncated = content[:max_length] + "\n\n[Response truncated due to length]"
            return GuardrailResult(
                passed=False,
                guardrail_name="output_length",
                guardrail_type=GuardrailType.OUTPUT_FILTER,
                severity=GuardrailSeverity.MODIFY,
                message=f"Output exceeded {max_length} characters",
                violations=[f"Length: {len(content)} > {max_length}"],
                modified_content=truncated
            )
        
        return GuardrailResult(
            passed=True,
            guardrail_name="output_length",
            guardrail_type=GuardrailType.OUTPUT_FILTER,
            severity=GuardrailSeverity.MODIFY
        )
    
    def _check_sql_injection(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Check for SQL injection patterns."""
        sql_patterns = [
            r";\s*DROP\s+TABLE",
            r";\s*DELETE\s+FROM",
            r";\s*UPDATE\s+\w+\s+SET",
            r"'\s*OR\s+'?1'?\s*=\s*'?1",
            r"UNION\s+SELECT",
            r"--\s*$",
            r";\s*EXEC\s*\(",
        ]
        
        violations = []
        for pattern in sql_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(f"SQL pattern detected: {pattern[:20]}...")
        
        return GuardrailResult(
            passed=len(violations) == 0,
            guardrail_name="sql_injection",
            guardrail_type=GuardrailType.INPUT_FILTER,
            severity=GuardrailSeverity.BLOCK,
            message="Potential SQL injection detected" if violations else "",
            violations=violations
        )
    
    def _log_violation(
        self,
        guardrail: Guardrail,
        content: str,
        result: GuardrailResult
    ):
        """Log a guardrail violation."""
        self.violation_log.append({
            "guardrail_name": guardrail.name,
            "guardrail_type": guardrail.guardrail_type.value,
            "severity": result.severity.value,
            "violations": result.violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 1000 violations
        if len(self.violation_log) > 1000:
            self.violation_log = self.violation_log[-1000:]
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """Get violation statistics."""
        if not self.violation_log:
            return {"total": 0, "by_guardrail": {}, "by_severity": {}}
        
        by_guardrail = {}
        by_severity = {}
        
        for v in self.violation_log:
            name = v["guardrail_name"]
            severity = v["severity"]
            
            by_guardrail[name] = by_guardrail.get(name, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total": len(self.violation_log),
            "by_guardrail": by_guardrail,
            "by_severity": by_severity
        }
    
    def add_restricted_content(self, content: str):
        """Add to restricted content list."""
        if content not in self.RESTRICTED_CONTENT:
            self.RESTRICTED_CONTENT.append(content)
    
    def add_pii_pattern(self, name: str, pattern: str):
        """Add a custom PII pattern."""
        self.PII_PATTERNS[name] = pattern

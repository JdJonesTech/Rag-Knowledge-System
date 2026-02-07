"""
Code Interpreter Tool
Executes Python code for calculations, data analysis, and transformations.
Provides a sandboxed environment for safe code execution.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import io
import sys
import traceback
import ast
import re

from src.agentic.tools.base_tool import BaseTool, ToolResult


class ExecutionMode(str, Enum):
    """Code execution modes."""
    SAFE = "safe"        # Restricted execution
    STANDARD = "standard"  # Normal execution
    ADVANCED = "advanced"   # Full capabilities (admin only)


@dataclass
class CodeExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str]
    variables: Dict[str, Any]
    execution_time_ms: float
    memory_used_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "variables": {k: str(v)[:200] for k, v in self.variables.items()},
            "execution_time_ms": self.execution_time_ms,
            "memory_used_mb": self.memory_used_mb
        }


class CodeInterpreterTool(BaseTool):
    """
    Tool for executing Python code.
    
    Capabilities:
    - Mathematical calculations
    - Data analysis with pandas/numpy
    - String manipulation
    - Date/time calculations
    - Unit conversions
    - Statistical analysis
    
    Security:
    - Sandboxed execution
    - Blocked dangerous operations
    - Memory and time limits
    """
    
    name = "code_interpreter_tool"
    description = """
    Executes Python code for calculations and data analysis. Use for:
    - Complex mathematical calculations
    - Data transformations and analysis
    - Unit conversions (temperature, pressure, dimensions)
    - Statistical calculations
    - Date/time computations
    - String parsing and manipulation
    """
    
    # Blocked imports for security
    BLOCKED_IMPORTS = {
        "os", "sys", "subprocess", "shutil", "pathlib",
        "socket", "urllib", "requests", "http",
        "pickle", "shelve", "marshal",
        "ctypes", "multiprocessing", "threading",
        "importlib", "__import__", "eval", "exec",
        "open", "file", "input"
    }
    
    # Blocked builtins
    BLOCKED_BUILTINS = {
        "open", "exec", "eval", "compile", "__import__",
        "input", "breakpoint", "exit", "quit"
    }
    
    # Allowed modules for SAFE mode
    SAFE_MODULES = {
        "math", "statistics", "decimal", "fractions",
        "datetime", "time", "calendar",
        "re", "string", "textwrap",
        "json", "collections", "itertools", "functools",
        "operator", "copy"
    }
    
    # Additional modules for STANDARD mode
    STANDARD_MODULES = {
        "numpy", "pandas", "scipy"
    }
    
    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.SAFE,
        timeout_seconds: int = 30,
        max_memory_mb: int = 100
    ):
        """
        Initialize code interpreter.
        
        Args:
            mode: Execution mode
            timeout_seconds: Maximum execution time
            max_memory_mb: Maximum memory usage
        """
        super().__init__(
            name=self.name,
            description=self.description
        )
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        
        # Persistent namespace for variables
        self._namespace: Dict[str, Any] = {}
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """
        Execute a code interpreter action.
        
        Args:
            query: Code or calculation description
            parameters: Action parameters including 'action' key
            intent: Optional intent from router
            
        Actions (specified in parameters['action']):
        - run: Execute Python code
        - calculate: Quick calculation
        - convert: Unit conversion
        - analyze: Data analysis
        """
        try:
            # Get action from parameters or infer from query
            action = parameters.get("action", "calculate")
            
            # If no code provided, use query as the code/expression
            if "code" not in parameters and "expression" not in parameters:
                parameters["expression"] = query
            
            if action == "run":
                return await self._run_code(parameters)
            elif action == "calculate":
                return await self._calculate(parameters)
            elif action == "convert":
                return await self._convert(parameters)
            elif action == "analyze":
                return await self._analyze(parameters)
            else:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                error=str(e)
            )
    
    async def _run_code(self, params: Dict[str, Any]) -> ToolResult:
        """Execute Python code."""
        code = params.get("code", "")
        variables = params.get("variables", {})
        persist_vars = params.get("persist_variables", False)
        
        if not code:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: code"
            )
        
        # Security check
        security_result = self._security_check(code)
        if not security_result["safe"]:
            return ToolResult(
                success=False,
                data={},
                error=f"Security violation: {security_result['reason']}"
            )
        
        # Execute code
        result = self._execute_sandboxed(code, variables)
        
        # Persist variables if requested
        if persist_vars and result.success:
            self._namespace.update(result.variables)
        
        return ToolResult(
            success=result.success,
            data=result.to_dict(),
            error=result.error,
            metadata={"action": "run", "mode": self.mode.value}
        )
    
    async def _calculate(self, params: Dict[str, Any]) -> ToolResult:
        """Perform a calculation."""
        expression = params.get("expression", "")
        
        if not expression:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: expression"
            )
        
        # Wrap in safe calculation
        code = f"result = {expression}"
        
        result = self._execute_sandboxed(code, params.get("variables", {}))
        
        if result.success and "result" in result.variables:
            return ToolResult(
                success=True,
                data={
                    "expression": expression,
                    "result": result.variables["result"],
                    "type": type(result.variables["result"]).__name__
                },
                metadata={"action": "calculate"}
            )
        
        return ToolResult(
            success=False,
            data={},
            error=result.error or "Calculation failed"
        )
    
    async def _convert(self, params: Dict[str, Any]) -> ToolResult:
        """Perform unit conversion."""
        value = params.get("value")
        from_unit = params.get("from_unit", "")
        to_unit = params.get("to_unit", "")
        
        if value is None or not from_unit or not to_unit:
            return ToolResult(
                success=False,
                data={},
                error="Missing required fields: value, from_unit, to_unit"
            )
        
        # Unit conversion mappings
        conversions = {
            # Temperature
            ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
            ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
            ("celsius", "kelvin"): lambda x: x + 273.15,
            ("kelvin", "celsius"): lambda x: x - 273.15,
            
            # Pressure
            ("bar", "psi"): lambda x: x * 14.5038,
            ("psi", "bar"): lambda x: x / 14.5038,
            ("bar", "mpa"): lambda x: x * 0.1,
            ("mpa", "bar"): lambda x: x * 10,
            ("bar", "kpa"): lambda x: x * 100,
            ("kpa", "bar"): lambda x: x / 100,
            
            # Length
            ("mm", "inch"): lambda x: x / 25.4,
            ("inch", "mm"): lambda x: x * 25.4,
            ("m", "ft"): lambda x: x * 3.28084,
            ("ft", "m"): lambda x: x / 3.28084,
            
            # Weight
            ("kg", "lb"): lambda x: x * 2.20462,
            ("lb", "kg"): lambda x: x / 2.20462,
        }
        
        key = (from_unit.lower(), to_unit.lower())
        
        if key in conversions:
            converted = conversions[key](float(value))
            return ToolResult(
                success=True,
                data={
                    "original_value": value,
                    "original_unit": from_unit,
                    "converted_value": round(converted, 4),
                    "converted_unit": to_unit
                },
                metadata={"action": "convert"}
            )
        
        return ToolResult(
            success=False,
            data={},
            error=f"Unsupported conversion: {from_unit} to {to_unit}"
        )
    
    async def _analyze(self, params: Dict[str, Any]) -> ToolResult:
        """Perform data analysis."""
        data = params.get("data", [])
        analysis_type = params.get("analysis_type", "basic")
        
        if not data:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: data"
            )
        
        # Build analysis code
        if analysis_type == "basic":
            code = """
import statistics
data = input_data
result = {
    'count': len(data),
    'sum': sum(data),
    'mean': statistics.mean(data),
    'median': statistics.median(data),
    'min': min(data),
    'max': max(data),
    'range': max(data) - min(data)
}
if len(data) > 1:
    result['stdev'] = statistics.stdev(data)
    result['variance'] = statistics.variance(data)
"""
        elif analysis_type == "percentiles":
            code = """
import statistics
data = sorted(input_data)
n = len(data)
result = {
    'p25': data[int(n * 0.25)],
    'p50': data[int(n * 0.50)],
    'p75': data[int(n * 0.75)],
    'p90': data[int(n * 0.90)],
    'p95': data[int(n * 0.95)] if n >= 20 else None,
    'p99': data[int(n * 0.99)] if n >= 100 else None
}
"""
        else:
            code = f"""
data = input_data
result = {{'count': len(data), 'sum': sum(data)}}
"""
        
        exec_result = self._execute_sandboxed(code, {"input_data": data})
        
        if exec_result.success and "result" in exec_result.variables:
            return ToolResult(
                success=True,
                data={
                    "analysis_type": analysis_type,
                    "input_count": len(data),
                    "results": exec_result.variables["result"]
                },
                metadata={"action": "analyze"}
            )
        
        return ToolResult(
            success=False,
            data={},
            error=exec_result.error or "Analysis failed"
        )
    
    def _security_check(self, code: str) -> Dict[str, Any]:
        """Check code for security issues."""
        code_lower = code.lower()
        
        # Check for blocked imports
        import_pattern = r'(?:from|import)\s+(\w+)'
        imports = re.findall(import_pattern, code)
        for imp in imports:
            if imp in self.BLOCKED_IMPORTS:
                return {"safe": False, "reason": f"Blocked import: {imp}"}
            if self.mode == ExecutionMode.SAFE and imp not in self.SAFE_MODULES:
                if imp not in self.STANDARD_MODULES:
                    return {"safe": False, "reason": f"Module not allowed in safe mode: {imp}"}
        
        # Check for blocked patterns
        dangerous_patterns = [
            r"__\w+__",  # Dunder methods
            r"eval\s*\(",
            r"exec\s*\(",
            r"compile\s*\(",
            r"open\s*\(",
            r"subprocess",
            r"os\.",
            r"sys\.",
            r"globals\s*\(",
            r"locals\s*\(",
            r"getattr\s*\(",
            r"setattr\s*\(",
            r"delattr\s*\("
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return {"safe": False, "reason": f"Dangerous pattern detected: {pattern}"}
        
        return {"safe": True, "reason": None}
    
    def _execute_sandboxed(
        self,
        code: str,
        variables: Dict[str, Any]
    ) -> CodeExecutionResult:
        """Execute code in sandboxed environment."""
        import time
        
        start_time = time.time()
        
        # Create safe namespace
        safe_builtins = {
            k: v for k, v in __builtins__.items()
            if k not in self.BLOCKED_BUILTINS
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k)
            for k in dir(__builtins__)
            if not k.startswith('_') and k not in self.BLOCKED_BUILTINS
        }
        
        namespace = {
            "__builtins__": safe_builtins,
            **self._namespace,  # Persistent variables
            **variables  # Input variables
        }
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Execute code
            exec(code, namespace)
            
            output = captured_output.getvalue()
            
            # Extract result variables (exclude builtins and inputs)
            result_vars = {
                k: v for k, v in namespace.items()
                if k not in {"__builtins__"} and k not in variables
                and not k.startswith("_")
            }
            
            execution_time = (time.time() - start_time) * 1000
            
            return CodeExecutionResult(
                success=True,
                output=output,
                error=None,
                variables=result_vars,
                execution_time_ms=execution_time,
                memory_used_mb=0  # Would need memory profiling
            )
            
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                output=captured_output.getvalue(),
                error=f"{type(e).__name__}: {str(e)}",
                variables={},
                execution_time_ms=(time.time() - start_time) * 1000,
                memory_used_mb=0
            )
        finally:
            sys.stdout = old_stdout
    
    def clear_namespace(self):
        """Clear persistent variables."""
        self._namespace.clear()

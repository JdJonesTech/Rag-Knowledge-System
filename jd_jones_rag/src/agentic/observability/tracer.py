"""
Agent Tracer
Traces every step of agent reasoning, tool usage, and retrieval.
Enables debugging, auditing, and optimization.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json
import asyncio
from contextlib import contextmanager


class SpanType(str, Enum):
    """Types of trace spans."""
    AGENT = "agent"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    LLM = "llm"
    EMBEDDING = "embedding"
    VALIDATION = "validation"
    ACTION = "action"
    REASONING = "reasoning"


class SpanStatus(str, Enum):
    """Status of a trace span."""
    STARTED = "started"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TraceSpan:
    """A single span in a trace."""
    span_id: str
    parent_id: Optional[str]
    trace_id: str
    name: str
    span_type: SpanType
    status: SpanStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    children: List['TraceSpan'] = field(default_factory=list)
    
    def end(self, status: SpanStatus = SpanStatus.SUCCESS, output: Any = None, error: str = None):
        """End the span."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        if output:
            self.output_data = {"result": output} if not isinstance(output, dict) else output
        if error:
            self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "span_type": self.span_type.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
            "error": self.error,
            "children": [c.to_dict() for c in self.children]
        }


@dataclass
class Trace:
    """A complete trace for a request."""
    trace_id: str
    name: str
    user_id: Optional[str]
    session_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0
    root_span: Optional[TraceSpan] = None
    spans: List[TraceSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Aggregated metrics
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_retrievals: int = 0
    llm_time_ms: float = 0
    tool_time_ms: float = 0
    retrieval_time_ms: float = 0
    
    def end(self):
        """End the trace."""
        self.end_time = datetime.now()
        self.total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute aggregated metrics from spans."""
        for span in self.spans:
            if span.span_type == SpanType.LLM:
                self.llm_time_ms += span.duration_ms
                self.total_tokens += span.metadata.get("tokens", 0)
            elif span.span_type == SpanType.TOOL:
                self.tool_time_ms += span.duration_ms
                self.total_tool_calls += 1
            elif span.span_type == SpanType.RETRIEVAL:
                self.retrieval_time_ms += span.duration_ms
                self.total_retrievals += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "root_span": self.root_span.to_dict() if self.root_span else None,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
            "metrics": {
                "total_tokens": self.total_tokens,
                "total_tool_calls": self.total_tool_calls,
                "total_retrievals": self.total_retrievals,
                "llm_time_ms": self.llm_time_ms,
                "tool_time_ms": self.tool_time_ms,
                "retrieval_time_ms": self.retrieval_time_ms
            }
        }


class AgentTracer:
    """
    Traces agent execution for debugging and monitoring.
    
    Compatible with:
    - LangSmith
    - Langfuse
    - Custom backends
    """
    
    def __init__(
        self,
        backend: str = "memory",  # memory, langsmith, langfuse, custom
        api_key: Optional[str] = None,
        project_name: str = "jd_jones_rag",
        max_traces: int = 1000
    ):
        """
        Initialize tracer.
        
        Args:
            backend: Tracing backend
            api_key: API key for external backends
            project_name: Project name for organization
            max_traces: Maximum traces to keep in memory
        """
        self.backend = backend
        self.api_key = api_key
        self.project_name = project_name
        self.max_traces = max_traces
        
        # In-memory storage
        self.traces: Dict[str, Trace] = {}
        self.current_trace: Optional[Trace] = None
        self.span_stack: List[TraceSpan] = []
        
        # Callbacks
        self.on_span_end: Optional[Callable] = None
        self.on_trace_end: Optional[Callable] = None
    
    def start_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Trace:
        """
        Start a new trace.
        
        Args:
            name: Trace name
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            
        Returns:
            New Trace object
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        
        trace = Trace(
            trace_id=trace_id,
            name=name,
            user_id=user_id,
            session_id=session_id,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        
        self.traces[trace_id] = trace
        self.current_trace = trace
        self.span_stack = []
        
        # Evict old traces if needed
        if len(self.traces) > self.max_traces:
            self._evict_old_traces()
        
        return trace
    
    def start_span(
        self,
        name: str,
        span_type: SpanType,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """
        Start a new span within the current trace.
        
        Args:
            name: Span name
            span_type: Type of span
            input_data: Input data for the span
            metadata: Additional metadata
            
        Returns:
            New TraceSpan object
        """
        if not self.current_trace:
            raise ValueError("No active trace. Call start_trace first.")
        
        parent_id = self.span_stack[-1].span_id if self.span_stack else None
        
        span = TraceSpan(
            span_id=f"span_{uuid.uuid4().hex[:12]}",
            parent_id=parent_id,
            trace_id=self.current_trace.trace_id,
            name=name,
            span_type=span_type,
            status=SpanStatus.STARTED,
            start_time=datetime.now(),
            input_data=input_data or {},
            metadata=metadata or {}
        )
        
        self.current_trace.spans.append(span)
        
        # Add as child of parent span
        if self.span_stack:
            self.span_stack[-1].children.append(span)
        else:
            self.current_trace.root_span = span
        
        self.span_stack.append(span)
        
        return span
    
    def end_span(
        self,
        status: SpanStatus = SpanStatus.SUCCESS,
        output: Any = None,
        error: str = None
    ) -> Optional[TraceSpan]:
        """
        End the current span.
        
        Args:
            status: Final status
            output: Output data
            error: Error message if any
            
        Returns:
            The ended span
        """
        if not self.span_stack:
            return None
        
        span = self.span_stack.pop()
        span.end(status, output, error)
        
        if self.on_span_end:
            self.on_span_end(span)
        
        return span
    
    def end_trace(self) -> Optional[Trace]:
        """
        End the current trace.
        
        Returns:
            The ended trace
        """
        if not self.current_trace:
            return None
        
        # End any remaining spans
        while self.span_stack:
            self.end_span(SpanStatus.ERROR, error="Trace ended with open spans")
        
        self.current_trace.end()
        
        if self.on_trace_end:
            self.on_trace_end(self.current_trace)
        
        # Export to backend
        self._export_trace(self.current_trace)
        
        trace = self.current_trace
        self.current_trace = None
        
        return trace
    
    @contextmanager
    def trace_context(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Context manager for traces."""
        trace = self.start_trace(name, user_id, session_id)
        try:
            yield trace
        finally:
            self.end_trace()
    
    @contextmanager
    def span_context(
        self,
        name: str,
        span_type: SpanType,
        input_data: Optional[Dict[str, Any]] = None
    ):
        """Context manager for spans."""
        span = self.start_span(name, span_type, input_data)
        try:
            yield span
        except Exception as e:
            self.end_span(SpanStatus.ERROR, error=str(e))
            raise
        else:
            self.end_span(SpanStatus.SUCCESS)
    
    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens: int = 0,
        latency_ms: float = 0
    ):
        """Log an LLM call."""
        span = self.start_span(
            name=f"llm_{model}",
            span_type=SpanType.LLM,
            input_data={"prompt": prompt[:500]},
            metadata={"model": model, "tokens": tokens}
        )
        span.duration_ms = latency_ms
        self.end_span(SpanStatus.SUCCESS, {"response": response[:500]})
    
    def log_tool_call(
        self,
        tool_name: str,
        input_params: Dict[str, Any],
        output: Any,
        success: bool,
        latency_ms: float = 0
    ):
        """Log a tool call."""
        span = self.start_span(
            name=f"tool_{tool_name}",
            span_type=SpanType.TOOL,
            input_data=input_params
        )
        span.duration_ms = latency_ms
        status = SpanStatus.SUCCESS if success else SpanStatus.ERROR
        self.end_span(status, output if success else None, str(output) if not success else None)
    
    def log_retrieval(
        self,
        query: str,
        results_count: int,
        latency_ms: float = 0
    ):
        """Log a retrieval operation."""
        span = self.start_span(
            name="retrieval",
            span_type=SpanType.RETRIEVAL,
            input_data={"query": query}
        )
        span.duration_ms = latency_ms
        self.end_span(SpanStatus.SUCCESS, {"results_count": results_count})
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        return self.traces.get(trace_id)
    
    def get_recent_traces(self, limit: int = 100) -> List[Trace]:
        """Get recent traces."""
        sorted_traces = sorted(
            self.traces.values(),
            key=lambda t: t.start_time,
            reverse=True
        )
        return sorted_traces[:limit]
    
    def _evict_old_traces(self):
        """Remove oldest traces when limit exceeded."""
        sorted_traces = sorted(
            self.traces.items(),
            key=lambda x: x[1].start_time
        )
        
        to_remove = len(self.traces) - self.max_traces + 100
        for i in range(to_remove):
            del self.traces[sorted_traces[i][0]]
    
    def _export_trace(self, trace: Trace):
        """Export trace to configured backend."""
        if self.backend == "memory":
            pass  # Already in memory
        elif self.backend == "langsmith":
            self._export_langsmith(trace)
        elif self.backend == "langfuse":
            self._export_langfuse(trace)
        # Add more backends as needed
    
    def _export_langsmith(self, trace: Trace):
        """Export to LangSmith."""
        # In production, use langsmith SDK
        pass
    
    def _export_langfuse(self, trace: Trace):
        """Export to Langfuse."""
        # In production, use langfuse SDK
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        traces = list(self.traces.values())
        
        if not traces:
            return {"total_traces": 0}
        
        total_duration = sum(t.total_duration_ms for t in traces if t.total_duration_ms)
        total_tokens = sum(t.total_tokens for t in traces)
        total_tools = sum(t.total_tool_calls for t in traces)
        
        return {
            "total_traces": len(traces),
            "avg_duration_ms": total_duration / len(traces) if traces else 0,
            "total_tokens_used": total_tokens,
            "total_tool_calls": total_tools,
            "traces_with_errors": sum(1 for t in traces if any(
                s.status == SpanStatus.ERROR for s in t.spans
            ))
        }

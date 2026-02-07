"""
LangFuse Integration
Concrete implementation for exporting traces to LangFuse.
Wires into AgentTracer._export_langfuse() to provide full observability.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.agentic.observability.tracer import (
    Trace, TraceSpan, SpanType, SpanStatus
)

logger = logging.getLogger(__name__)


class LangFuseExporter:
    """
    Exports AgentTracer traces to LangFuse.

    Maps our trace/span model to LangFuse's model:
    - Trace -> langfuse.trace()
    - SpanType.LLM -> langfuse generation()
    - SpanType.TOOL -> langfuse span()
    - SpanType.RETRIEVAL -> langfuse span() with retrieval metadata
    - SpanType.AGENT -> langfuse span()
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "http://localhost:3002",
        project_name: str = "jd_jones_rag",
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.project_name = project_name
        self._client = None

        if enabled and public_key and secret_key:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info(f"LangFuse connected: {host}")
            except ImportError:
                logger.warning("langfuse package not installed. Run: pip install langfuse")
            except Exception as e:
                logger.error(f"LangFuse initialization failed: {e}")

    def export_trace(self, trace: Trace) -> Optional[str]:
        """
        Export a complete Trace to LangFuse.

        Returns the LangFuse trace ID on success, None on failure.
        """
        if not self._client or not self.enabled:
            return None

        try:
            lf_trace = self._client.trace(
                id=trace.trace_id,
                name=trace.name,
                user_id=trace.user_id,
                session_id=trace.session_id,
                metadata=trace.metadata,
                input={"query": trace.metadata.get("query", "")},
                output={
                    "total_duration_ms": trace.total_duration_ms,
                    "total_tokens": trace.total_tokens,
                    "total_tool_calls": trace.total_tool_calls,
                },
            )

            # Export each span
            for span in trace.spans:
                self._export_span(lf_trace, span)

            self._client.flush()
            logger.debug(
                f"LangFuse: exported trace {trace.trace_id} "
                f"({trace.total_duration_ms:.0f}ms, {len(trace.spans)} spans)"
            )
            return trace.trace_id

        except Exception as e:
            logger.error(f"LangFuse export error: {e}")
            return None

    def _export_span(self, lf_trace, span: TraceSpan):
        """Export a single span to LangFuse."""
        if span.span_type == SpanType.LLM:
            lf_trace.generation(
                name=span.name,
                start_time=span.start_time,
                end_time=span.end_time,
                model=span.metadata.get("model", "unknown"),
                input=span.input_data.get("prompt", ""),
                output=span.output_data.get("response", ""),
                usage={
                    "total_tokens": span.metadata.get("tokens", 0),
                },
                metadata=span.metadata,
                status_message=span.error if span.error else None,
            )
        else:
            lf_span = lf_trace.span(
                name=span.name,
                start_time=span.start_time,
                end_time=span.end_time,
                input=span.input_data,
                output=span.output_data,
                metadata={
                    **span.metadata,
                    "span_type": span.span_type.value,
                    "status": span.status.value,
                },
                status_message=span.error if span.error else None,
            )

            for child in span.children:
                self._export_span_nested(lf_span, child)

    def _export_span_nested(self, parent, span: TraceSpan):
        """Export a nested child span under a parent."""
        if span.span_type == SpanType.LLM:
            parent.generation(
                name=span.name,
                start_time=span.start_time,
                end_time=span.end_time,
                model=span.metadata.get("model", "unknown"),
                input=span.input_data.get("prompt", ""),
                output=span.output_data.get("response", ""),
                usage={"total_tokens": span.metadata.get("tokens", 0)},
                metadata=span.metadata,
            )
        else:
            child_span = parent.span(
                name=span.name,
                start_time=span.start_time,
                end_time=span.end_time,
                input=span.input_data,
                output=span.output_data,
                metadata={**span.metadata, "span_type": span.span_type.value},
            )
            for child in span.children:
                self._export_span_nested(child_span, child)

    def log_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str = "",
    ):
        """Log an evaluation score to LangFuse for a trace."""
        if not self._client:
            return
        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as e:
            logger.error(f"LangFuse score error: {e}")

    def log_cost(
        self,
        trace_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ):
        """Log cost data for a trace."""
        total_cost = (
            (input_tokens / 1000) * cost_per_1k_input +
            (output_tokens / 1000) * cost_per_1k_output
        )
        if self._client and total_cost > 0:
            try:
                self._client.score(
                    trace_id=trace_id,
                    name="cost_usd",
                    value=total_cost,
                    comment=f"model={model}, in={input_tokens}, out={output_tokens}",
                )
            except Exception as e:
                logger.error(f"LangFuse cost logging error: {e}")

    def shutdown(self):
        """Flush and close the LangFuse client."""
        if self._client:
            try:
                self._client.flush()
                self._client.shutdown()
            except Exception as e:
                logger.error(f"LangFuse shutdown error: {e}")

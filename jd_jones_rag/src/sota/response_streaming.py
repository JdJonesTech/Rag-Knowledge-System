"""
Response Streaming Module

Implements streaming responses for better perceived latency.

SOTA Features:
- Token-by-token streaming
- SSE (Server-Sent Events) support
- Partial response assembly
- Stream interruption handling

Reference:
- OpenAI Streaming: https://platform.openai.com/docs/api-reference/streaming
- LangChain Streaming: https://python.langchain.com/docs/expression_language/streaming
"""

import logging
import asyncio
import json
from typing import AsyncGenerator, Generator, Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)

from src.config.settings import settings, get_llm


@dataclass
class StreamChunk:
    """A chunk of streaming response."""
    content: str
    chunk_index: int
    is_final: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Event format."""
        data = {
            "content": self.content,
            "index": self.chunk_index,
            "is_final": self.is_final,
            **self.metadata
        }
        return f"data: {json.dumps(data)}\n\n"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "chunk_index": self.chunk_index,
            "is_final": self.is_final,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class StreamingStats:
    """Statistics for a streaming session."""
    total_chunks: int = 0
    total_characters: int = 0
    total_tokens: int = 0
    start_time: float = 0
    first_chunk_time: float = 0
    end_time: float = 0
    
    @property
    def time_to_first_chunk_ms(self) -> float:
        if self.first_chunk_time and self.start_time:
            return (self.first_chunk_time - self.start_time) * 1000
        return 0
    
    @property
    def total_time_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0
    
    @property
    def chars_per_second(self) -> float:
        if self.total_time_ms > 0:
            return self.total_characters / (self.total_time_ms / 1000)
        return 0


class StreamingResponseGenerator:
    """
    Streaming Response Generator for RAG.
    
    Enables token-by-token streaming for better UX:
    1. Get retrieval results
    2. Start streaming response immediately
    3. User sees tokens as they're generated
    
    Benefits:
    - Better perceived latency
    - Users can start reading immediately
    - Ability to stop generation early
    
    Usage:
        streamer = StreamingResponseGenerator()
        
        async for chunk in streamer.stream(query, context):
            print(chunk.content, end="", flush=True)
    """
    
    def __init__(
        self,
        chunk_size: int = 1,  # Characters per chunk (1 = token-level)
        buffer_size: int = 100,
        timeout_seconds: float = 60.0,
        include_sources_at_end: bool = True
    ):
        """
        Initialize streaming generator.
        
        Args:
            chunk_size: Minimum characters per chunk
            buffer_size: Internal buffer size
            timeout_seconds: Maximum streaming time
            include_sources_at_end: Append sources after response
        """
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.timeout_seconds = timeout_seconds
        self.include_sources_at_end = include_sources_at_end
        
        self._llm = None
        self._stats = StreamingStats()
        self._interrupt_flag = False
    
    def _get_llm(self):
        """Get streaming-enabled LLM."""
        if self._llm is None:
            self._llm = get_llm(temperature=0.7, streaming=True)
        return self._llm
    
    async def stream(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        on_chunk: Optional[Callable[[StreamChunk], None]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream response for a query.
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional custom system prompt
            on_chunk: Optional callback for each chunk
            
        Yields:
            StreamChunk objects
        """
        self._interrupt_flag = False
        self._stats = StreamingStats(start_time=time.time())
        
        # Build messages
        from langchain_core.messages import SystemMessage, HumanMessage
        
        default_system = """You are a helpful industrial products assistant. 
Answer questions about seals, packings, and gaskets based on the provided context.
Be specific about product codes and specifications."""
        
        messages = [
            SystemMessage(content=system_prompt or default_system)
        ]
        
        if context:
            messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"))
        else:
            messages.append(HumanMessage(content=query))
        
        chunk_index = 0
        llm = self._get_llm()
        
        try:
            # Stream from LLM
            async for token in llm.astream(messages):
                if self._interrupt_flag:
                    break
                
                content = token.content if hasattr(token, 'content') else str(token)
                
                if content:
                    # Record first chunk time
                    if chunk_index == 0:
                        self._stats.first_chunk_time = time.time()
                    
                    chunk = StreamChunk(
                        content=content,
                        chunk_index=chunk_index,
                        is_final=False
                    )
                    
                    self._stats.total_chunks += 1
                    self._stats.total_characters += len(content)
                    chunk_index += 1
                    
                    if on_chunk:
                        on_chunk(chunk)
                    
                    yield chunk
            
            # Final chunk
            final_chunk = StreamChunk(
                content="",
                chunk_index=chunk_index,
                is_final=True,
                metadata={
                    "stats": {
                        "total_chunks": self._stats.total_chunks,
                        "total_characters": self._stats.total_characters,
                        "time_to_first_chunk_ms": self._stats.time_to_first_chunk_ms
                    }
                }
            )
            
            self._stats.end_time = time.time()
            
            if on_chunk:
                on_chunk(final_chunk)
            
            yield final_chunk
            
        except asyncio.TimeoutError:
            logger.warning("Streaming timed out")
            yield StreamChunk(
                content="\n\n[Response timed out]",
                chunk_index=chunk_index,
                is_final=True,
                metadata={"error": "timeout"}
            )
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield StreamChunk(
                content=f"\n\n[Error: {str(e)}]",
                chunk_index=chunk_index,
                is_final=True,
                metadata={"error": str(e)}
            )
    
    async def stream_with_sources(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream response and append formatted sources at the end.
        
        Args:
            query: User query
            sources: Retrieved sources to cite
            context: Optional pre-formatted context
            
        Yields:
            StreamChunk objects
        """
        # Build context from sources if not provided
        if context is None and sources:
            context = "\n\n".join([
                f"[{i+1}] {src.get('content', '')[:500]}"
                for i, src in enumerate(sources[:5])
            ])
        
        # Stream main response
        async for chunk in self.stream(query, context):
            if chunk.is_final and self.include_sources_at_end and sources:
                # Add sources before final chunk
                chunk.is_final = False
                yield chunk
                
                # Yield sources
                sources_text = "\n\n**Sources:**\n"
                for i, src in enumerate(sources[:5]):
                    source_name = src.get("source", src.get("document_id", f"Source {i+1}"))
                    sources_text += f"- [{i+1}] {source_name}\n"
                
                yield StreamChunk(
                    content=sources_text,
                    chunk_index=chunk.chunk_index + 1,
                    is_final=True,
                    metadata={"sources_count": len(sources)}
                )
            else:
                yield chunk
    
    def interrupt(self):
        """Interrupt the current stream."""
        self._interrupt_flag = True
    
    async def stream_to_sse(
        self,
        query: str,
        context: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream as Server-Sent Events.
        
        Args:
            query: User query
            context: Retrieved context
            
        Yields:
            SSE formatted strings
        """
        async for chunk in self.stream(query, context):
            yield chunk.to_sse()
    
    def collect_stream(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """
        Collect entire stream into a string (sync version).
        """
        async def _collect():
            parts = []
            async for chunk in self.stream(query, context):
                parts.append(chunk.content)
            return "".join(parts)
        
        return asyncio.get_event_loop().run_until_complete(_collect())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "total_chunks": self._stats.total_chunks,
            "total_characters": self._stats.total_characters,
            "time_to_first_chunk_ms": round(self._stats.time_to_first_chunk_ms, 2),
            "total_time_ms": round(self._stats.total_time_ms, 2),
            "chars_per_second": round(self._stats.chars_per_second, 2)
        }


class SSEFormatter:
    """Helper class for SSE formatting."""
    
    @staticmethod
    def format_event(
        data: Any,
        event: Optional[str] = None,
        id: Optional[str] = None,
        retry: Optional[int] = None
    ) -> str:
        """
        Format data as SSE event.
        
        Args:
            data: Data to send
            event: Optional event type
            id: Optional event ID
            retry: Optional retry timeout in ms
            
        Returns:
            SSE formatted string
        """
        lines = []
        
        if id:
            lines.append(f"id: {id}")
        if event:
            lines.append(f"event: {event}")
        if retry:
            lines.append(f"retry: {retry}")
        
        if isinstance(data, str):
            for line in data.split('\n'):
                lines.append(f"data: {line}")
        else:
            lines.append(f"data: {json.dumps(data)}")
        
        lines.append("")
        lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_heartbeat() -> str:
        """Format a heartbeat (keep-alive) event."""
        return ": heartbeat\n\n"
    
    @staticmethod
    def format_done() -> str:
        """Format the done event."""
        return "data: [DONE]\n\n"


# FastAPI integration helpers
def create_streaming_response(generator: AsyncGenerator[str, None]):
    """
    Create a FastAPI StreamingResponse.
    
    Usage:
        @app.post("/chat/stream")
        async def chat_stream(request: ChatRequest):
            streamer = StreamingResponseGenerator()
            generator = streamer.stream_to_sse(request.query, request.context)
            return create_streaming_response(generator)
    """
    try:
        from fastapi.responses import StreamingResponse
        
        async def generate():
            try:
                async for chunk in generator:
                    yield chunk
            except Exception as e:
                yield SSEFormatter.format_event({"error": str(e)}, event="error")
            finally:
                yield SSEFormatter.format_done()
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    except ImportError:
        logger.warning("FastAPI not installed")
        return None


# Singleton instance
_streamer: Optional[StreamingResponseGenerator] = None


def get_streamer() -> StreamingResponseGenerator:
    """Get singleton streaming generator instance."""
    global _streamer
    if _streamer is None:
        _streamer = StreamingResponseGenerator()
    return _streamer

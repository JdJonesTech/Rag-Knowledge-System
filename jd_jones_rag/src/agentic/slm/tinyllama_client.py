"""
TinyLlama Client for SLM Tasks
Lightweight client for entity extraction and quick generation using TinyLlama.

OPTIMIZATIONS:
- Singleton pattern for persistent connections
- HTTP connection pooling via httpx
- Client reuse across requests
"""

import logging
from typing import Dict, Any, List, Optional
import json
import re
from threading import Lock

logger = logging.getLogger(__name__)

# Singleton instance
_tinyllama_instance = None
_instance_lock = Lock()


class TinyLlamaClient:
    """
    Client for TinyLlama (via Ollama) for SLM tasks.
    
    OPTIMIZATION: Uses singleton pattern with persistent HTTP connections.
    
    Uses:
    - Entity extraction (product codes, specifications, standards)
    - Quick answer generation
    - Summary generation
    
    ~200ms response time vs ~2s for Llama 3.2
    """
    
    def __init__(
        self,
        model: str = "tinyllama",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        self._client = None
        self._http_client = None  # Persistent HTTP client with connection pooling
    
    def _get_client(self):
        """Lazy initialization of Ollama client with connection pooling."""
        if self._client is None:
            try:
                from langchain_openai import ChatOpenAI
                
                # OPTIMIZATION: Configure HTTP client with connection pooling
                import httpx
                self._http_client = httpx.Client(
                    timeout=30.0,
                    limits=httpx.Limits(
                        max_connections=10,
                        max_keepalive_connections=5,
                        keepalive_expiry=30.0
                    )
                )
                
                self._client = ChatOpenAI(
                    model=self.model,
                    base_url=f"{self.base_url}/v1",
                    api_key="ollama",
                    temperature=0.1,  # Low for extraction accuracy
                    max_tokens=256,   # Short responses for speed
                    http_client=self._http_client  # Use pooled client
                )
            except Exception as e:
                logger.error(f"Error initializing TinyLlama client: {e}")
                raise
        return self._client
    
    def close(self):
        """Close HTTP connections."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None
        self._client = None
    
    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from text using TinyLlama.
        
        Args:
            text: Input text
            entity_types: Types of entities to extract
                Default: product_code, temperature, pressure, material, standard
        
        Returns:
            Dict with extracted entities
        """
        entity_types = entity_types or [
            "product_code", "temperature", "pressure", 
            "material", "standard", "application"
        ]
        
        prompt = f"""Extract entities from this text. Return JSON only.

Text: {text}

Extract these entity types: {', '.join(entity_types)}

Return format:
{{"product_code": [...], "temperature": "...", "pressure": "...", "material": "...", "standard": [...], "application": "..."}}

JSON:"""

        try:
            client = self._get_client()
            from langchain_core.messages import HumanMessage
            
            response = await client.ainvoke([HumanMessage(content=prompt)])
            
            # Parse JSON from response
            result = self._parse_json_response(response.content)
            return result
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            # Fallback to regex extraction
            return self._regex_extract(text)
    
    async def quick_answer(
        self,
        question: str,
        context: Optional[str] = None,
        max_tokens: int = 150
    ) -> str:
        """
        Generate a quick answer using TinyLlama.
        
        Args:
            question: User question
            context: Optional context information
            max_tokens: Maximum response length
            
        Returns:
            Answer string
        """
        prompt = f"""Answer briefly and accurately.

Question: {question}
{f'Context: {context}' if context else ''}

Answer:"""

        try:
            client = self._get_client()
            from langchain_core.messages import HumanMessage
            
            response = await client.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Quick answer error: {e}")
            return f"Unable to generate answer: {e}"
    
    async def summarize(
        self,
        text: str,
        max_length: int = 100
    ) -> str:
        """
        Generate a brief summary using TinyLlama.
        
        Args:
            text: Text to summarize
            max_length: Maximum word count
            
        Returns:
            Summary string
        """
        prompt = f"""Summarize this in {max_length} words or less:

{text[:2000]}

Summary:"""

        try:
            client = self._get_client()
            from langchain_core.messages import HumanMessage
            
            response = await client.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            # Fallback: return first few sentences
            sentences = text.split('. ')[:3]
            return '. '.join(sentences) + '...'
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
        
        # Return empty dict if parsing fails
        return {}
    
    def _regex_extract(self, text: str) -> Dict[str, Any]:
        """Fallback regex-based extraction."""
        return {
            "product_code": re.findall(r'NA\s*\d+|NJ\s*\d+', text, re.IGNORECASE),
            "temperature": re.findall(r'(\d+)\s*°?[CF]', text),
            "pressure": re.findall(r'(\d+)\s*(psi|bar|mpa)', text, re.IGNORECASE),
            "standard": re.findall(r'API\s*\d+|FDA|ATEX|ASME', text, re.IGNORECASE),
            "material": re.findall(r'PTFE|graphite|aramid|rubber|silicone', text, re.IGNORECASE),
            "application": None
        }


class HybridSLMProcessor:
    """
    Hybrid processor using sklearn + TinyLlama for optimal performance.
    
    Architecture:
    - sklearn: Intent classification (< 10ms)
    - TinyLlama: Entity extraction, quick generation (~200ms)
    - Llama 3.2: Complex reasoning (escalated by main brain)
    """
    
    def __init__(self):
        self.tinyllama = TinyLlamaClient()
        self._sklearn_classifier = None
    
    def _get_sklearn_classifier(self):
        """Get or create sklearn classifier."""
        if self._sklearn_classifier is None:
            from src.agentic.slm.training import SLMInference
            self._sklearn_classifier = SLMInference()
        return self._sklearn_classifier
    
    async def process(
        self,
        query: str,
        task: str = "classify"
    ) -> Dict[str, Any]:
        """
        Process query using appropriate SLM.
        
        Args:
            query: User query
            task: Task type (classify, extract, answer, summarize)
            
        Returns:
            Processing result
        """
        import time
        start_time = time.time()
        
        result = {
            "query": query,
            "task": task,
            "backend": None,
            "result": None,
            "processing_time_ms": 0
        }
        
        if task == "classify":
            # Use sklearn (fastest)
            from src.agentic.slm.training import SLMType
            classifier = self._get_sklearn_classifier()
            classification = classifier.predict(SLMType.INTENT_CLASSIFIER, query)
            result["backend"] = "sklearn"
            result["result"] = classification
            
        elif task == "extract":
            # Use TinyLlama for extraction
            entities = await self.tinyllama.extract_entities(query)
            result["backend"] = "tinyllama"
            result["result"] = entities
            
        elif task == "answer":
            # Use TinyLlama for quick answers
            answer = await self.tinyllama.quick_answer(query)
            result["backend"] = "tinyllama"
            result["result"] = {"answer": answer}
            
        elif task == "summarize":
            # Use TinyLlama for summarization
            summary = await self.tinyllama.summarize(query)
            result["backend"] = "tinyllama"
            result["result"] = {"summary": summary}
            
        else:
            result["error"] = f"Unknown task: {task}"
        
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        return result


# Singleton instance
_hybrid_processor = None

def get_hybrid_processor() -> HybridSLMProcessor:
    """Get the global hybrid SLM processor."""
    global _hybrid_processor
    if _hybrid_processor is None:
        _hybrid_processor = HybridSLMProcessor()
    return _hybrid_processor

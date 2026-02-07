"""
Multi-Hop GraphRAG

Implements multi-hop reasoning over knowledge graphs for complex queries
that require traversing multiple relationships.

SOTA Features:
- Iterative subgraph expansion
- Entity path finding
- Relationship context aggregation
- Multi-hop question answering

Reference:
- GraphRAG: https://arxiv.org/abs/2404.16130
- Multi-hop QA: https://arxiv.org/abs/2112.12777
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class GraphEntity:
    """An entity in the knowledge graph."""
    id: str
    name: str
    type: str  # product, material, standard, application
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelation:
    """A relationship between entities."""
    source_id: str
    target_id: str
    relation_type: str  # MADE_OF, COMPATIBLE_WITH, CERTIFIED_BY, USED_IN
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class HopResult:
    """Result from a single hop traversal."""
    hop_number: int
    entities: List[GraphEntity]
    relations: List[GraphRelation]
    query_focus: str


@dataclass
class MultiHopResult:
    """Result from multi-hop retrieval."""
    query: str
    hops: List[HopResult]
    final_entities: List[GraphEntity]
    paths: List[List[str]]  # Entity ID paths
    reasoning_chain: List[str]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "num_hops": len(self.hops),
            "final_entities": [e.name for e in self.final_entities[:10]],
            "paths": self.paths[:5],
            "reasoning_chain": self.reasoning_chain,
            "confidence": self.confidence
        }


class MultiHopGraphRAG:
    """
    Multi-Hop Knowledge Graph Retrieval.
    
    Answers complex questions by traversing knowledge graph relationships:
    
    Example query: "What materials are compatible with seals used in Shell refineries?"
    - Hop 1: Find products used in Shell applications
    - Hop 2: Find materials these products are made of
    - Hop 3: Find compatible materials
    
    Usage:
        multihop = MultiHopGraphRAG(knowledge_graph)
        result = await multihop.retrieve("What materials are compatible with...")
    """
    
    # Query patterns that typically require multi-hop
    MULTI_HOP_PATTERNS = [
        "compatible with",
        "used in",
        "certified for",
        "alternative to",
        "same material as",
        "works with",
        "replacement for"
    ]
    
    def __init__(
        self,
        knowledge_graph: Any = None,
        max_hops: int = 3,
        max_entities_per_hop: int = 20,
        use_llm_planning: bool = True
    ):
        """
        Initialize Multi-Hop GraphRAG.
        
        Args:
            knowledge_graph: NetworkX graph or similar
            max_hops: Maximum number of hops
            max_entities_per_hop: Max entities to explore per hop
            use_llm_planning: Use LLM for hop planning
        """
        self.knowledge_graph = knowledge_graph
        self.max_hops = max_hops
        self.max_entities_per_hop = max_entities_per_hop
        self.use_llm_planning = use_llm_planning
        
        self._llm = None
        self._entity_index: Dict[str, GraphEntity] = {}
        self._relation_index: Dict[str, List[GraphRelation]] = defaultdict(list)
        
        self._stats = {
            "queries_processed": 0,
            "total_hops": 0,
            "avg_hops_per_query": 0,
            "cache_hits": 0
        }
    
    def _get_llm(self):
        """Get LLM for planning."""
        if self._llm is None:
            try:
                from src.config.settings import get_llm
                self._llm = get_llm(temperature=0.3)
            except Exception as e:
                logger.error(f"Failed to get LLM: {e}")
        return self._llm
    
    def build_index(self, entities: List[Dict], relations: List[Dict]):
        """
        Build in-memory index from entities and relations.
        
        Args:
            entities: List of entity dictionaries
            relations: List of relation dictionaries
        """
        # Index entities
        for entity in entities:
            ge = GraphEntity(
                id=entity.get("id", ""),
                name=entity.get("name", ""),
                type=entity.get("type", "unknown"),
                properties=entity.get("properties", {})
            )
            self._entity_index[ge.id] = ge
        
        # Index relations
        for relation in relations:
            gr = GraphRelation(
                source_id=relation.get("source_id", ""),
                target_id=relation.get("target_id", ""),
                relation_type=relation.get("relation_type", "RELATED_TO"),
                properties=relation.get("properties", {}),
                weight=relation.get("weight", 1.0)
            )
            self._relation_index[gr.source_id].append(gr)
        
        logger.info(f"Built index: {len(self._entity_index)} entities, {sum(len(v) for v in self._relation_index.values())} relations")
    
    def _needs_multi_hop(self, query: str) -> bool:
        """Determine if query needs multi-hop reasoning."""
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in self.MULTI_HOP_PATTERNS)
    
    async def _plan_hops(self, query: str) -> List[Dict[str, str]]:
        """
        Use LLM to plan the hop sequence.
        
        Returns list of {"focus": ..., "relation": ...} for each hop.
        """
        if not self.use_llm_planning:
            return self._rule_based_plan(query)
        
        llm = self._get_llm()
        if llm is None:
            return self._rule_based_plan(query)
        
        try:
            from langchain_core.messages import HumanMessage
            
            prompt = f"""Plan a multi-hop knowledge graph traversal for this query about industrial products:

Query: "{query}"

Available relation types:
- MADE_OF (product -> material)
- COMPATIBLE_WITH (material -> material)
- CERTIFIED_BY (product -> standard)
- USED_IN (product -> application/industry)
- REPLACES (product -> product)
- SIMILAR_TO (product -> product)

Respond with a JSON array of hops:
[{{"focus": "what to find", "relation": "RELATION_TYPE"}}]

Example for "What materials are compatible with seals used in refineries?":
[
  {{"focus": "products used in refineries", "relation": "USED_IN"}},
  {{"focus": "materials of those products", "relation": "MADE_OF"}},
  {{"focus": "compatible materials", "relation": "COMPATIBLE_WITH"}}
]

Plan:"""

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            
            # Parse JSON
            text = response.content
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
                
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}")
        
        return self._rule_based_plan(query)
    
    def _rule_based_plan(self, query: str) -> List[Dict[str, str]]:
        """Rule-based hop planning fallback."""
        query_lower = query.lower()
        hops = []
        
        if "compatible" in query_lower:
            hops.append({"focus": "products", "relation": "MADE_OF"})
            hops.append({"focus": "materials", "relation": "COMPATIBLE_WITH"})
            
        elif "used in" in query_lower or "for" in query_lower:
            hops.append({"focus": "applications", "relation": "USED_IN"})
            hops.append({"focus": "products", "relation": "MADE_OF"})
            
        elif "certified" in query_lower:
            hops.append({"focus": "products", "relation": "CERTIFIED_BY"})
            hops.append({"focus": "standards", "relation": "CERTIFIED_BY"})
            
        elif "alternative" in query_lower or "replacement" in query_lower:
            hops.append({"focus": "products", "relation": "SIMILAR_TO"})
            hops.append({"focus": "similar products", "relation": "REPLACES"})
        
        else:
            hops.append({"focus": "entities", "relation": "RELATED_TO"})
        
        return hops[:self.max_hops]
    
    def _find_seed_entities(self, query: str) -> List[GraphEntity]:
        """Find initial entities from query."""
        query_lower = query.lower()
        matches = []
        
        for entity_id, entity in self._entity_index.items():
            # Match by name
            if any(word in entity.name.lower() for word in query_lower.split() if len(word) > 2):
                matches.append(entity)
            # Match by type
            elif entity.type.lower() in query_lower:
                matches.append(entity)
        
        return matches[:self.max_entities_per_hop]
    
    def _traverse_hop(
        self,
        entities: List[GraphEntity],
        relation_type: str
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        """
        Traverse one hop from given entities.
        
        Returns (new_entities, relations_used).
        """
        new_entities = []
        relations_used = []
        seen_ids: Set[str] = set()
        
        for entity in entities:
            relations = self._relation_index.get(entity.id, [])
            
            for rel in relations:
                # Filter by relation type if specified
                if relation_type != "RELATED_TO" and rel.relation_type != relation_type:
                    continue
                
                target_id = rel.target_id
                if target_id in seen_ids:
                    continue
                
                target_entity = self._entity_index.get(target_id)
                if target_entity:
                    new_entities.append(target_entity)
                    relations_used.append(rel)
                    seen_ids.add(target_id)
                
                if len(new_entities) >= self.max_entities_per_hop:
                    break
            
            if len(new_entities) >= self.max_entities_per_hop:
                break
        
        return new_entities, relations_used
    
    def _find_paths(
        self,
        seed_entities: List[GraphEntity],
        final_entities: List[GraphEntity],
        hops: List[HopResult]
    ) -> List[List[str]]:
        """Find entity paths from seeds to final entities."""
        paths = []
        
        for seed in seed_entities[:3]:
            for final in final_entities[:3]:
                # Simple path: seed -> intermediate -> final
                path = [seed.id]
                for hop in hops:
                    for entity in hop.entities[:1]:
                        if entity.id not in path:
                            path.append(entity.id)
                if final.id not in path:
                    path.append(final.id)
                paths.append(path)
        
        return paths[:5]
    
    async def retrieve(
        self,
        query: str,
        seed_entities: Optional[List[GraphEntity]] = None
    ) -> MultiHopResult:
        """
        Perform multi-hop retrieval.
        
        Args:
            query: User query
            seed_entities: Optional starting entities
            
        Returns:
            MultiHopResult with traversal results
        """
        self._stats["queries_processed"] += 1
        
        # Find seed entities if not provided
        if seed_entities is None:
            seed_entities = self._find_seed_entities(query)
        
        if not seed_entities:
            return MultiHopResult(
                query=query,
                hops=[],
                final_entities=[],
                paths=[],
                reasoning_chain=["No seed entities found"],
                confidence=0
            )
        
        # Plan hops
        hop_plan = await self._plan_hops(query)
        
        # Execute hops
        current_entities = seed_entities
        hops = []
        reasoning_chain = [f"Starting with {len(seed_entities)} seed entities"]
        
        for i, hop_spec in enumerate(hop_plan):
            relation_type = hop_spec.get("relation", "RELATED_TO")
            focus = hop_spec.get("focus", f"hop {i+1}")
            
            new_entities, relations = self._traverse_hop(current_entities, relation_type)
            
            if not new_entities:
                reasoning_chain.append(f"Hop {i+1}: No new entities found for {focus}")
                break
            
            hops.append(HopResult(
                hop_number=i + 1,
                entities=new_entities,
                relations=relations,
                query_focus=focus
            ))
            
            reasoning_chain.append(
                f"Hop {i+1}: Found {len(new_entities)} {focus} via {relation_type}"
            )
            
            current_entities = new_entities
            self._stats["total_hops"] += 1
        
        # Find paths
        paths = self._find_paths(seed_entities, current_entities, hops)
        
        # Calculate confidence based on path length and entity count
        confidence = min(1.0, len(current_entities) / 10 * (1 - len(hops) * 0.1))
        
        # Update stats
        self._stats["avg_hops_per_query"] = (
            self._stats["total_hops"] / self._stats["queries_processed"]
        )
        
        return MultiHopResult(
            query=query,
            hops=hops,
            final_entities=current_entities,
            paths=paths,
            reasoning_chain=reasoning_chain,
            confidence=confidence
        )
    
    async def answer(
        self,
        query: str,
        use_llm_synthesis: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a query using multi-hop retrieval.
        
        Args:
            query: User query
            use_llm_synthesis: Use LLM to synthesize answer
            
        Returns:
            Dict with answer and retrieval details
        """
        # Retrieve
        result = await self.retrieve(query)
        
        if not result.final_entities:
            return {
                "answer": "I couldn't find relevant information in the knowledge graph.",
                "retrieval": result.to_dict(),
                "confidence": 0
            }
        
        # Synthesize answer
        if use_llm_synthesis:
            llm = self._get_llm()
            if llm:
                try:
                    from langchain_core.messages import HumanMessage
                    
                    # Build context from entities
                    context = "\n".join([
                        f"- {e.name} ({e.type}): {json.dumps(e.properties)}"
                        for e in result.final_entities[:10]
                    ])
                    
                    prompt = f"""Based on the following knowledge graph traversal, answer the question.

Question: {query}

Reasoning chain:
{chr(10).join(result.reasoning_chain)}

Found entities:
{context}

Provide a concise, accurate answer:"""

                    response = await llm.ainvoke([HumanMessage(content=prompt)])
                    
                    return {
                        "answer": response.content,
                        "retrieval": result.to_dict(),
                        "confidence": result.confidence
                    }
                    
                except Exception as e:
                    logger.error(f"LLM synthesis failed: {e}")
        
        # Fallback: simple entity listing
        entities_text = ", ".join([e.name for e in result.final_entities[:5]])
        return {
            "answer": f"Based on the knowledge graph: {entities_text}",
            "retrieval": result.to_dict(),
            "confidence": result.confidence
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            **self._stats,
            "indexed_entities": len(self._entity_index),
            "indexed_relations": sum(len(v) for v in self._relation_index.values())
        }


# Singleton instance
_multihop_rag: Optional[MultiHopGraphRAG] = None


def get_multihop_graph_rag() -> MultiHopGraphRAG:
    """Get singleton multi-hop GraphRAG instance."""
    global _multihop_rag
    if _multihop_rag is None:
        _multihop_rag = MultiHopGraphRAG()
    return _multihop_rag

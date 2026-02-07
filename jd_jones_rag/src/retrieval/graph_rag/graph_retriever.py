"""
Graph Retriever for GraphRAG
Enhances retrieval results using knowledge graph context.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.retrieval.graph_rag.entity_extractor import EntityExtractor, EntityType
from src.retrieval.graph_rag.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class GraphEnhancedResult:
    """Retrieval result enhanced with graph context."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    # Graph enhancements
    entities: List[Dict[str, Any]]
    related_products: List[Dict[str, Any]]
    applicable_standards: List[Dict[str, Any]]
    materials: List[Dict[str, Any]]
    graph_context: str  # Natural language context from graph


class GraphRetriever:
    """
    Enhances retrieval results with knowledge graph context.
    Provides expanded context for better LLM responses.
    """
    
    def __init__(
        self, 
        knowledge_graph: KnowledgeGraph,
        entity_extractor: Optional[EntityExtractor] = None
    ):
        """
        Initialize graph retriever.
        
        Args:
            knowledge_graph: Knowledge graph instance
            entity_extractor: Optional entity extractor (creates default if not provided)
        """
        self.kg = knowledge_graph
        self.extractor = entity_extractor or EntityExtractor()
        logger.info("GraphRetriever initialized")
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Enhance query with graph context.
        
        Args:
            query: User query
            
        Returns:
            Enhanced query information
        """
        # Extract entities from query
        entities = self.extractor.extract_entities(query)
        
        enhancement = {
            "original_query": query,
            "detected_entities": [
                {"id": e.id, "name": e.name, "type": e.entity_type.value}
                for e in entities
            ],
            "expanded_context": [],
            "related_products": [],
            "standards": [],
            "materials": []
        }
        
        # For each detected entity, get graph context
        for entity in entities:
            entity_data = self.kg.get_entity(entity.id)
            
            if entity_data:
                # Get subgraph around entity
                subgraph = self.kg.get_subgraph(entity.id, max_hops=2)
                
                # Add related context
                neighbors = self.kg.get_neighbors(entity.id, direction="both")
                enhancement["expanded_context"].extend(neighbors)
                
                # Get related products if this is a product
                if entity.entity_type == EntityType.PRODUCT:
                    related = self.kg.get_related_products(entity.id)
                    enhancement["related_products"].extend(related)
        
        # Remove duplicates
        enhancement["expanded_context"] = self._deduplicate_by_id(
            enhancement["expanded_context"]
        )
        enhancement["related_products"] = self._deduplicate_by_id(
            enhancement["related_products"]
        )
        
        logger.debug(f"Enhanced query with {len(enhancement['expanded_context'])} context entities")
        return enhancement
    
    def enhance_result(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        relevance_score: float
    ) -> GraphEnhancedResult:
        """
        Enhance a single retrieval result with graph context.
        
        Args:
            content: Document content
            metadata: Document metadata
            relevance_score: Original relevance score
            
        Returns:
            Enhanced result with graph context
        """
        # Extract entities from content
        entities = self.extractor.extract_entities(content)
        
        # Collect related information
        related_products = []
        standards = []
        materials = []
        
        for entity in entities:
            if entity.entity_type == EntityType.PRODUCT:
                related = self.kg.get_related_products(entity.id)
                related_products.extend(related)
            elif entity.entity_type == EntityType.STANDARD:
                standards.append({
                    "id": entity.id,
                    "name": entity.name
                })
            elif entity.entity_type == EntityType.MATERIAL:
                materials.append({
                    "id": entity.id,
                    "name": entity.name
                })
        
        # Build natural language context
        graph_context = self._build_context_string(
            entities, related_products, standards, materials
        )
        
        return GraphEnhancedResult(
            content=content,
            metadata=metadata,
            relevance_score=relevance_score,
            entities=[
                {"id": e.id, "name": e.name, "type": e.entity_type.value}
                for e in entities
            ],
            related_products=self._deduplicate_by_id(related_products)[:5],
            applicable_standards=standards,
            materials=materials,
            graph_context=graph_context
        )
    
    def enhance_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[GraphEnhancedResult]:
        """
        Enhance multiple retrieval results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of enhanced results
        """
        enhanced = []
        for result in results:
            enhanced_result = self.enhance_result(
                content=result.get("content", ""),
                metadata=result.get("metadata", {}),
                relevance_score=result.get("relevance_score", 0.0)
            )
            enhanced.append(enhanced_result)
        
        return enhanced
    
    def get_context_for_entity(self, entity_id: str) -> Dict[str, Any]:
        """
        Get complete context for a specific entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Complete entity context
        """
        entity = self.kg.get_entity(entity_id)
        if not entity:
            return {"error": f"Entity {entity_id} not found"}
        
        context = {
            "entity": entity,
            "neighbors": self.kg.get_neighbors(entity_id, direction="both"),
            "related_products": [],
            "path_to_standards": []
        }
        
        # Get related products
        if entity.get("entity_type") == "product":
            context["related_products"] = self.kg.get_related_products(entity_id)
        
        return context
    
    def find_products_for_application(
        self, 
        application: str,
        material: Optional[str] = None,
        standard: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find products suitable for an application with optional filters.
        
        Args:
            application: Application type (e.g., "valve sealing")
            material: Optional material filter
            standard: Optional compliance standard filter
            
        Returns:
            List of matching products
        """
        # Search for application
        app_entities = self.kg.search_entities(
            application,
            entity_types=[EntityType.APPLICATION]
        )
        
        if not app_entities:
            return []
        
        # Get products for this application
        products = []
        for app in app_entities:
            neighbors = self.kg.get_neighbors(
                app["id"],
                direction="in"
            )
            for neighbor in neighbors:
                if neighbor.get("entity_type") == "product":
                    products.append(neighbor)
        
        # Filter by material if specified
        if material:
            filtered = []
            for product in products:
                product_materials = self.kg.get_neighbors(
                    product["id"],
                    direction="out"
                )
                material_names = [m.get("name", "").lower() for m in product_materials]
                if any(material.lower() in name for name in material_names):
                    filtered.append(product)
            products = filtered
        
        # Filter by standard if specified
        if standard:
            filtered = []
            for product in products:
                product_standards = self.kg.get_neighbors(
                    product["id"],
                    direction="out"
                )
                standard_names = [s.get("name", "").lower() for s in product_standards]
                if any(standard.lower() in name for name in standard_names):
                    filtered.append(product)
            products = filtered
        
        return self._deduplicate_by_id(products)
    
    def _build_context_string(
        self,
        entities: List,
        related_products: List[Dict[str, Any]],
        standards: List[Dict[str, Any]],
        materials: List[Dict[str, Any]]
    ) -> str:
        """Build natural language context from graph data."""
        parts = []
        
        # Products mentioned
        product_entities = [e for e in entities if e.entity_type == EntityType.PRODUCT]
        if product_entities:
            product_names = [e.name for e in product_entities]
            parts.append(f"Products: {', '.join(product_names)}")
        
        # Materials
        if materials:
            material_names = [m["name"] for m in materials]
            parts.append(f"Materials: {', '.join(material_names)}")
        
        # Standards
        if standards:
            standard_names = [s["name"] for s in standards]
            parts.append(f"Standards: {', '.join(standard_names)}")
        
        # Related products
        if related_products:
            related_names = [p.get("name", p["id"]) for p in related_products[:3]]
            parts.append(f"Related products: {', '.join(related_names)}")
        
        return " | ".join(parts) if parts else ""
    
    def _deduplicate_by_id(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates based on id field."""
        seen = set()
        unique = []
        for item in items:
            item_id = item.get("id")
            if item_id and item_id not in seen:
                seen.add(item_id)
                unique.append(item)
        return unique

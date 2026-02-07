"""
GraphRAG Pipeline
Combines entity extraction, knowledge graph, and graph-enhanced retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.retrieval.graph_rag.entity_extractor import EntityExtractor
from src.retrieval.graph_rag.knowledge_graph import KnowledgeGraph
from src.retrieval.graph_rag.graph_retriever import GraphRetriever, GraphEnhancedResult

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """
    Complete GraphRAG pipeline for enhanced retrieval.
    
    Usage:
        pipeline = GraphRAGPipeline()
        
        # Index documents
        pipeline.index_document(content, metadata)
        
        # Query with graph enhancement
        results = pipeline.query("What is NA 701 made of?")
    """
    
    def __init__(
        self,
        graph_path: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        Initialize GraphRAG pipeline.
        
        Args:
            graph_path: Path to persist knowledge graph
            auto_save: Whether to auto-save after indexing
        """
        self.graph_path = graph_path or "data/knowledge_graph.json"
        self.auto_save = auto_save
        
        # Initialize components
        self.extractor = EntityExtractor()
        self.kg = KnowledgeGraph(graph_path=self.graph_path)
        self.retriever = GraphRetriever(self.kg, self.extractor)
        
        logger.info(f"GraphRAG pipeline initialized (graph: {self.graph_path})")
    
    def index_document(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index a document into the knowledge graph.
        
        Args:
            content: Document content
            metadata: Optional document metadata
            doc_id: Optional document identifier
            
        Returns:
            Indexing statistics
        """
        doc_id = doc_id or metadata.get("document_id", "unknown")
        
        # Extract entities and relationships
        entities = self.extractor.extract_entities(content, source_doc=doc_id)
        relationships = self.extractor.extract_relationships(entities, content)
        
        # Add to knowledge graph
        self.kg.add_entities_and_relationships(entities, relationships)
        
        # Extract and add properties if product entities found
        properties = self.extractor.extract_properties(content)
        for entity in entities:
            if entity.entity_type.value == "product" and properties:
                entity_data = self.kg.get_entity(entity.id)
                if entity_data:
                    entity_data["properties"] = properties
        
        # Auto-save if enabled
        if self.auto_save:
            self._ensure_graph_dir()
            self.kg.save()
        
        stats = {
            "document_id": doc_id,
            "entities_extracted": len(entities),
            "relationships_extracted": len(relationships),
            "properties_extracted": len(properties)
        }
        
        logger.info(f"Indexed document {doc_id}: {stats}")
        return stats
    
    def index_documents(
        self, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Bulk index documents.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
            
        Returns:
            Bulk indexing statistics
        """
        total_entities = 0
        total_relationships = 0
        
        # Temporarily disable auto-save for bulk indexing
        original_auto_save = self.auto_save
        self.auto_save = False
        
        for doc in documents:
            stats = self.index_document(
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                doc_id=doc.get("id")
            )
            total_entities += stats["entities_extracted"]
            total_relationships += stats["relationships_extracted"]
        
        # Save once at the end
        self._ensure_graph_dir()
        self.kg.save()
        
        # Restore auto-save setting
        self.auto_save = original_auto_save
        
        stats = {
            "documents_indexed": len(documents),
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "graph_stats": self.kg.get_statistics()
        }
        
        logger.info(f"Bulk indexed {len(documents)} documents")
        return stats
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Enhance a query with graph context.
        
        Args:
            query: User query
            
        Returns:
            Enhanced query information
        """
        return self.retriever.enhance_query(query)
    
    def enhance_results(
        self, 
        results: List[Dict[str, Any]]
    ) -> List[GraphEnhancedResult]:
        """
        Enhance retrieval results with graph context.
        
        Args:
            results: Original retrieval results
            
        Returns:
            Enhanced results with graph context
        """
        return self.retriever.enhance_results(results)
    
    def query_with_graph_context(
        self,
        query: str,
        base_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Full query with graph enhancement.
        
        Args:
            query: User query
            base_results: Base retrieval results
            
        Returns:
            Enhanced query response
        """
        # Enhance query
        query_enhancement = self.enhance_query(query)
        
        # Enhance results
        enhanced_results = self.enhance_results(base_results)
        
        # Build context for LLM
        graph_context = self._build_llm_context(
            query_enhancement,
            enhanced_results
        )
        
        return {
            "query": query,
            "query_entities": query_enhancement["detected_entities"],
            "enhanced_results": [
                {
                    "content": r.content,
                    "score": r.relevance_score,
                    "entities": r.entities,
                    "graph_context": r.graph_context
                }
                for r in enhanced_results
            ],
            "graph_context_for_llm": graph_context,
            "related_products": query_enhancement["related_products"][:5]
        }
    
    def find_products(
        self,
        application: Optional[str] = None,
        material: Optional[str] = None,
        standard: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find products matching criteria using graph.
        
        Args:
            application: Application type
            material: Material type
            standard: Compliance standard
            
        Returns:
            Matching products
        """
        if application:
            return self.retriever.find_products_for_application(
                application, material, standard
            )
        
        # If no application specified, search by material or standard
        products = []
        
        if material:
            products = self.kg.get_products_by_material(material)
        
        if standard:
            standard_products = self.kg.get_products_by_standard(standard)
            if products:
                # Intersection
                product_ids = {p["id"] for p in products}
                products = [p for p in standard_products if p["id"] in product_ids]
            else:
                products = standard_products
        
        return products
    
    def get_product_info(self, product_code: str) -> Dict[str, Any]:
        """
        Get complete information about a product from graph.
        
        Args:
            product_code: Product code (e.g., "NA 701")
            
        Returns:
            Product information with relationships
        """
        # Normalize product code
        normalized = product_code.upper().replace("-", " ").replace("  ", " ").strip()
        entity_id = f"product:{normalized.replace(' ', '_')}"
        
        entity = self.kg.get_entity(entity_id)
        if not entity:
            # Try search
            results = self.kg.search_entities(product_code)
            if results:
                entity = results[0]
                entity_id = entity["id"]
            else:
                return {"error": f"Product {product_code} not found in graph"}
        
        # Get all related information
        neighbors = self.kg.get_neighbors(entity_id, direction="both")
        
        # Categorize relationships
        materials = [n for n in neighbors if n.get("entity_type") == "material"]
        applications = [n for n in neighbors if n.get("entity_type") == "application"]
        standards = [n for n in neighbors if n.get("entity_type") == "standard"]
        industries = [n for n in neighbors if n.get("entity_type") == "industry"]
        
        # Get related products
        related_products = self.kg.get_related_products(entity_id)
        
        return {
            "product": entity,
            "materials": materials,
            "applications": applications,
            "standards": standards,
            "industries": industries,
            "related_products": related_products[:5]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return self.kg.get_statistics()
    
    def save(self) -> None:
        """Save knowledge graph."""
        self._ensure_graph_dir()
        self.kg.save()
    
    def _ensure_graph_dir(self) -> None:
        """Ensure graph directory exists."""
        Path(self.graph_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _build_llm_context(
        self,
        query_enhancement: Dict[str, Any],
        enhanced_results: List[GraphEnhancedResult]
    ) -> str:
        """Build context string for LLM prompt."""
        parts = []
        
        # Query entities
        if query_enhancement["detected_entities"]:
            entity_names = [e["name"] for e in query_enhancement["detected_entities"]]
            parts.append(f"Query mentions: {', '.join(entity_names)}")
        
        # Graph context from results
        all_materials = set()
        all_standards = set()
        
        for result in enhanced_results:
            for m in result.materials:
                all_materials.add(m.get("name", ""))
            for s in result.applicable_standards:
                all_standards.add(s.get("name", ""))
        
        if all_materials:
            parts.append(f"Materials: {', '.join(all_materials)}")
        
        if all_standards:
            parts.append(f"Compliance: {', '.join(all_standards)}")
        
        # Related products
        if query_enhancement["related_products"]:
            related_names = [p.get("name", p.get("id", ""))[:20] 
                           for p in query_enhancement["related_products"][:3]]
            parts.append(f"Related: {', '.join(related_names)}")
        
        return " | ".join(parts) if parts else "No additional graph context"


# Global instance
_graph_rag_pipeline: Optional[GraphRAGPipeline] = None


def get_graph_rag_pipeline() -> GraphRAGPipeline:
    """Get or create global GraphRAG pipeline instance."""
    global _graph_rag_pipeline
    if _graph_rag_pipeline is None:
        _graph_rag_pipeline = GraphRAGPipeline()
    return _graph_rag_pipeline

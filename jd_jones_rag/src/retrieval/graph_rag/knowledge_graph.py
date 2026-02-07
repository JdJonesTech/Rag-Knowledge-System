"""
Knowledge Graph for GraphRAG
Stores entities and relationships using NetworkX for graph operations.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from src.retrieval.graph_rag.entity_extractor import (
    Entity, Relationship, EntityType, RelationType
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Knowledge graph for storing entities and relationships.
    Uses NetworkX for graph operations and persistence.
    """
    
    def __init__(self, graph_path: Optional[str] = None):
        """
        Initialize knowledge graph.
        
        Args:
            graph_path: Optional path to load/save graph
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for KnowledgeGraph. Install with: pip install networkx")
        
        self.graph = nx.DiGraph()
        self.graph_path = graph_path
        
        # Load existing graph if path provided
        if graph_path and Path(graph_path).exists():
            self.load(graph_path)
            logger.info(f"Loaded knowledge graph from {graph_path}")
        else:
            logger.info("Initialized empty knowledge graph")
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the graph.
        
        Args:
            entity: Entity to add
        """
        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            properties=entity.properties,
            source_doc=entity.source_doc,
            confidence=entity.confidence
        )
    
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the graph.
        
        Args:
            relationship: Relationship to add
        """
        if relationship.source_id in self.graph and relationship.target_id in self.graph:
            self.graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                relation_type=relationship.relation_type.value,
                properties=relationship.properties,
                confidence=relationship.confidence
            )
    
    def add_entities_and_relationships(
        self, 
        entities: List[Entity], 
        relationships: List[Relationship]
    ) -> None:
        """
        Bulk add entities and relationships.
        
        Args:
            entities: List of entities
            relationships: List of relationships
        """
        for entity in entities:
            self.add_entity(entity)
        
        for relationship in relationships:
            self.add_relationship(relationship)
        
        logger.info(f"Added {len(entities)} entities and {len(relationships)} relationships")
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity data dict or None
        """
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        return None
    
    def get_neighbors(
        self, 
        entity_id: str, 
        relation_types: Optional[List[RelationType]] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring entities.
        
        Args:
            entity_id: Starting entity
            relation_types: Optional filter by relation types
            direction: "in", "out", or "both"
            
        Returns:
            List of neighbor entity data
        """
        neighbors = []
        
        if entity_id not in self.graph:
            return neighbors
        
        edges = []
        if direction in ("out", "both"):
            edges.extend(self.graph.out_edges(entity_id, data=True))
        if direction in ("in", "both"):
            edges.extend(self.graph.in_edges(entity_id, data=True))
        
        for edge in edges:
            source, target, data = edge
            
            # Filter by relation type
            if relation_types:
                rel_type = data.get("relation_type")
                if rel_type not in [rt.value for rt in relation_types]:
                    continue
            
            # Get the other entity
            other_id = target if source == entity_id else source
            neighbor_data = dict(self.graph.nodes[other_id])
            neighbor_data["id"] = other_id
            neighbor_data["relation"] = data.get("relation_type")
            neighbors.append(neighbor_data)
        
        return neighbors
    
    def find_path(
        self, 
        source_id: str, 
        target_id: str, 
        max_length: int = 3
    ) -> Optional[List[str]]:
        """
        Find shortest path between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_length: Maximum path length
            
        Returns:
            List of entity IDs in path, or None
        """
        try:
            path = nx.shortest_path(
                self.graph.to_undirected(),
                source_id,
                target_id
            )
            if len(path) <= max_length + 1:
                return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return None
    
    def get_subgraph(
        self, 
        center_id: str, 
        max_hops: int = 2
    ) -> "KnowledgeGraph":
        """
        Get subgraph around a central entity.
        
        Args:
            center_id: Central entity ID
            max_hops: Maximum hops from center
            
        Returns:
            New KnowledgeGraph with subgraph
        """
        if center_id not in self.graph:
            return KnowledgeGraph()
        
        # BFS to find all nodes within max_hops
        nodes = set([center_id])
        frontier = {center_id}
        
        for _ in range(max_hops):
            new_frontier = set()
            for node in frontier:
                # Add successors and predecessors
                new_frontier.update(self.graph.successors(node))
                new_frontier.update(self.graph.predecessors(node))
            new_frontier -= nodes
            nodes.update(new_frontier)
            frontier = new_frontier
        
        # Create subgraph
        subgraph = KnowledgeGraph()
        subgraph.graph = self.graph.subgraph(nodes).copy()
        return subgraph
    
    def search_entities(
        self, 
        query: str, 
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search entities by name.
        
        Args:
            query: Search query
            entity_types: Optional filter by entity types
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        results = []
        query_lower = query.lower()
        
        for node_id, data in self.graph.nodes(data=True):
            # Filter by type
            if entity_types:
                if data.get("entity_type") not in [et.value for et in entity_types]:
                    continue
            
            # Match by name
            name = data.get("name", "").lower()
            if query_lower in name or name in query_lower:
                result = dict(data)
                result["id"] = node_id
                results.append(result)
        
        # Sort by name match quality
        results.sort(key=lambda x: abs(len(x.get("name", "")) - len(query)))
        return results[:limit]
    
    def get_products_by_material(self, material_name: str) -> List[Dict[str, Any]]:
        """Get all products made of a specific material."""
        products = []
        
        # Find material entity
        material_entities = self.search_entities(
            material_name, 
            entity_types=[EntityType.MATERIAL]
        )
        
        if not material_entities:
            return products
        
        # Get products connected to this material
        for material in material_entities:
            neighbors = self.get_neighbors(
                material["id"],
                relation_types=[RelationType.MADE_OF],
                direction="in"
            )
            products.extend(neighbors)
        
        return products
    
    def get_products_by_standard(self, standard_name: str) -> List[Dict[str, Any]]:
        """Get all products complying with a specific standard."""
        products = []
        
        # Find standard entity
        standard_entities = self.search_entities(
            standard_name,
            entity_types=[EntityType.STANDARD]
        )
        
        if not standard_entities:
            return products
        
        # Get products connected to this standard
        for standard in standard_entities:
            neighbors = self.get_neighbors(
                standard["id"],
                relation_types=[RelationType.COMPLIES_WITH],
                direction="in"
            )
            products.extend(neighbors)
        
        return products
    
    def get_related_products(self, product_id: str) -> List[Dict[str, Any]]:
        """
        Get products related to a given product (sharing materials, standards, etc.)
        """
        related = []
        
        if product_id not in self.graph:
            return related
        
        # Get all neighbors of this product
        neighbors = self.get_neighbors(product_id, direction="out")
        
        # For each neighbor, get other products connected to it
        for neighbor in neighbors:
            other_products = self.get_neighbors(
                neighbor["id"],
                direction="in"
            )
            for product in other_products:
                if product.get("entity_type") == "product" and product["id"] != product_id:
                    related.append(product)
        
        # Deduplicate
        seen = set()
        unique_related = []
        for product in related:
            if product["id"] not in seen:
                seen.add(product["id"])
                unique_related.append(product)
        
        return unique_related
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        entity_counts = {}
        for _, data in self.graph.nodes(data=True):
            entity_type = data.get("entity_type", "unknown")
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        relation_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get("relation_type", "unknown")
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        
        return {
            "total_entities": self.graph.number_of_nodes(),
            "total_relationships": self.graph.number_of_edges(),
            "entities_by_type": entity_counts,
            "relationships_by_type": relation_counts,
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }
    
    def save(self, path: Optional[str] = None) -> None:
        """Save graph to file."""
        save_path = path or self.graph_path
        if not save_path:
            raise ValueError("No path specified for saving graph")
        
        # Convert to node-link format for JSON
        data = nx.node_link_data(self.graph)
        
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved knowledge graph to {save_path}")
    
    def load(self, path: str) -> None:
        """Load graph from file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.graph = nx.node_link_graph(data)
        self.graph_path = path
        
        logger.info(f"Loaded knowledge graph with {self.graph.number_of_nodes()} entities")

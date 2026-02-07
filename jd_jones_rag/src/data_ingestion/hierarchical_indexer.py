"""
Hierarchical Indexer
Organizes data into multi-layered categories based on semantic relationships.
Helps agents navigate complex datasets more accurately.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from langchain_core.documents import Document

from src.config.settings import settings


class IndexLevel(str, Enum):
    """Levels in the hierarchical index."""
    DOMAIN = "domain"           # Top level: Sales, Engineering, HR
    CATEGORY = "category"       # Second level: Products, Policies, Procedures
    SUBCATEGORY = "subcategory" # Third level: Gaskets, Packings, Expansion Joints
    DOCUMENT = "document"       # Fourth level: Individual documents
    CHUNK = "chunk"             # Fifth level: Document chunks


@dataclass
class IndexNode:
    """A node in the hierarchical index."""
    node_id: str
    level: IndexLevel
    name: str
    description: str
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "level": self.level.value,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "children_count": len(self.children),
            "document_count": self.document_count,
            "metadata": self.metadata
        }


@dataclass
class HierarchicalSearchResult:
    """Result from hierarchical search."""
    path: List[IndexNode]
    documents: List[Document]
    relevance_scores: List[float]
    navigation_summary: str


class HierarchicalIndexer:
    """
    Creates and manages hierarchical document indices.
    
    Structure:
    - Domain (Engineering, Sales, HR, etc.)
      - Category (Products, Policies, Procedures)
        - Subcategory (Gaskets, Packings, Standards)
          - Documents
            - Chunks
    
    Benefits:
    - Faster navigation for agents
    - Context-aware retrieval
    - Better disambiguation
    - Efficient multi-hop queries
    """
    
    # Default hierarchy for JD Jones
    DEFAULT_HIERARCHY = {
        "Engineering": {
            "description": "Technical documentation and specifications",
            "categories": {
                "Products": {
                    "description": "Product technical information",
                    "subcategories": ["Gaskets", "Packings", "Expansion Joints", "Seals", "Insulation"]
                },
                "Standards": {
                    "description": "Industry standards and certifications",
                    "subcategories": ["API", "ASME", "ISO", "FDA", "Shell SPE"]
                },
                "Specifications": {
                    "description": "Technical specifications and requirements",
                    "subcategories": ["Materials", "Dimensions", "Performance", "Testing"]
                }
            }
        },
        "Sales": {
            "description": "Sales and commercial documentation",
            "categories": {
                "Pricing": {
                    "description": "Price lists and discount structures",
                    "subcategories": ["Standard Pricing", "Volume Discounts", "Special Pricing"]
                },
                "Customers": {
                    "description": "Customer information and agreements",
                    "subcategories": ["Contracts", "Agreements", "Requirements"]
                },
                "Marketing": {
                    "description": "Marketing materials and collateral",
                    "subcategories": ["Brochures", "Case Studies", "Presentations"]
                }
            }
        },
        "Quality": {
            "description": "Quality assurance and compliance",
            "categories": {
                "Procedures": {
                    "description": "Quality procedures and processes",
                    "subcategories": ["Testing", "Inspection", "Certification"]
                },
                "Compliance": {
                    "description": "Regulatory compliance documentation",
                    "subcategories": ["Certifications", "Audits", "Reports"]
                }
            }
        },
        "HR": {
            "description": "Human resources documentation",
            "categories": {
                "Policies": {
                    "description": "Company policies",
                    "subcategories": ["Employee Handbook", "Benefits", "Safety"]
                },
                "Training": {
                    "description": "Training materials",
                    "subcategories": ["Onboarding", "Technical Training", "Compliance"]
                }
            }
        }
    }
    
    def __init__(
        self,
        hierarchy: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hierarchical indexer.
        
        Args:
            hierarchy: Custom hierarchy definition
        """
        self.hierarchy = hierarchy or self.DEFAULT_HIERARCHY
        
        # Lazy initialized embedding generator
        self._embedding_generator = None
        
        # Index storage
        self.nodes: Dict[str, IndexNode] = {}
        self.root_nodes: List[str] = []
        
        # Document to node mapping
        self.doc_to_node: Dict[str, str] = {}
        
        # Initialize hierarchy
        self._initialize_hierarchy()
    
    def _get_embedding_generator(self):
        """Lazy initialization of embedding generator."""
        if self._embedding_generator is None:
            from src.data_ingestion.embedding_generator import EmbeddingGenerator
            self._embedding_generator = EmbeddingGenerator()
        return self._embedding_generator
    
    def _initialize_hierarchy(self):
        """Initialize the hierarchical structure."""
        for domain_name, domain_data in self.hierarchy.items():
            # Create domain node
            domain_id = self._create_node(
                level=IndexLevel.DOMAIN,
                name=domain_name,
                description=domain_data["description"],
                parent_id=None
            )
            self.root_nodes.append(domain_id)
            
            # Create category nodes
            for cat_name, cat_data in domain_data.get("categories", {}).items():
                cat_id = self._create_node(
                    level=IndexLevel.CATEGORY,
                    name=cat_name,
                    description=cat_data["description"],
                    parent_id=domain_id
                )
                
                # Create subcategory nodes
                for subcat_name in cat_data.get("subcategories", []):
                    self._create_node(
                        level=IndexLevel.SUBCATEGORY,
                        name=subcat_name,
                        description=f"{subcat_name} under {cat_name}",
                        parent_id=cat_id
                    )
    
    def _create_node(
        self,
        level: IndexLevel,
        name: str,
        description: str,
        parent_id: Optional[str]
    ) -> str:
        """Create a new index node."""
        node_id = f"node_{level.value}_{uuid.uuid4().hex[:8]}"
        
        node = IndexNode(
            node_id=node_id,
            level=level,
            name=name,
            description=description,
            parent_id=parent_id
        )
        
        self.nodes[node_id] = node
        
        # Update parent's children list
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node_id)
        
        return node_id
    
    def add_document(
        self,
        document: Document,
        domain: str,
        category: str,
        subcategory: Optional[str] = None
    ) -> str:
        """
        Add a document to the hierarchical index.
        
        Args:
            document: Document to add
            domain: Domain name
            category: Category name
            subcategory: Optional subcategory name
            
        Returns:
            Node ID where document was added
        """
        # Find the appropriate node
        target_node = self._find_node(domain, category, subcategory)
        
        if not target_node:
            # Create the path if it doesn't exist
            target_node = self._ensure_path(domain, category, subcategory)
        
        # Create document node
        doc_id = document.metadata.get("doc_id", f"doc_{uuid.uuid4().hex[:8]}")
        doc_node_id = self._create_node(
            level=IndexLevel.DOCUMENT,
            name=document.metadata.get("title", doc_id),
            description=document.page_content[:200] + "...",
            parent_id=target_node
        )
        
        # Store document metadata
        self.nodes[doc_node_id].metadata = {
            "doc_id": doc_id,
            "source": document.metadata.get("source"),
            "content_length": len(document.page_content),
            **document.metadata
        }
        
        # Update document count up the hierarchy
        self._update_document_count(doc_node_id)
        
        # Map document to node
        self.doc_to_node[doc_id] = doc_node_id
        
        return doc_node_id
    
    def _find_node(
        self,
        domain: str,
        category: Optional[str] = None,
        subcategory: Optional[str] = None
    ) -> Optional[str]:
        """Find a node by path."""
        # Find domain
        domain_node = None
        for node_id in self.root_nodes:
            if self.nodes[node_id].name.lower() == domain.lower():
                domain_node = node_id
                break
        
        if not domain_node:
            return None
        
        if not category:
            return domain_node
        
        # Find category
        cat_node = None
        for child_id in self.nodes[domain_node].children:
            if self.nodes[child_id].name.lower() == category.lower():
                cat_node = child_id
                break
        
        if not cat_node:
            return None
        
        if not subcategory:
            return cat_node
        
        # Find subcategory
        for child_id in self.nodes[cat_node].children:
            if self.nodes[child_id].name.lower() == subcategory.lower():
                return child_id
        
        return None
    
    def _ensure_path(
        self,
        domain: str,
        category: str,
        subcategory: Optional[str] = None
    ) -> str:
        """Ensure the path exists, creating nodes as needed."""
        # Find or create domain
        domain_node = self._find_node(domain)
        if not domain_node:
            domain_node = self._create_node(
                level=IndexLevel.DOMAIN,
                name=domain,
                description=f"{domain} domain",
                parent_id=None
            )
            self.root_nodes.append(domain_node)
        
        # Find or create category
        cat_node = self._find_node(domain, category)
        if not cat_node:
            cat_node = self._create_node(
                level=IndexLevel.CATEGORY,
                name=category,
                description=f"{category} under {domain}",
                parent_id=domain_node
            )
        
        if not subcategory:
            return cat_node
        
        # Find or create subcategory
        subcat_node = self._find_node(domain, category, subcategory)
        if not subcat_node:
            subcat_node = self._create_node(
                level=IndexLevel.SUBCATEGORY,
                name=subcategory,
                description=f"{subcategory} under {category}",
                parent_id=cat_node
            )
        
        return subcat_node
    
    def _update_document_count(self, node_id: str):
        """Update document count up the hierarchy."""
        current = node_id
        while current:
            self.nodes[current].document_count += 1
            current = self.nodes[current].parent_id
    
    async def navigate(
        self,
        query: str,
        start_node: Optional[str] = None,
        max_depth: int = 3
    ) -> HierarchicalSearchResult:
        """
        Navigate the hierarchy to find relevant documents.
        
        Args:
            query: Search query
            start_node: Starting node (None for root)
            max_depth: Maximum depth to traverse
            
        Returns:
            Search result with path and documents
        """
        # Generate query embedding
        generator = self._get_embedding_generator()
        query_embedding = generator.generate_embedding(query)
        
        # Start navigation
        if start_node:
            current_nodes = [start_node]
        else:
            current_nodes = self.root_nodes.copy()
        
        path = []
        
        # Navigate down the hierarchy
        for depth in range(max_depth):
            if not current_nodes:
                break
            
            # Score each node
            best_node = None
            best_score = -1
            
            for node_id in current_nodes:
                node = self.nodes[node_id]
                
                # Generate or use cached embedding
                if not node.embedding:
                    node.embedding = generator.generate_embedding(
                        f"{node.name}: {node.description}"
                    )
                
                # Calculate similarity
                score = self._cosine_similarity(query_embedding, node.embedding)
                
                if score > best_score:
                    best_score = score
                    best_node = node_id
            
            if best_node:
                path.append(self.nodes[best_node])
                current_nodes = self.nodes[best_node].children
            else:
                break
        
        # Collect documents from the final node and its children
        documents = []
        relevance_scores = []
        
        if path:
            final_node = path[-1]
            doc_nodes = self._collect_document_nodes(final_node.node_id)
            
            for doc_node_id in doc_nodes[:10]:  # Limit results
                doc_node = self.nodes[doc_node_id]
                documents.append(Document(
                    page_content=doc_node.description,
                    metadata=doc_node.metadata
                ))
                relevance_scores.append(0.8)  # Placeholder score
        
        # Generate navigation summary
        path_names = " → ".join([n.name for n in path])
        summary = f"Navigated: {path_names}. Found {len(documents)} documents."
        
        return HierarchicalSearchResult(
            path=path,
            documents=documents,
            relevance_scores=relevance_scores,
            navigation_summary=summary
        )
    
    def _collect_document_nodes(self, node_id: str) -> List[str]:
        """Collect all document nodes under a given node."""
        results = []
        
        def traverse(nid: str):
            node = self.nodes[nid]
            if node.level == IndexLevel.DOCUMENT:
                results.append(nid)
            for child_id in node.children:
                traverse(child_id)
        
        traverse(node_id)
        return results
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def get_hierarchy_tree(self, max_depth: int = 3) -> Dict[str, Any]:
        """Get the hierarchy as a tree structure."""
        def build_tree(node_id: str, depth: int) -> Dict[str, Any]:
            if depth > max_depth:
                return None
            
            node = self.nodes[node_id]
            tree = {
                "name": node.name,
                "level": node.level.value,
                "document_count": node.document_count
            }
            
            if node.children and depth < max_depth:
                tree["children"] = [
                    build_tree(child_id, depth + 1)
                    for child_id in node.children
                    if build_tree(child_id, depth + 1)
                ]
            
            return tree
        
        return {
            "hierarchy": [build_tree(nid, 0) for nid in self.root_nodes]
        }
    
    def suggest_category(self, document: Document) -> Tuple[str, str, Optional[str]]:
        """
        Suggest the best category for a document.
        
        Args:
            document: Document to categorize
            
        Returns:
            Tuple of (domain, category, subcategory)
        """
        content = document.page_content.lower()
        
        # Simple keyword-based suggestion
        if any(kw in content for kw in ["gasket", "seal", "packing", "specification", "temperature", "pressure"]):
            return ("Engineering", "Products", "Gaskets")
        elif any(kw in content for kw in ["price", "discount", "quote", "cost"]):
            return ("Sales", "Pricing", "Standard Pricing")
        elif any(kw in content for kw in ["api", "asme", "iso", "standard", "certification"]):
            return ("Engineering", "Standards", "API")
        elif any(kw in content for kw in ["quality", "test", "inspection"]):
            return ("Quality", "Procedures", "Testing")
        elif any(kw in content for kw in ["policy", "employee", "hr", "training"]):
            return ("HR", "Policies", "Employee Handbook")
        
        return ("Engineering", "Products", None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        level_counts = {}
        for node in self.nodes.values():
            level = node.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            "total_nodes": len(self.nodes),
            "root_nodes": len(self.root_nodes),
            "nodes_by_level": level_counts,
            "total_documents": sum(
                1 for n in self.nodes.values() if n.level == IndexLevel.DOCUMENT
            )
        }

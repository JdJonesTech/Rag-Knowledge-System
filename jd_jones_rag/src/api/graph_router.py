"""
GraphRAG API Endpoints
Provides graph-enhanced retrieval and entity queries.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.retrieval.graph_rag import GraphRAGPipeline, EntityExtractor

router = APIRouter(prefix="/graph", tags=["GraphRAG"])

# Initialize components
_pipeline: Optional[GraphRAGPipeline] = None
_extractor: Optional[EntityExtractor] = None


def get_pipeline() -> GraphRAGPipeline:
    """Get or create GraphRAG pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = GraphRAGPipeline(graph_path="data/knowledge_graph.json")
    return _pipeline


def get_extractor() -> EntityExtractor:
    """Get or create entity extractor."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


# Request/Response Models
class IndexDocumentRequest(BaseModel):
    content: str
    document_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IndexDocumentResponse(BaseModel):
    document_id: str
    entities_extracted: int
    relationships_extracted: int


class EnhanceQueryRequest(BaseModel):
    query: str


class ProductSearchRequest(BaseModel):
    application: Optional[str] = None
    material: Optional[str] = None
    standard: Optional[str] = None


class EntitySearchRequest(BaseModel):
    query: str
    entity_types: Optional[List[str]] = None
    limit: int = 10


# Endpoints
@router.post("/index", response_model=IndexDocumentResponse)
async def index_document(request: IndexDocumentRequest):
    """
    Index a document into the knowledge graph.
    Extracts entities and relationships automatically.
    """
    pipeline = get_pipeline()
    
    stats = pipeline.index_document(
        content=request.content,
        metadata=request.metadata,
        doc_id=request.document_id
    )
    
    return IndexDocumentResponse(
        document_id=stats["document_id"],
        entities_extracted=stats["entities_extracted"],
        relationships_extracted=stats["relationships_extracted"]
    )


@router.post("/enhance-query")
async def enhance_query(request: EnhanceQueryRequest):
    """
    Enhance a query with graph context.
    Returns detected entities and related information.
    """
    pipeline = get_pipeline()
    enhancement = pipeline.enhance_query(request.query)
    return enhancement


@router.get("/product/{product_code}")
async def get_product_info(product_code: str):
    """
    Get detailed product information from knowledge graph.
    Includes materials, applications, standards, and related products.
    """
    pipeline = get_pipeline()
    info = pipeline.get_product_info(product_code)
    
    if "error" in info:
        raise HTTPException(status_code=404, detail=info["error"])
    
    return info


@router.post("/search/products")
async def search_products(request: ProductSearchRequest):
    """
    Search for products matching criteria.
    Filter by application, material, or compliance standard.
    """
    pipeline = get_pipeline()
    
    if not any([request.application, request.material, request.standard]):
        raise HTTPException(
            status_code=400, 
            detail="At least one filter (application, material, or standard) is required"
        )
    
    products = pipeline.find_products(
        application=request.application,
        material=request.material,
        standard=request.standard
    )
    
    return {"products": products, "count": len(products)}


@router.post("/search/entities")
async def search_entities(request: EntitySearchRequest):
    """
    Search entities in the knowledge graph.
    """
    pipeline = get_pipeline()
    
    # Convert string entity types to EntityType enum
    entity_types = None
    if request.entity_types:
        from src.retrieval.graph_rag.entity_extractor import EntityType
        entity_types = []
        for et in request.entity_types:
            try:
                entity_types.append(EntityType(et))
            except ValueError:
                pass
    
    results = pipeline.kg.search_entities(
        query=request.query,
        entity_types=entity_types,
        limit=request.limit
    )
    
    return {"entities": results, "count": len(results)}


@router.post("/extract")
async def extract_entities(request: IndexDocumentRequest):
    """
    Extract entities from text without indexing.
    Useful for previewing what will be extracted.
    """
    extractor = get_extractor()
    
    entities = extractor.extract_entities(request.content)
    relationships = extractor.extract_relationships(entities, request.content)
    properties = extractor.extract_properties(request.content)
    
    return {
        "entities": [
            {
                "id": e.id,
                "name": e.name,
                "type": e.entity_type.value,
                "confidence": e.confidence
            }
            for e in entities
        ],
        "relationships": [
            {
                "source": r.source_id,
                "target": r.target_id,
                "type": r.relation_type.value,
                "confidence": r.confidence
            }
            for r in relationships
        ],
        "properties": properties
    }


@router.get("/stats")
async def get_graph_stats():
    """
    Get knowledge graph statistics.
    """
    pipeline = get_pipeline()
    return pipeline.get_statistics()


@router.post("/save")
async def save_graph():
    """
    Save knowledge graph to disk.
    """
    pipeline = get_pipeline()
    pipeline.save()
    return {"status": "saved", "path": pipeline.graph_path}

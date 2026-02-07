"""
Multi-Modal RAG API Endpoints
Provides image indexing and multi-modal search capabilities.
"""

import os
import tempfile
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.retrieval.multimodal import MultiModalRetriever, ImageProcessor

router = APIRouter(prefix="/multimodal", tags=["Multi-Modal RAG"])

# Initialize components
_retriever: Optional[MultiModalRetriever] = None
_processor: Optional[ImageProcessor] = None


def get_retriever() -> MultiModalRetriever:
    """Get or create multi-modal retriever."""
    global _retriever
    if _retriever is None:
        _retriever = MultiModalRetriever(image_dir="data/images")
    return _retriever


def get_processor() -> ImageProcessor:
    """Get or create image processor."""
    global _processor
    if _processor is None:
        _processor = ImageProcessor()
    return _processor


# Request/Response Models
class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.0


class ImageSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int


class IndexDirectoryRequest(BaseModel):
    directory: str


# Endpoints
@router.post("/upload")
async def upload_and_index_image(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    product_code: Optional[str] = Form(None)
):
    """
    Upload and index an image.
    Automatically extracts metadata and creates embeddings.
    """
    retriever = get_retriever()
    
    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save to temp file and index
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Build metadata
        metadata = {}
        if product_code:
            metadata["product_code"] = product_code
        if file.filename:
            metadata["original_filename"] = file.filename
        
        # Index the image
        image_id = retriever.index_image(
            image_path=tmp_path,
            text_description=description,
            metadata=metadata
        )
        
        return {
            "status": "indexed",
            "image_id": image_id,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass


@router.post("/search/text", response_model=ImageSearchResponse)
async def search_by_text(request: TextSearchRequest):
    """
    Search images by text query.
    Uses CLIP embeddings for semantic matching.
    """
    retriever = get_retriever()
    
    results = retriever.search_by_text(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score
    )
    
    return ImageSearchResponse(
        results=[r.to_dict() for r in results],
        count=len(results)
    )


@router.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    """
    Search for similar images using an uploaded image.
    """
    retriever = get_retriever()
    
    # Validate and save temp file
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        results = retriever.search_by_image(
            query_image_path=tmp_path,
            top_k=top_k
        )
        
        return {
            "results": [r.to_dict() for r in results],
            "count": len(results)
        }
        
    finally:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass


@router.get("/product/{product_code}/images")
async def get_product_images(product_code: str, limit: int = 10):
    """
    Find images related to a specific product code.
    """
    retriever = get_retriever()
    
    results = retriever.find_product_images(
        product_code=product_code,
        include_related=True
    )
    
    return {
        "product_code": product_code,
        "images": [r.to_dict() for r in results[:limit]],
        "count": min(len(results), limit)
    }


@router.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Process an image and return metadata without indexing.
    Useful for previewing what will be extracted.
    """
    processor = get_processor()
    
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    
    try:
        content = await file.read()
        processed = processor.process_image_bytes(content, file.filename or "unknown")
        
        return {
            "image_id": processed.image_id,
            "width": processed.width,
            "height": processed.height,
            "format": processed.format,
            "file_size": processed.file_size,
            "detected_objects": processed.detected_objects,
            "metadata": processed.metadata,
            "thumbnail": processed.base64_thumbnail
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index-directory")
async def index_directory(request: IndexDirectoryRequest):
    """
    Index all images in a directory.
    """
    retriever = get_retriever()
    
    if not os.path.isdir(request.directory):
        raise HTTPException(status_code=404, detail="Directory not found")
    
    try:
        stats = retriever.index_directory(request.directory)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """
    Get multi-modal retriever statistics.
    """
    retriever = get_retriever()
    return retriever.get_statistics()


@router.get("/image/{image_id}")
async def get_image_info(image_id: str):
    """
    Get information about an indexed image.
    """
    retriever = get_retriever()
    doc = retriever.get_image_document(image_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return {
        "image_id": doc.image_id,
        "path": doc.image_path,
        "description": doc.text_description,
        "metadata": doc.metadata,
        "processed": {
            "width": doc.processed_image.width if doc.processed_image else None,
            "height": doc.processed_image.height if doc.processed_image else None,
            "detected_objects": doc.processed_image.detected_objects if doc.processed_image else []
        }
    }

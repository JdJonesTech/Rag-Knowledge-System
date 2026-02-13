"""
JD Jones RAG System - FastAPI Application
Main entry point for the API server.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import uuid

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.config.settings import settings
from src.api.routers import internal_chat, external_portal, admin, agentic
from src.api.schemas.responses import (
    EnquiryDemoDashboardResponse, QuotationDemoDashboardResponse,
    QuotationListResponse, PDFGenerationResponse, MarkSentResponse,
    SavePricesResponse
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with proper resource initialization."""
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # === Initialize Production Optimizations ===
    try:
        from src.optimizations.startup import initialize_optimizations
        await initialize_optimizations()
        logger.info("Production optimizations initialized")
    except ImportError:
        logger.info("Optimizations module not available, skipping")
    except Exception as e:
        logger.warning(f"Failed to initialize optimizations: {e}")
    
    # === Initialize Database Connection Pool ===
    try:
        import asyncpg
        db_url = settings.database_url.replace("+asyncpg", "").replace("postgresql", "postgres")
        app.state.db_pool = await asyncpg.create_pool(
            db_url,
            min_size=5,
            max_size=20,
            command_timeout=60,
            max_inactive_connection_lifetime=300
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        app.state.db_pool = None
        logger.error(f"Database connection pool failed: {e}")
    
    # === Initialize Redis Connection ===
    try:
        import redis.asyncio as redis_async
        app.state.redis = redis_async.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        await app.state.redis.ping()
        logger.info("Redis connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        app.state.redis = None
        logger.error(f"Redis connection failed: {e}")
    
    # === Initialize Super Memory Manager ===
    try:
        from src.super_memory.memory_manager import SuperMemoryManager
        app.state.memory_manager = SuperMemoryManager()
        await app.state.memory_manager.initialize()
        logger.info("Super Memory Manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Memory Manager: {e}")
        app.state.memory_manager = None
        logger.error(f"Super Memory Manager failed: {e}")
    
    # === Initialize ChromaDB Client ===
    try:
        import chromadb
        app.state.chroma_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port
        )
        app.state.chroma_client.heartbeat()
        logger.info("ChromaDB connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        app.state.chroma_client = None
        logger.error(f"ChromaDB connection failed: {e}")
    
    yield
    
    # Shutdown - cleanup resources
    logger.info("Shutting down application...")
    
    # === Shutdown Optimizations ===
    try:
        from src.optimizations.startup import shutdown_optimizations
        await shutdown_optimizations()
        logger.info("Optimizations shutdown complete")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error shutting down optimizations: {e}")
    
    if hasattr(app.state, 'db_pool') and app.state.db_pool:
        await app.state.db_pool.close()
        logger.info("Database pool closed")
    
    if hasattr(app.state, 'redis') and app.state.redis:
        await app.state.redis.close()
        logger.info("Redis connection closed")
    
    if hasattr(app.state, 'memory_manager') and app.state.memory_manager:
        await app.state.memory_manager.close()
        logger.info("Memory Manager closed")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    JD Jones RAG System API
    
    A production-ready Retrieval-Augmented Generation system with:
    - Hierarchical knowledge bases
    - Role-based access control
    - Super Memory integration
    - Internal employee chatbot
    - External customer decision tree
    
    **Agentic AI Capabilities:**
    - ReAct agents for iterative reasoning
    - Multi-agent coordination
    - Guided product selection
    - Enquiry classification and routing
    - Human-in-the-loop approvals
    - Observability and monitoring
    """,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    if settings.debug:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error"}
    )


# Include routers
app.include_router(
    internal_chat.router,
    prefix="/internal",
    tags=["Internal Chat"]
)

app.include_router(
    external_portal.router,
    prefix="/external",
    tags=["External Portal"]
)

app.include_router(
    admin.router,
    prefix="/admin",
    tags=["Administration"]
)

# Agentic AI router
app.include_router(
    agentic.router,
    tags=["Agentic AI"]
)

# GraphRAG router
try:
    from src.api.graph_router import router as graph_router
    app.include_router(graph_router)
except ImportError:
    pass  # GraphRAG not available

# Multi-Modal router
try:
    from src.api.multimodal_router import router as multimodal_router
    app.include_router(multimodal_router)
except ImportError:
    pass  # Multi-Modal not available

# Document Generation router
try:
    from src.api.routers.documents import router as documents_router
    app.include_router(documents_router)
except ImportError:
    pass  # Documents module not available

# Quotation Management router (Internal only)
try:
    from src.api.routers.quotations import router as quotations_router
    # Internal management endpoints (with auth)
    app.include_router(
        quotations_router,
        prefix="/internal",
        tags=["Quotation Management"]
    )
    # External submission endpoints (no auth) - for customer portal
    app.include_router(
        quotations_router,
        prefix="/v1/quotations",
        tags=["Customer Quotations"]
    )
except ImportError:
    pass  # Quotations module not available


# Enquiry Management router
try:
    from src.api.routers.enquiries import router as enquiries_router
    # External submit endpoint (no auth)
    app.include_router(
        enquiries_router,
        prefix="/external/enquiry",
        tags=["Customer Enquiries"]
    )
    # Internal management endpoints (with auth)
    app.include_router(
        enquiries_router,
        prefix="/internal/enquiries",
        tags=["Enquiry Management"]
    )
except ImportError:
    pass  # Enquiries module not available


# Prometheus metrics endpoint
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    
    @app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
except ImportError:
    pass  # prometheus_client not available


# Health check endpoints

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }


# Demo dashboard endpoints (no auth required for testing)
@app.get(
    "/internal/enquiries/dashboard",
    tags=["Demo Dashboards"],
    response_model=EnquiryDemoDashboardResponse
)
async def demo_enquiries_dashboard() -> EnquiryDemoDashboardResponse:
    """Demo enquiries dashboard - returns sample data without authentication."""
    from datetime import datetime
    return {
        "enquiries": [
            {
                "id": "ENQ-20260205-001",
                "customer": {"name": "John Smith", "company": "ABC Industries", "email": "john@abc.com"},
                "subject": "Valve packing for high temperature application",
                "status": "under_review",
                "created_at": datetime.now().isoformat(),
                "ai_quick_view": {
                    "one_liner": "Needs valve packing for 350°C steam application",
                    "type": "product_selection",
                    "priority": "high",
                    "urgency": 4,
                    "main_ask": "Product recommendation for high-temp steam valve",
                    "products": ["NA 701", "NA 750"],
                    "actions_needed": {"technical_review": True, "pricing": True}
                }
            },
            {
                "id": "ENQ-20260205-002",
                "customer": {"name": "Sarah Lee", "company": "PetroChem Ltd", "email": "sarah@petrochem.com"},
                "subject": "Quote for pump seal sets",
                "status": "ai_ready",
                "created_at": datetime.now().isoformat(),
                "ai_quick_view": {
                    "one_liner": "Requesting quote for 50 pump seal sets",
                    "type": "quotation",
                    "priority": "medium",
                    "urgency": 3,
                    "main_ask": "Pricing for bulk pump seals",
                    "products": ["NA 715"],
                    "actions_needed": {"pricing": True}
                }
            }
        ],
        "stats": {
            "new": 0,
            "under_review": 1,
            "ai_ready": 1,
            "high_priority": 1,
            "total": 2
        },
        "retrieved_at": datetime.now().isoformat()
    }


@app.get(
    "/internal/quotations/dashboard",
    tags=["Demo Dashboards"],
    response_model=QuotationDemoDashboardResponse
)
async def demo_quotations_dashboard() -> QuotationDemoDashboardResponse:
    """Demo quotations dashboard - returns sample data without authentication."""
    from datetime import datetime
    return {
        "quotations": [
            {
                "id": "QUO-2026-0001",
                "customer": {"name": "John Smith", "company": "ABC Industries"},
                "status": "pending_pricing",
                "priority": "high",
                "created_at": datetime.now().isoformat(),
                "line_items": [
                    {"product_code": "NA 701", "quantity": 50, "size": "6mm x 6mm"},
                    {"product_code": "NA 715", "quantity": 25, "size": "8mm x 8mm"}
                ],
                "ai_analysis": {
                    "one_liner": "Standard valve packing order for refinery customer",
                    "estimated_value": "₹50,000 - ₹75,000",
                    "lead_time": "2-3 weeks"
                }
            },
            {
                "id": "QUO-2026-0002",
                "customer": {"name": "Sarah Lee", "company": "PetroChem Ltd"},
                "status": "ready_to_send",
                "priority": "medium",
                "created_at": datetime.now().isoformat(),
                "line_items": [
                    {"product_code": "NA 750", "quantity": 100, "size": "10mm x 10mm"}
                ],
                "ai_analysis": {
                    "one_liner": "Fugitive emission sealing for petrochemical plant",
                    "estimated_value": "₹45,000",
                    "lead_time": "1 week"
                },
                "pdf_ready": True
            }
        ],
        "stats": {
            "pending_pricing": 1,
            "ready_to_send": 1,
            "sent": 0,
            "total_pipeline_value": 50000
        },
        "retrieved_at": datetime.now().isoformat()
    }


# Main quotations list endpoint - frontend calls this (unauthenticated demo)
@app.get(
    "/demo/quotations",
    tags=["Quotation Management"],
    response_model=QuotationListResponse
)
async def get_quotations_list() -> QuotationListResponse:
    """Get all quotation requests - returns real data from storage."""
    from datetime import datetime
    from src.quotation.models import get_all_quotation_requests
    
    quotation_requests = get_all_quotation_requests()
    
    quotations = []
    by_status = {}
    
    for req in quotation_requests:
        # Convert line items with full structure
        line_items = []
        for item in req.line_items:
            # Build dimensions table if available
            item_dim_unit = getattr(item, 'dimension_unit', 'mm') or 'mm'
            dimensions = getattr(item, 'dimensions', None)
            if not dimensions and (item.size_od or item.size_id or item.size_th):
                dimensions = {
                    "od": f"{item.size_od}{item_dim_unit}" if item.size_od else None,
                    "id": f"{item.size_id}{item_dim_unit}" if item.size_id else None,
                    "th": f"{item.size_th}{item_dim_unit}" if item.size_th else None
                }
            
            line_items.append({
                "id": getattr(item, 'id', None) or str(uuid.uuid4())[:8],
                "product_code": item.product_code,
                "product_name": item.product_name or item.product_code,
                "size": getattr(item, 'size', None) or (f"{item.size_od}×{item.size_id}×{item.size_th}" if item.size_od else None),
                "size_od": getattr(item, 'size_od', None),
                "size_id": getattr(item, 'size_id', None),
                "size_th": getattr(item, 'size_th', None),
                "dimension_unit": item_dim_unit,
                "style": getattr(item, 'style', None),
                "dimensions": dimensions,
                "material_grade": getattr(item, 'material_grade', None),
                "material_code": getattr(item, 'material_code', None),
                "colour": getattr(item, 'colour', None),
                "quantity": item.quantity,
                "unit": getattr(item, 'unit', 'Nos.'),
                "rings_per_set": getattr(item, 'rings_per_set', None),
                "specific_requirements": getattr(item, 'specific_requirements', None),
                "notes": getattr(item, 'notes', None),
                "is_ai_suggested": getattr(item, 'is_ai_suggested', False),
                "ai_confidence": getattr(item, 'ai_confidence', None),
                "unit_price": getattr(item, 'unit_price', None),
                "total_price": getattr(item, 'amount', None) or (item.unit_price * item.quantity if item.unit_price else None)
            })
        
        # Track status counts
        status = req.status.value if hasattr(req.status, 'value') else str(req.status)
        by_status[status] = by_status.get(status, 0) + 1
        
        # Build AI analysis from stored analysis if available
        ai_analysis_data = None
        specs_recommendations = None
        ai_suggested_line_items = None
        if hasattr(req, 'ai_analysis') and req.ai_analysis:
            try:
                ai_analysis_data = req.ai_analysis.get_quick_view() if hasattr(req.ai_analysis, 'get_quick_view') else req.ai_analysis.to_dict()
                # Extract specifications recommendations and AI-suggested line items from sub_agent_results
                if hasattr(req.ai_analysis, 'sub_agent_results') and req.ai_analysis.sub_agent_results:
                    specs_recommendations = req.ai_analysis.sub_agent_results.get('specifications_recommendations')
                    ai_suggested_line_items = req.ai_analysis.sub_agent_results.get('ai_suggested_line_items')
            except:
                pass
        
        # Get original message if available
        original_msg = getattr(req, 'original_message', None) or ""
        
        if not ai_analysis_data:
            # Build AI analysis from available data
            one_liner = getattr(req, 'ai_summary', None)
            if not one_liner and original_msg:
                # Generate a summary from the original message
                one_liner = original_msg[:100] + ("..." if len(original_msg) > 100 else "")
            elif not one_liner:
                one_liner = f"Quote request from {req.customer.name}" if req.customer else "Quote request"
            
            ai_analysis_data = {
                "one_liner": one_liner,
                "requirements_summary": getattr(req, 'special_requirements', '') or original_msg[:200] if original_msg else "",
                "recommended_products": [item.product_code for item in req.line_items[:3]],
                "value_estimate": f"₹{getattr(req, 'total_value', 0):,}" if getattr(req, 'total_value', 0) else "TBD"
            }
        
        # Add specifications recommendations to the ai_analysis
        if specs_recommendations:
            ai_analysis_data["specifications_recommendations"] = specs_recommendations
        
        quotations.append({
            "id": req.id,
            "quotation_number": getattr(req, 'quotation_number', req.id),
            "customer": {
                "name": req.customer.name if req.customer else "Unknown",
                "company": req.customer.company if req.customer else "Unknown",
                "email": req.customer.email if req.customer else ""
            },
            "status": status,
            "priority": getattr(req, 'priority', 'medium'),
            "created_at": req.created_at.isoformat() if req.created_at else datetime.now().isoformat(),
            # Use AI-suggested line items for generic quotations (when no customer line items exist)
            "line_items": line_items if line_items else ai_suggested_line_items or [],
            "is_generic": bool(original_msg),  # True if customer submitted free-text message (generic request)
            "ai_analysis": ai_analysis_data,
            "pdf_generated": getattr(req, 'pdf_generated', False),
            "assigned_to": getattr(req, 'assigned_to', None),
            "original_message": original_msg,
            "notes": getattr(req, 'special_requirements', '') or "",
            "requires_ai_processing": getattr(req, 'requires_ai_processing', False)
        })

    
    return {
        "quotations": quotations,
        "stats": {
            "total": len(quotations),
            "by_status": by_status,
            "total_value_pending": sum(getattr(q, 'total_value', 0) or 0 for q in quotation_requests)
        }
    }


# Demo PDF generation endpoint (following datasheet generation pattern)
@app.post(
    "/demo/quotations/{quotation_id}/generate-pdf",
    tags=["Quotation Management"],
    response_model=PDFGenerationResponse
)
async def demo_generate_quotation_pdf(quotation_id: str, include_pricing: bool = True) -> PDFGenerationResponse:
    """
    Generate a quotation PDF - uses documents/pdf_generator like datasheet generation.
    Returns a download URL for the generated PDF.
    """
    import logging
    _logger = logging.getLogger(__name__)
    
    from src.quotation.models import get_quotation_request
    from src.documents.pdf_generator import get_pdf_generator
    from src.api.routers.quotations import _quotation_analyses
    
    try:
        quotation = get_quotation_request(quotation_id)
        
        if not quotation:
            raise HTTPException(
                status_code=404,
                detail=f"Quotation '{quotation_id}' not found"
            )
        
        # Get AI-suggested line items if the quotation has no line items
        analysis = _quotation_analyses.get(quotation_id)
        line_items_for_pdf = []
        
        if quotation.line_items:
            # Use existing line items from the quotation
            for item in quotation.line_items:
                line_items_for_pdf.append({
                    "code": item.product_code,
                    "name": item.product_name or item.product_code,
                    "size": item.size or "-",
                    "size_od": getattr(item, 'size_od', None),
                    "size_id": getattr(item, 'size_id', None),
                    "size_th": getattr(item, 'size_th', None),
                    "dimension_unit": getattr(item, 'dimension_unit', 'mm') or 'mm',
                    "style": item.style or "-",
                    "material": item.material_grade or "-",
                    "material_grade": item.material_grade or "-",
                    "material_code": getattr(item, 'material_code', '') or "",
                    "quantity": item.quantity or 1,
                    "unit": getattr(item, 'unit', 'Nos.') or "Nos.",
                    "rings_per_set": getattr(item, 'rings_per_set', None),
                    "unit_price": item.unit_price or 500  # Default price
                })
        elif analysis and analysis.sub_agent_results:
            # Use AI-suggested line items
            ai_line_items = analysis.sub_agent_results.get('ai_suggested_line_items', [])
            for item_dict in ai_line_items:
                line_items_for_pdf.append({
                    "code": item_dict.get('product_code', 'Unknown'),
                    "name": item_dict.get('product_name', 'Unknown Product'),
                    "size": item_dict.get('size', '-'),
                    "size_od": item_dict.get('size_od'),
                    "size_id": item_dict.get('size_id'),
                    "size_th": item_dict.get('size_th'),
                    "dimension_unit": item_dict.get('dimension_unit', 'mm'),
                    "style": item_dict.get('style', '-'),
                    "material": item_dict.get('material_grade', '-'),
                    "material_grade": item_dict.get('material_grade', '-'),
                    "material_code": item_dict.get('material_code', ''),
                    "quantity": item_dict.get('quantity', 1),
                    "unit": item_dict.get('unit', 'Nos.'),
                    "rings_per_set": item_dict.get('rings_per_set'),
                    "unit_price": item_dict.get('unit_price') or 500
                })
        
        if not line_items_for_pdf:
            # Fallback - create a placeholder item
            line_items_for_pdf.append({
                "code": "NA XXX",
                "name": "Product to be specified",
                "quantity": 1,
                "unit_price": 0
            })
        
        # Get customer info
        customer_name = "Customer"
        customer_email = "customer@example.com"
        customer_company = ""
        customer_designation = ""
        customer_address = ""
        if quotation.customer:
            customer_name = quotation.customer.name or "Customer"
            customer_email = quotation.customer.email or "customer@example.com"
            customer_company = getattr(quotation.customer, 'company', '') or ""
            customer_designation = getattr(quotation.customer, 'designation', '') or ""
            customer_address = getattr(quotation.customer, 'address', '') or ""
        
        # Use documents/pdf_generator (same as datasheet)
        generator = get_pdf_generator()
        
        # Convert line items to the format expected by generate_quotation
        # Pass full details for the reference-matching PDF layout
        products = []
        for item in line_items_for_pdf:
            # Build size dict from individual OD/ID/TH if available
            size_val = item.get("size", "-")
            if item.get("size_od") or item.get("size_id") or item.get("size_th"):
                size_val = {
                    "od": str(item.get("size_od", "-") or "-"),
                    "id": str(item.get("size_id", "-") or "-"),
                    "th": str(item.get("size_th", "-") or "-"),
                }
            
            # For material column: prefer explicit material_code, fall back to material_grade
            mat_code = item.get("material_code", "")
            mat_grade = item.get("material_grade", item.get("material", "-"))
            # Show both grade and code if both present
            material_display = mat_code if mat_code else mat_grade
            
            products.append({
                "code": item.get("code", ""),
                "name": item.get("name", ""),
                "size": size_val,
                "dimension_unit": item.get("dimension_unit", "mm"),
                "material": material_display,
                "material_code": mat_code,
                "material_grade": mat_grade,
                "quantity": item.get("quantity", 1),
                "unit": item.get("unit", "Nos."),
                "unit_price": item.get("unit_price", 500) if include_pricing else 0,
                "rings_per_set": item.get("rings_per_set") or "-",
            })
        
        doc = generator.generate_quotation(
            customer_name=customer_name,
            customer_email=customer_email,
            products=products,
            notes=getattr(quotation, 'original_message', '') or "",
            validity_days=30,
            terms="Standard terms and conditions apply"
        )
        
        _logger.info(f"PDF generated for quotation {quotation_id}: {doc.filename}")
        
        # Return JSON with download URL (like datasheet endpoint)
        return {
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type.value,
            "title": doc.title,
            "filename": doc.filename,
            "download_url": f"/documents/download/{doc.doc_id}",
            "format": doc.format,
            "created_at": doc.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error generating quotation PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Demo mark-sent endpoint (unauthenticated for internal portal)
@app.post(
    "/demo/quotations/{quotation_id}/mark-sent",
    tags=["Quotation Management"],
    response_model=MarkSentResponse
)
async def demo_mark_sent(quotation_id: str) -> MarkSentResponse:
    """Mark a quotation as sent - demo endpoint without auth."""
    from src.quotation.models import get_quotation_request, save_quotation_request, QuotationStatus
    
    quotation = get_quotation_request(quotation_id)
    if not quotation:
        raise HTTPException(status_code=404, detail=f"Quotation '{quotation_id}' not found")
    
    quotation.status = QuotationStatus.SENT
    save_quotation_request(quotation)
    
    return {
        "success": True,
        "quotation_id": quotation_id,
        "status": quotation.status.value,
        "message": "Quotation marked as sent"
    }


# Demo save-prices endpoint (unauthenticated for internal portal)
@app.post(
    "/demo/quotations/{quotation_id}/save-prices",
    tags=["Quotation Management"],
    response_model=SavePricesResponse
)
async def demo_save_prices(quotation_id: str, request: Request) -> SavePricesResponse:
    """Save line item details (prices, specs, etc.) - demo endpoint without auth."""
    from src.quotation.models import get_quotation_request, save_quotation_request
    
    quotation = get_quotation_request(quotation_id)
    if not quotation:
        raise HTTPException(status_code=404, detail=f"Quotation '{quotation_id}' not found")
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    # body.prices is a dict of {line_item_index: unit_price}
    prices = body.get("prices", {})
    # body.line_item_updates is a dict of {line_item_index: {field: value, ...}}
    line_item_updates = body.get("line_item_updates", {})
    
    updated_count = 0
    total = 0.0
    
    # Apply price updates
    for idx_str, price in prices.items():
        idx = int(idx_str)
        if 0 <= idx < len(quotation.line_items):
            quotation.line_items[idx].unit_price = float(price)
            quotation.line_items[idx].amount = float(price) * quotation.line_items[idx].quantity
            total += quotation.line_items[idx].amount
            updated_count += 1
    
    # Apply line item field updates (all editable fields)
    for idx_str, updates in line_item_updates.items():
        idx = int(idx_str)
        if 0 <= idx < len(quotation.line_items):
            item = quotation.line_items[idx]
            if 'product_code' in updates and updates['product_code'] is not None:
                item.product_code = str(updates['product_code'])
            if 'material_code' in updates and updates['material_code'] is not None:
                item.material_code = str(updates['material_code'])
            if 'material_grade' in updates and updates['material_grade'] is not None:
                item.material_grade = str(updates['material_grade'])
            if 'size' in updates and updates['size'] is not None:
                item.size = str(updates['size'])
            if 'size_od' in updates and updates['size_od'] is not None:
                item.size_od = float(updates['size_od']) if updates['size_od'] != '' else None
            if 'size_id' in updates and updates['size_id'] is not None:
                item.size_id = float(updates['size_id']) if updates['size_id'] != '' else None
            if 'size_th' in updates and updates['size_th'] is not None:
                item.size_th = float(updates['size_th']) if updates['size_th'] != '' else None
            if 'dimension_unit' in updates and updates['dimension_unit'] is not None:
                item.dimension_unit = str(updates['dimension_unit'])
            if 'style' in updates and updates['style'] is not None:
                item.style = str(updates['style'])
            if 'colour' in updates and updates['colour'] is not None:
                item.colour = str(updates['colour'])
            if 'rings_per_set' in updates:
                item.rings_per_set = int(updates['rings_per_set']) if updates['rings_per_set'] not in (None, '', '-') else None
            if 'unit' in updates and updates['unit'] is not None:
                item.unit = str(updates['unit'])
            if 'quantity' in updates and updates['quantity'] is not None:
                item.quantity = int(updates['quantity'])
                # Recalculate amount if unit_price is set
                if item.unit_price:
                    item.amount = item.unit_price * item.quantity
            updated_count += 1
    
    # Recalculate totals
    total = sum((item.amount or 0) for item in quotation.line_items)
    quotation.total_amount = total
    quotation.gst_amount = total * 0.18
    quotation.grand_total = total + quotation.gst_amount
    
    save_quotation_request(quotation)
    
    return {
        "success": True,
        "quotation_id": quotation_id,
        "updated_items": updated_count,
        "total_amount": quotation.total_amount,
        "grand_total": quotation.grand_total,
        "message": f"Updated {updated_count} line items"
    }


# Demo enquiries endpoint for internal portal (unauthenticated)
@app.get("/demo/enquiries", tags=["Quotation Management"])
async def get_enquiries_list():
    """Get all enquiries - returns real data from storage for internal portal dashboard."""
    from datetime import datetime
    from src.enquiry.models import get_all_enquiries, get_enquiry_stats, EnquiryStatus
    
    all_enquiries = get_all_enquiries()
    stats = get_enquiry_stats()
    
    enquiries = []
    for enq in all_enquiries:
        # Build AI quick view from AI analysis if available
        ai_quick_view = None
        if enq.ai_analysis:
            ai_quick_view = enq.ai_analysis.get_quick_view()
        else:
            # Fallback quick view
            ai_quick_view = {
                "one_liner": enq.subject or "Customer enquiry",
                "type": enq.enquiry_type.value,
                "priority": enq.priority.value,
                "urgency": 3,
                "main_ask": enq.subject or "General enquiry",
                "products": [],
                "actions_needed": {"technical_review": False, "pricing": False}
            }
        
        # Include the latest suggested response if available
        latest_response = None
        if enq.suggested_responses:
            latest = enq.suggested_responses[-1]
            latest_response = latest.to_dict() if hasattr(latest, 'to_dict') else latest
        
        enquiries.append({
            "id": enq.id,
            "customer": enq.customer.to_dict() if enq.customer else {"name": "Unknown", "company": "", "email": ""},
            "subject": enq.subject,
            "raw_message": enq.raw_message,
            "status": enq.status.value,
            "created_at": enq.created_at.isoformat(),
            "assigned_to": enq.assigned_to,
            "ai_quick_view": ai_quick_view,
            "ai_analysis": enq.ai_analysis.to_dict() if enq.ai_analysis else None,
            "latest_suggested_response": latest_response
        })
    
    # Calculate high priority count
    high_priority_count = len([e for e in all_enquiries if e.priority.value == "high" or e.priority.value == "urgent"])
    
    return {
        "enquiries": enquiries,
        "stats": {
            "new": stats.get("by_status", {}).get("new", 0),
            "under_review": stats.get("by_status", {}).get("under_review", 0),
            "ai_ready": stats.get("by_status", {}).get("ai_response_generated", 0),
            "high_priority": high_priority_count,
            "total": stats.get("total", 0),
            "by_status": stats.get("by_status", {}),
            "by_priority": stats.get("by_priority", {}),
            "by_type": stats.get("by_type", {})
        },
        "retrieved_at": datetime.now().isoformat()
    }


# Demo endpoint for re-analyzing an enquiry (bypasses auth for testing)
@app.post("/demo/enquiries/{enquiry_id}/re-analyze", tags=["Demo"])
async def demo_reanalyze_enquiry(enquiry_id: str):
    """Re-run AI analysis on an enquiry - demo endpoint without auth."""
    from src.enquiry.models import get_enquiry, save_enquiry
    from src.enquiry.analyzer import get_enquiry_analyzer
    
    enquiry = get_enquiry(enquiry_id)
    if not enquiry:
        raise HTTPException(status_code=404, detail=f"Enquiry '{enquiry_id}' not found")
    
    try:
        # Run fresh analysis
        analyzer = get_enquiry_analyzer()
        analysis = await analyzer.analyze_enquiry(enquiry)
        
        # Update enquiry
        enquiry.ai_analysis = analysis
        enquiry.enquiry_type = analysis.detected_type
        enquiry.priority = analysis.detected_priority
        save_enquiry(enquiry)
        
        return {
            "success": True,
            "enquiry_id": enquiry_id,
            "analysis": analysis.to_dict(),
            "quick_view": analysis.get_quick_view(),
            "message": "AI analysis refreshed successfully"
        }
    except Exception as e:
        import logging
        logging.error(f"Re-analyze failed: {e}")
        return {"success": False, "error": str(e)}


# Demo endpoint for generating AI response (bypasses auth for testing)
@app.post("/demo/enquiries/{enquiry_id}/generate-response", tags=["Demo"])
async def demo_generate_response(enquiry_id: str, request: Request):
    """Generate AI response for an enquiry - demo endpoint without auth."""
    from src.enquiry.models import get_enquiry, save_enquiry, EnquiryStatus
    from src.enquiry.analyzer import get_enquiry_analyzer
    
    enquiry = get_enquiry(enquiry_id)
    if not enquiry:
        raise HTTPException(status_code=404, detail=f"Enquiry '{enquiry_id}' not found")
    
    try:
        body = await request.json()
    except:
        body = {}
    
    tone = body.get("tone", "professional")
    include_products = body.get("include_products", True)
    
    try:
        analyzer = get_enquiry_analyzer()
        
        # Ensure AI analysis exists
        if not enquiry.ai_analysis:
            enquiry.ai_analysis = await analyzer.analyze_enquiry(enquiry)
        
        # Generate response
        suggested = await analyzer.generate_suggested_response(
            enquiry=enquiry,
            tone=tone,
            include_products=include_products
        )
        
        # Add to enquiry
        enquiry.suggested_responses.append(suggested)
        # Keep status as under_review - only mark as sent explicitly
        if enquiry.status == EnquiryStatus.NEW:
            enquiry.status = EnquiryStatus.UNDER_REVIEW
        save_enquiry(enquiry)
        
        return {
            "success": True,
            "suggested_response": suggested.to_dict(),
            "message": "AI response generated successfully"
        }
    except Exception as e:
        import logging
        logging.error(f"Generate response failed: {e}")
        return {"success": False, "error": str(e)}


# Demo endpoint for marking enquiry as sent
@app.post("/demo/enquiries/{enquiry_id}/mark-sent", tags=["Demo"])
async def demo_mark_enquiry_sent(enquiry_id: str):
    """Mark an enquiry as sent - demo endpoint without auth."""
    from src.enquiry.models import get_enquiry, save_enquiry, EnquiryStatus
    from datetime import datetime
    
    enquiry = get_enquiry(enquiry_id)
    if not enquiry:
        raise HTTPException(status_code=404, detail=f"Enquiry '{enquiry_id}' not found")
    
    try:
        enquiry.status = EnquiryStatus.RESPONSE_SENT
        enquiry.response_sent_at = datetime.now()
        save_enquiry(enquiry)
        
        return {
            "success": True,
            "enquiry_id": enquiry_id,
            "status": enquiry.status.value,
            "message": "Enquiry marked as sent"
        }
    except Exception as e:
        import logging
        logging.error(f"Mark as sent failed: {e}")
        return {"success": False, "error": str(e)}


# Demo endpoint for listing available products
@app.get("/demo/products", tags=["Demo"])
async def demo_list_products():
    """List all available products from the knowledge base - demo endpoint without auth."""
    try:
        from src.data_ingestion.product_catalog_loader import get_product_catalog
        catalog = get_product_catalog()
        
        products = []
        for code, product in sorted(catalog.products.items()):
            products.append({
                "code": product.code,
                "name": product.name,
                "category": product.category,
                "description": product.description[:120] + "..." if len(product.description) > 120 else product.description
            })
        
        return {
            "success": True,
            "products": products,
            "total": len(products)
        }
    except Exception as e:
        import logging
        logging.error(f"List products failed: {e}")
        return {"success": False, "products": [], "error": str(e)}


# Demo chat endpoint (bypasses auth for testing)
# Initialize conversation memory for demo endpoint
from src.agentic.memory.conversation_memory import ConversationMemory
demo_conversation_memory = ConversationMemory(
    max_messages=50,
    context_window=10,
    ttl_hours=24
)

@app.post("/demo/chat", tags=["Demo"], include_in_schema=False)
async def demo_chat(request: Request) -> Dict[str, Any]:
    """
    Demo chat endpoint that bypasses authentication.
    Includes conversation memory for context tracking.
    For development and testing purposes only.
    """
    from src.agents.internal_agent import InternalAgent
    from src.auth.authentication import User
    import uuid
    
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid JSON body"}
        )
    
    message = body.get("message", "")
    session_id = body.get("session_id") or f"demo_{uuid.uuid4().hex[:12]}"
    
    if not message:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Message is required"}
        )
    
    # Store user message in short-term memory
    demo_conversation_memory.add_message(
        session_id=session_id,
        role="user",
        content=message
    )
    
    # Get conversation history for context
    conversation_context = demo_conversation_memory.get_context(session_id, num_messages=10)
    conversation_history = []
    if conversation_context and conversation_context.messages:
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in conversation_context.messages[:-1]  # Exclude current message
        ]
    
    # Create demo user
    demo_user = User(
        user_id="demo_001",
        email="demo@jdjones.com",
        full_name="Demo User",
        role="employee",
        department="general",
        is_active=True,
        is_internal=True
    )
    
    try:
        agent = InternalAgent()
        
        # Pass conversation history to agent
        result = await agent.chat(
            user=demo_user,
            query=message,
            session_id=session_id,
            conversation_history=conversation_history
        )
        
        response_text = result.get("response", "I could not generate a response.")
        
        # Store assistant response in short-term memory
        demo_conversation_memory.add_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            metadata={
                "sources": result.get("sources", [])
            }
        )
        
        return {
            "response": response_text,
            "sources": result.get("sources", []),
            "session_id": session_id,
            "metadata": result.get("metadata", {}),
            "conversation_turn": len(conversation_context.messages) if conversation_context else 1
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e), "detail": "Failed to process request"}
        )



@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check(request: Request) -> Dict[str, Any]:
    """Detailed health check with service status using shared connections."""
    health_status = {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "services": {}
    }
    
    # Check database connection using shared pool
    try:
        if hasattr(request.app.state, 'db_pool') and request.app.state.db_pool:
            async with request.app.state.db_pool.acquire() as conn:
                await conn.execute("SELECT 1")
            health_status["services"]["database"] = "healthy"
        else:
            health_status["services"]["database"] = "unhealthy: no connection pool"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis connection using shared client
    try:
        if hasattr(request.app.state, 'redis') and request.app.state.redis:
            await request.app.state.redis.ping()
            health_status["services"]["redis"] = "healthy"
        else:
            health_status["services"]["redis"] = "unhealthy: no redis connection"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check ChromaDB connection using shared client
    try:
        if hasattr(request.app.state, 'chroma_client') and request.app.state.chroma_client:
            request.app.state.chroma_client.heartbeat()
            health_status["services"]["chromadb"] = "healthy"
        else:
            health_status["services"]["chromadb"] = "unhealthy: no chromadb client"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["chromadb"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Memory Manager status
    try:
        if hasattr(request.app.state, 'memory_manager') and request.app.state.memory_manager:
            health_status["services"]["memory_manager"] = "healthy"
        else:
            health_status["services"]["memory_manager"] = "not initialized"
    except Exception as e:
        health_status["services"]["memory_manager"] = f"error: {str(e)}"
    
    return health_status


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else "Documentation disabled in production",
        "health": "/health"
    }


# Authentication endpoint
@app.post("/auth/token", tags=["Authentication"])
async def login(email: str, password: str) -> Dict[str, Any]:
    """
    Authenticate user and return JWT token.
    
    Args:
        email: User email
        password: User password
        
    Returns:
        Access token and user info
    """
    from src.auth.authentication import authenticate_user, create_access_token
    from datetime import timedelta
    
    user = authenticate_user(email, password)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Invalid credentials"}
        )
    
    access_token = create_access_token(
        data={
            "sub": user.user_id,
            "email": user.email,
            "role": user.role,
            "department": user.department
        },
        expires_delta=timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user.to_dict()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )

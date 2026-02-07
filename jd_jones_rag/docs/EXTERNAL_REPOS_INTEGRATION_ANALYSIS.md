# External Repositories Integration Analysis

**Analysis Date:** 2026-02-04  
**Analyst:** AI Assistant  
**Purpose:** Assess integration potential of JD Jones external repositories into the RAG system

---

## Executive Summary

All 5 repositories are **highly relevant** to the JD Jones RAG system and can be integrated to create a comprehensive AI-powered business automation platform. The repositories contain:

- **Real product data** with pricing and material costs
- **Production monitoring** with order tracking and delivery analytics
- **Estimation calculations** for customer quotes
- **PO validation** for vendor management
- **Production optimization** tools

### Integration Priority Matrix

| Repository | Integration Priority | Effort | Value | Status |
|------------|---------------------|--------|-------|--------|
| **Estimation-Dashboard** | 🔴 Critical | Medium | Very High | Production-ready |
| **JD-Jones-Production-Dashboard** | 🔴 Critical | Medium | High | Production-ready |
| **Open-orders-Dashboard** | 🟠 High | Low | High | Production-ready |
| **automation-hub** | 🟡 Medium | High | Medium | Multiple tools |
| **Estimation-calculator-backend** | 🟢 Low | Low | Low | Skeleton only |

---

## Repository 1: Estimation-Dashboard

### Overview
A **Node.js/Express + React** estimation system for calculating product costs with detailed pricing logic for JD Jones products.

### Tech Stack
- **Backend:** Node.js, Express, MongoDB
- **Frontend:** React (in `frontend/` folder)
- **Key Libraries:** helmet, cors, body-parser, cookie-parser

### Key Components

#### 1. Product Calculation Engines
Location: `backend/calculationEngine/`
- `707Calculation.js` - Graphite ring calculations
- `710Calculation.js` - Graphite-based product calculations
- `710VAngularCalculation.js` - Angular graphite products
- `703Calculation.js` - Braided + Graphite combinations
- `715Calculation.js` - Advanced graphite products
- `B3_707Calculation.js` - B3 yarn variants

#### 2. Product Data (`products.json`)
Contains **real pricing data** for JD Jones products:
```json
{
  "productCode": "707",
  "productName": "707 Graphite",
  "metadata": {
    "GraphiteMaterials": [
      { "name": "Graphite sheet Local", "costPerKg": 770 },
      { "name": "Graphite foil (Sigraflex - E)", "costPerKg": 1600 },
      { "name": "Graphite foil (Sigraflex - APX 2)", "costPerKg": 4800 }
    ]
  },
  "tables": {
    "tableName": "707_labour_cost",
    "data": [
      { "odMin": 1, "odMax": 39, "ratePerRing": 2 },
      { "odMin": 40, "odMax": 50, "ratePerRing": 2.5 }
    ]
  }
}
```

#### 3. API Endpoints
```
POST /api/estimation/calculate    - Product estimation
GET  /api/auth/*                  - Authentication
GET  /api/customers/*             - Customer management
POST /api/export/*                - Export results
POST /api/import/*                - Import data
GET  /api/ptfe/*                  - PTFE die cost calculations
```

### Integration Opportunities

#### A. Product Selection Agent Enhancement
**Current State:** ProductSelectionAgent uses static product catalog  
**Enhancement:** Connect to Estimation-Dashboard's calculation engines

```python
# Integration approach
class EstimationIntegration:
    def get_live_pricing(self, product_code: str, specs: dict) -> dict:
        """Call Estimation-Dashboard API for real-time pricing"""
        response = requests.post(
            f"{ESTIMATION_API}/api/estimation/calculate",
            json={"productCode": product_code, "input": specs}
        )
        return response.json()
```

#### B. Document Generator Enhancement
**Current State:** DocumentGeneratorTool uses placeholder pricing  
**Enhancement:** Pull actual costs from calculation engines

#### C. Knowledge Base Enrichment
**Action:** Ingest `products.json` into ChromaDB for product recommendations

---

## Repository 2: JD-Jones-Production-Dashboard

### Overview
A **FastAPI + React** production monitoring system with dual backends:
- `openorders/` - Order tracking and analytics
- `estimationcalculator/` - Estimation calculations (duplicate of Estimation-Dashboard)

### Tech Stack
- **Backend:** FastAPI (Python)
- **Database:** Supabase (PostgreSQL)
- **Frontend:** React + TypeScript

### Database Schema
Two main tables in Supabase:

#### `delivery_report`
```sql
- delivery_id, cust_code, order_no, pack_serial
- delivery_date, party_name, sales_order_no
- product_code, description, quantity, unit
- net_weight, rate, amount
- invoice_no, invoice_date, transporter
- dispatch_status, department
```

#### `order_outstanding`
```sql
- order_line_id, cust_code, order_date
- style_no, buyer_name, sales_order_no
- drawing_no, size, order_qty, outstanding_qty
- rate, gross_value, currency
- commitment_date, delivery_date
```

### Key Features
1. **Real-time Order Tracking** - Outstanding orders monitoring
2. **Delivery Analytics** - Sales and dispatch reporting
3. **Email IMAP Integration** - Automatic Excel file import from emails
4. **Financial Year Analytics** - Year-over-year comparisons

### Integration Opportunities

#### A. Enquiry Management Agent Enhancement
**Current State:** EnquiryManagementAgent cannot check order status  
**Enhancement:** Connect to Production Dashboard for order queries

```python
class OrderStatusIntegration:
    async def get_order_status(self, po_number: str) -> dict:
        """Query Production Dashboard for order status"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PRODUCTION_API}/api/orders/{po_number}"
            )
            return response.json()
    
    async def check_delivery_status(self, order_no: str) -> dict:
        """Check if order has been dispatched"""
        # Query delivery_report table
        pass
```

#### B. Customer Portal Enhancement
Add real-time order tracking to external customer portal:
- "Where's my order?" queries answered automatically
- Delivery ETA based on historical dispatch data

#### C. Internal Dashboard Data
Connect internal portal to production analytics:
- Today's open orders
- Orders by customer/product
- Sales performance metrics

---

## Repository 3: Open-orders-Dashboard

### Overview
A **Factory Order Management System** built with FastAPI + React for tracking sales orders and deliveries.

### Tech Stack
- **Backend:** FastAPI, SQLAlchemy, SQLite/Supabase
- **Frontend:** React + TypeScript, Tailwind CSS
- **Auth:** JWT tokens with role-based access

### Features
1. **CSV Data Ingestion** - Upload factory reports
2. **Search & Filter Dashboard** - PO, serial, part number search
3. **Open Orders View** - Pending orders prioritization
4. **Sales Analytics** - Financial year analysis

### Database Tables
- `order_outstanding` - Same schema as Production Dashboard
- `delivery_report` - Same schema as Production Dashboard

### Integration Opportunities

#### A. Duplicate Data Source
**Note:** This appears to share the same Supabase database as JD-Jones-Production-Dashboard. Consider consolidating.

#### B. CRM Tool Enhancement
Connect CRMTool to real order data:
```python
class CRMIntegration:
    def get_customer_orders(self, customer_code: str) -> List[Order]:
        """Fetch customer's active orders"""
        # Query order_outstanding where cust_code = customer_code
        pass
```

---

## Repository 4: automation-hub

### Overview
A **comprehensive automation suite** with multiple Python tools for business process automation.

### Components

#### 1. PO Checker (`jdjones_Po_checker/`)
**Purpose:** Validate Purchase Orders against vendor pricing databases

**Supported Vendors:**
- Emerson
- KSB
- Emerson Chennai
- Ampo Spain
- Ampo India/Arabia

**Features:**
- PDF purchase order parsing
- Price validation against Excel databases
- PDF report generation
- Batch processing

#### 2. Estimation Calculator (`Estimation Calculator/`)
**Purpose:** Desktop GUI for product cost calculations

**Products Supported:**
- NA 707 (Graphite)
- NA 710V (Angular and Rectangle profiles)

**Features:**
- Multi-product support with unique calculation logic
- Excel integration for labor/material costs
- Real-time validation
- Export to Excel

#### 3. Braided Production Planner
**Purpose:** Production efficiency analysis for braiding machines

**Features:**
- Best-case efficiency tracking per machine-style-size
- Standardized metrics (normalized to 1kg)
- Excel/CSV output with statistical analysis

#### 4. WaterJet Optimizer
**Purpose:** Nesting optimization for waterjet cutting

### Integration Opportunities

#### A. PO Checker → EnquiryManagementAgent
Auto-validate incoming POs mentioned in enquiries:
```python
class POValidationTool(BaseTool):
    def validate_po(self, po_pdf_path: str, vendor: str) -> dict:
        """Validate PO pricing against vendor database"""
        # Call PO Checker logic
        pass
```

#### B. Estimation Calculator → DocumentGeneratorTool
Use calculation logic for generating accurate quotations:
```python
class EstimationCalculatorIntegration:
    def calculate_quote(self, product: str, specs: dict) -> dict:
        """Generate quote using estimation calculator logic"""
        # Import calculation engine
        from automation_hub.estimation_calculator import calculate
        return calculate(product, specs)
```

#### C. Production Planner → Knowledge Base
Ingest efficiency data for production planning queries

---

## Repository 5: Estimation-calculator-backend

### Overview
A **FastAPI skeleton** intended to mirror the desktop Estimation Calculator.

### Status: Incomplete
- All modules raise `NotImplementedError`
- Contains TODO comments
- No production logic implemented

### Structure
```
app/
├── main.py              # Application factory (empty)
├── routes/              # Endpoint declarations (stubs)
├── domain/              # Pydantic models (stubs)
├── data/                # Data loaders (stubs)
```

### Integration Recommendation
**Skip this repository.** Use the fully-implemented Estimation-Dashboard instead.

---

## Recommended Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JD Jones RAG System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐   ┌─────────────────┐                 │
│  │ ProductSelection │   │ EnquiryManagement│                │
│  │     Agent        │   │      Agent       │                │
│  └────────┬────────┘   └────────┬─────────┘                │
│           │                      │                          │
│           ▼                      ▼                          │
│  ┌─────────────────────────────────────────┐               │
│  │           Integration Layer              │               │
│  │  ┌─────────────┐  ┌─────────────┐       │               │
│  │  │ Estimation  │  │  Order      │       │               │
│  │  │ API Client  │  │  API Client │       │               │
│  │  └──────┬──────┘  └──────┬──────┘       │               │
│  └─────────┼────────────────┼──────────────┘               │
│            │                │                               │
└────────────┼────────────────┼───────────────────────────────┘
             │                │
             ▼                ▼
┌─────────────────┐  ┌─────────────────────┐
│ Estimation      │  │ Production          │
│ Dashboard       │  │ Dashboard           │
│ (Node.js API)   │  │ (FastAPI)           │
└────────┬────────┘  └────────┬────────────┘
         │                    │
         ▼                    ▼
    ┌─────────┐         ┌─────────┐
    │ MongoDB │         │Supabase │
    └─────────┘         └─────────┘
```

---

## Implementation Plan

### Phase 1: Data Integration (Week 1)

#### 1.1: Ingest products.json into RAG
```bash
# Copy products.json to data directory
cp external_repos/Estimation-Dashboard/products.json data/products/

# Update data ingestion to include product pricing
```

#### 1.2: Create API Clients
Create integration clients for external services:

```python
# src/integrations/estimation_client.py
class EstimationAPIClient:
    base_url = "http://localhost:5000"  # Estimation Dashboard
    
    async def calculate_product(self, product_code: str, input: dict):
        pass

# src/integrations/production_client.py  
class ProductionAPIClient:
    base_url = "http://localhost:8001"  # Production Dashboard
    
    async def get_order_status(self, order_no: str):
        pass
```

### Phase 2: Agent Enhancement (Week 2)

#### 2.1: ProductSelectionAgent + Live Pricing
- Connect to Estimation Dashboard for real-time quotes
- Display accurate pricing in recommendations

#### 2.2: EnquiryManagementAgent + Order Status
- Add order status lookup capability
- Answer "Where's my order?" queries

### Phase 3: Tool Enhancement (Week 3)

#### 3.1: DocumentGeneratorTool + Accurate Pricing
- Pull actual material and labor costs
- Generate accurate quotations

#### 3.2: Add PO Validation Tool
- Integrate PO Checker logic
- Auto-validate incoming purchase orders

---

## Data Synchronization Strategy

### Option 1: API-First Integration
- RAG system calls external APIs in real-time
- Pro: Always up-to-date data
- Con: Dependency on external services

### Option 2: Data Replication
- Periodic sync of external data into RAG knowledge base
- Pro: Fast queries, no external dependencies
- Con: Data may be stale

### Recommended: Hybrid Approach
- **Static data** (products, materials): Sync to ChromaDB daily
- **Dynamic data** (orders, pricing): API calls in real-time

---

## Environment Configuration

Add to `.env`:
```bash
# External Service URLs
ESTIMATION_DASHBOARD_URL=http://localhost:5000
PRODUCTION_DASHBOARD_URL=http://localhost:8001
OPEN_ORDERS_URL=http://localhost:8002

# Supabase (shared by Production & Open Orders)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=xxxxx

# Feature Flags
ENABLE_LIVE_PRICING=true
ENABLE_ORDER_TRACKING=true
ENABLE_PO_VALIDATION=false
```

---

## Conclusion

The external repositories provide valuable real-world data and business logic that can significantly enhance the JD Jones RAG system:

1. **Estimation-Dashboard** → Accurate product pricing for quotes
2. **Production-Dashboard** → Real-time order status for customers
3. **Open-orders-Dashboard** → Order analytics integration
4. **automation-hub** → PO validation and production planning

Priority should be:
1. ✅ Ingest `products.json` into knowledge base
2. ✅ Create API clients for Estimation and Production dashboards
3. ✅ Enhance agents to use live data
4. ✅ Add order tracking capability to enquiry handling

**Estimated Total Effort:** 2-3 weeks for full integration

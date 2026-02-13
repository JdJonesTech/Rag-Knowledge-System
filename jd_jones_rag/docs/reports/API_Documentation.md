# JD Jones RAG System - API Documentation

## Overview
This document provides a detailed reference for the JD Jones RAG System API.
The API is built using FastAPI and follows RESTful principles.

**Base URL**: `http://<host>:<port>`
**Auth Header**: `Authorization: Bearer <token>`

---

## 1. External Portal (Customer Facing)
Endpoints used by the customer-facing portal for decision trees, chat, and form submissions.

### GET /external/decision-tree
**Summary**: Get decision tree structure.
**Description**: Retrieve the complete decision tree structure used for guiding customers.
**Response**: `DecisionTreeResponse`
```json
{
  "root_node": { ... },
  "tree_structure": { ... }
}
```

### GET /external/decision-tree/node/{node_id}
**Summary**: Get specific node.
**Description**: Get a specific node from the decision tree.
**Response**: `DecisionTreeNodeResponse`

### POST /external/navigate
**Summary**: Navigate decision tree.
**Description**: Move to the next node in the decision tree based on user selection.
**Request Body**: `NavigationRequest`
```json
{
  "node_id": "string (current node id)",
  "option_index": 0,
  "collected_data": { "key": "value" },
  "session_id": "string (optional)"
}
```
**Response**: `NavigationResponse`
```json
{
  "session_id": "string",
  "current_node": { ... },
  "collected_data": { ... },
  "history_length": 5
}
```

### POST /external/classify
**Summary**: Classify customer intent.
**Description**: Classify a customer query to determine intent.
**Request Body**: `CustomerQueryRequest`
```json
{
  "query": "string",
  "session_id": "string (optional)"
}
```
**Response**: `ClassificationResponse`

### POST /external/query
**Summary**: Process customer query.
**Description**: Process natural language query from customer, classify intent, and generate response.
**Request Body**: `CustomerQueryRequest`
```json
{
  "query": "I need packing for a high temp valve",
  "session_id": "string (optional)"
}
```
**Response**: `QueryProcessResponse`
```json
{
  "session_id": "string",
  "response": "Based on your requirements...",
  "intent": "product_selection",
  "confidence": 0.95,
  "sources": [ ... ],
  "suggested_actions": ["view_product", "contact_sales"],
  "suggested_node": { ... }
}
```

### POST /external/submit-form
**Summary**: Submit form data.
**Description**: Submit data collected from a decision tree form node.
**Request Body**: `FormSubmissionRequest`
```json
{
  "node_id": "string",
  "form_data": { ... },
  "session_id": "string"
}
```
**Response**: `FormSubmissionResponse`
```json
{
  "success": true,
  "submission_id": "string",
  "action_type": "string",
  "next_steps": "string"
}
```

### POST /external/quote-request
**Summary**: Submit quote request.
**Request Body**: `QuoteRequestForm`
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "+1234567890",
  "company": "Acme Corp",
  "products": "Valve packing details...",
  "quantity": 10,
  "rfq_number": "optional-ref"
}
```
**Response**: `QuoteRequestResponse`
```json
{
  "success": true,
  "quote_reference": "string",
  "status": "pending"
}
```

### POST /external/track-order
**Summary**: Track order status.
**Request Body**: `OrderTrackingRequest`
```json
{
  "order_number": "ORD-12345",
  "email": "john@example.com"
}
```
**Response**: `OrderTrackingResponse`
```json
{
  "success": true,
  "order_number": "ORD-12345",
  "status": "Shipped",
  "tracking_number": "..."
}
```

### GET /external/faq
**Summary**: Get FAQ answers.
**Query Params**: `question` (string)
**Response**: `FAQResponse`

### GET /external/session/{session_id}
**Summary**: Get session info.
**Response**: `SessionResponse`

---

## 2. Internal Chat (Employee Facing)
Endpoints for the internal employee chatbot.

### POST /internal/chat
**Summary**: Send a chat message.
**Request Body**: `ChatRequest`
```json
{
  "message": "What is the pressure rating for NA 701?",
  "session_id": "optional-session-id"
}
```
**Response**: `ChatResponse`
```json
{
  "response": "NA 701 has a pressure rating of...",
  "sources": [ { "title": "Datasheet", "url": "..." } ],
  "session_id": "string",
  "metadata": { ... }
}
```

### GET /internal/sessions/{session_id}/history
**Summary**: Get conversation history.
**Response**: `SessionHistoryResponse`

### DELETE /internal/sessions/{session_id}
**Summary**: Clear session.
**Response**: `ClearSessionResponse`

### GET /internal/suggestions
**Summary**: Get suggested questions.
**Response**: `SuggestedQuestionsResponse`

### GET /internal/knowledge-stats
**Summary**: Get knowledge base statistics.
**Response**: `KnowledgeStatsResponse`

### GET /internal/search
**Summary**: Search knowledge base.
**Query Params**:
- `query`: string (required)
- `limit`: int (default 10)
**Response**: `SearchResponse`
```json
{
  "query": "string",
  "total_results": 10,
  "results": [ ... ]
}
```

### GET /internal/user/profile
**Summary**: Get user profile.
**Response**: `UserProfileResponse`

---

## 3. Enquiry Management (Internal)

### GET /internal/enquiries/
**Summary**: List enquiries.
**Query Params**:
- `status`: string (optional)
- `priority`: string (optional)
- `assigned_to`: string (optional)
**Response**: `EnquiryListResponse` (schema ref from imported models)

### GET /internal/enquiries/stats
**Summary**: Get enquiry statistics.
**Response**: `EnquiryStatsResponse`

### GET /internal/enquiries/{enquiry_id}
**Summary**: Get enquiry details.
**Response**: `EnquiryDetailsResponse`

### POST /internal/enquiries/{enquiry_id}/assign
**Summary**: Assign enquiry.
**Request Body**: `AssignEnquiryRequest`
```json
{
  "assigned_to": "user_id_or_email",
  "department": "Sales",
  "internal_note": "Please handle this priority customer"
}
```
**Response**: `AssignmentResponse`

### POST /internal/enquiries/{enquiry_id}/note
**Summary**: Add internal note.
**Request Body**: `AddNoteRequest`
```json
{
  "note": "Customer called, verified requirements."
}
```
**Response**: `NoteResponse`

### POST /internal/enquiries/{enquiry_id}/generate-response
**Summary**: Generate AI response.
**Request Body**: `GenerateResponseRequest`
```json
{
  "tone": "professional",
  "include_products": true
}
```
**Response**: `GenerateResponseResponse`
```json
{
  "success": true,
  "suggested_response": {
    "subject": "Re: Your Enquiry",
    "body": "Dear Customer..."
  }
}
```

### POST /internal/enquiries/{enquiry_id}/send-response
**Summary**: Send response.
**Request Body**: `SendResponseRequest`
**Response**: `SendResponseResponse`

### PUT /internal/enquiries/{enquiry_id}/status
**Summary**: Update status.
**Response**: `StatusUpdateResponse`

### POST /internal/enquiries/{enquiry_id}/analyze
**Summary**: Analyze enquiry.
**Response**: `EnquiryAnalysisResponse`

### POST /internal/enquiries/{enquiry_id}/escalate
**Summary**: Escalate enquiry.
**Response**: `EscalationResponse`

---

## 4. Quotation Management

### POST /v1/quotations/external/submit-specific
**Summary**: Submit specific quotation request (Customer).
**Request Body**: `SpecificQuotationRequest`
```json
{
  "customer": {
    "name": "string",
    "email": "string",
    "company": "string"
  },
  "line_items": [
    {
      "product_code": "NA 701",
      "product_name": "Graphite Packing",
      "quantity": 10,
      "size": "10mm x 10mm"
    }
  ]
}
```
**Response**: `QuotationSubmitResponse`

### POST /v1/quotations/external/submit-generic
**Summary**: Submit generic quotation request (Customer).
**Request Body**: `GenericQuotationRequest`
```json
{
  "customer": { ... },
  "message": "I need packing for a steam valve...",
  "industry": "Power Generation"
}
```
**Response**: `QuotationSubmitResponse`

### GET /internal/quotations/
**Summary**: List all quotations.
**Response**: `QuotationListResponse`

### GET /internal/quotations/{request_id}
**Summary**: Get quotation details.
**Response**: `QuotationDetailsResponse`

### PUT /internal/quotations/{request_id}
**Summary**: Update quotation (Internal).
**Request Body**: `UpdateQuotationRequest`
```json
{
  "status": "quoted",
  "internal_notes": "Pricing approved",
  "line_items": [ ... ]
}
```
**Response**: `QuotationUpdateResponse`

### POST /internal/quotations/{request_id}/mark-sent
**Summary**: Mark as sent.
**Response**: `QuotationSentResponse`

### GET /internal/quotations/stats/summary
**Summary**: Get quotation statistics.
**Response**: `QuotationStatsResponse`

### POST /internal/quotations/{request_id}/analyze
**Summary**: Run AI analysis on quotation.
**Response**: `QuotationAnalysisResponse`
```json
{
  "success": true,
  "analysis": {
    "one_liner": "Summary...",
    "estimated_value": "...",
    "sub_agent_results": { ... }
  }
}
```

### POST /internal/quotations/{request_id}/re-analyze
**Summary**: Re-run AI analysis.
**Response**: `QuotationAnalysisResponse`

### GET /internal/quotations/{request_id}/analysis
**Summary**: Get analysis.
**Response**: `QuotationAnalysisResponse`

### GET /internal/quotations/dashboard/overview
**Summary**: Get dashboard.
**Response**: `QuotationDashboardResponse`

---

## 5. Demo Endpoints
Unauthenticated endpoints for testing dashboards.

- **GET /internal/enquiries/dashboard**: Returns sample enquiry dashboard data. `EnquiryDemoDashboardResponse`
- **GET /internal/quotations/dashboard**: Returns sample quotation dashboard data. `QuotationDemoDashboardResponse`
- **GET /demo/quotations**: Returns list of quotations. `QuotationListResponse`
- **POST /demo/quotations/{id}/generate-pdf**: Generates a PDF for a quotation. `PDFGenerationResponse`
- **POST /demo/quotations/{id}/mark-sent**: Marks a quotation as sent. `MarkSentResponse`
- **POST /demo/quotations/{id}/save-prices**: Save prices for a quotation. `SavePricesResponse`


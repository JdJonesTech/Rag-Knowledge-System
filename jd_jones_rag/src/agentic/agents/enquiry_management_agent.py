"""
Enquiry Management Agent
Handles enquiry classification, routing, and auto-response.

Implements:
- Classify incoming enquiries
- Route complex technical queries
- Provide instant responses for common queries
- Update CRM and trigger email routing
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings
from src.agentic.tools.email_tool import EmailTool, EmailCategory, EmailPriority
from src.agentic.tools.crm_tool import CRMTool


class EnquiryType(str, Enum):
    """Types of enquiries."""
    PRODUCT_INQUIRY = "product_inquiry"
    PRICING_REQUEST = "pricing_request"
    TECHNICAL_QUESTION = "technical_question"
    ORDER_STATUS = "order_status"
    COMPLAINT = "complaint"
    PARTNERSHIP = "partnership"
    GENERAL = "general"
    URGENT_TECHNICAL = "urgent_technical"


class ResponseType(str, Enum):
    """Type of response to provide."""
    INSTANT_FAQ = "instant_faq"           # Can answer immediately from FAQ
    RAG_RESPONSE = "rag_response"         # Use RAG to generate response
    ROUTE_TO_TEAM = "route_to_team"       # Route to human team
    HYBRID = "hybrid"                      # Provide initial response + route
    ESCALATE = "escalate"                 # Urgent escalation needed


class RoutingDestination(str, Enum):
    """Where to route enquiries."""
    SALES = "sales"
    TECHNICAL = "technical"
    ENGINEERING = "engineering"
    CUSTOMER_SERVICE = "customer_service"
    MANAGEMENT = "management"


@dataclass
class EnquiryClassification:
    """Classification result for an enquiry."""
    enquiry_type: EnquiryType
    confidence: float
    response_type: ResponseType
    routing_destination: Optional[RoutingDestination]
    priority: EmailPriority
    keywords: List[str]
    entities: Dict[str, List[str]]
    requires_technical_review: bool
    estimated_complexity: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enquiry_type": self.enquiry_type.value,
            "confidence": self.confidence,
            "response_type": self.response_type.value,
            "routing_destination": self.routing_destination.value if self.routing_destination else None,
            "priority": self.priority.value,
            "keywords": self.keywords,
            "entities": self.entities,
            "requires_technical_review": self.requires_technical_review,
            "estimated_complexity": self.estimated_complexity
        }


@dataclass
class EnquiryResponse:
    """Response for an enquiry."""
    enquiry_id: str
    classification: EnquiryClassification
    instant_response: Optional[str]
    routed: bool
    routed_to: Optional[str]
    reference_number: str
    crm_logged: bool
    acknowledgment_sent: bool
    follow_up_scheduled: bool
    actions_taken: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enquiry_id": self.enquiry_id,
            "classification": self.classification.to_dict(),
            "instant_response": self.instant_response,
            "routed": self.routed,
            "routed_to": self.routed_to,
            "reference_number": self.reference_number,
            "crm_logged": self.crm_logged,
            "acknowledgment_sent": self.acknowledgment_sent,
            "follow_up_scheduled": self.follow_up_scheduled,
            "actions_taken": self.actions_taken
        }


class EnquiryManagementAgent:
    """
    Manages incoming enquiries end-to-end.
    
    Workflow:
    1. Classify enquiry type and urgency
    2. Determine response strategy
    3. For simple queries: Provide instant response
    4. For complex queries: Route to appropriate team
    5. Log to CRM
    6. Send acknowledgment
    7. Schedule follow-up if needed
    """
    
    # FAQ database for instant responses
    FAQ_RESPONSES = {
        "shipping_time": {
            "keywords": ["shipping", "delivery", "how long", "lead time", "when"],
            "response": """**Delivery Times**

Standard delivery times are:
- **India**: 2-5 business days
- **Middle East**: 5-7 business days
- **Europe/USA**: 7-10 business days
- **Express shipping**: Available on request

For stock items, orders placed before 2 PM ship same day.

For non-stock items, please check with our sales team for lead times."""
        },
        "returns": {
            "keywords": ["return", "refund", "exchange", "warranty", "defect"],
            "response": """**Returns & Warranty**

JD Jones products come with a comprehensive warranty:
- **Standard products**: 12-month warranty
- **Custom solutions**: 6-month warranty

To initiate a return:
1. Contact our customer service within 30 days
2. Provide order number and reason
3. We'll issue an RMA number
4. Ship product back with RMA

Refunds are processed within 5-7 business days of receiving the product."""
        },
        "payment_terms": {
            "keywords": ["payment", "terms", "credit", "invoice", "pay"],
            "response": """**Payment Terms**

Standard payment terms:
- **New customers**: Advance payment or LC
- **Established customers**: Net 30

We accept:
- Bank transfer
- Letter of Credit (LC)
- Credit cards (for orders under $5,000)

For custom payment arrangements, please contact our accounts team."""
        },
        "certifications": {
            "keywords": ["certified", "certification", "api 622", "api 624", "fda", "approval"],
            "response": """**Certifications**

Our products are certified to multiple international standards:

**Oil & Gas:**
- API 622 (Fugitive Emissions - Packing)
- API 624 (Rising Stem Valves)
- Shell SPE 77/312
- Saudi Aramco approved

**Food & Pharma:**
- FDA 21 CFR 177
- USP Class VI
- 3A Sanitary
- NSF certified

For specific certification requirements, please contact our technical team."""
        }
    }
    
    CLASSIFICATION_PROMPT = """Classify this customer enquiry for JD Jones Manufacturing (industrial sealing solutions).

ENQUIRY:
From: {from_email}
Subject: {subject}
Content: {content}

Classify into:
1. enquiry_type: product_inquiry, pricing_request, technical_question, order_status, complaint, partnership, general, urgent_technical
2. response_type: instant_faq (simple question), rag_response (needs KB search), route_to_team (needs human), hybrid (auto-respond + route), escalate (urgent)
3. routing_destination: sales, technical, engineering, customer_service, management
4. priority: low, normal, high, urgent

INDICATORS:
- Urgent: mentions safety, failure, immediate, down, emergency
- Technical: mentions specifications, compatibility, standards, API, temperature, pressure
- Pricing: mentions cost, quote, budget, price, discount
- Complaint: mentions problem, issue, dissatisfied, wrong

Respond in JSON:
{{
    "enquiry_type": "string",
    "confidence": 0.0-1.0,
    "response_type": "string",
    "routing_destination": "string or null",
    "priority": "string",
    "keywords": ["keyword1", "keyword2"],
    "entities": {{
        "products": ["if mentioned"],
        "companies": ["if mentioned"],
        "order_numbers": ["if mentioned"],
        "standards": ["if mentioned"]
    }},
    "requires_technical_review": true/false,
    "complexity": "low/medium/high",
    "reasoning": "brief explanation"
}}
"""

    def __init__(self, retriever=None):
        """
        Initialize enquiry management agent.
        
        Args:
            retriever: Knowledge base retriever for RAG responses
        """
        from src.config.settings import get_llm
        self.llm = get_llm(temperature=0)
        self.retriever = retriever
        self.email_tool = EmailTool()
        self.crm_tool = CRMTool()
    
    async def process_enquiry(
        self,
        enquiry_content: str,
        from_email: str,
        subject: str = "",
        customer_name: Optional[str] = None,
        company: Optional[str] = None
    ) -> EnquiryResponse:
        """
        Process an incoming enquiry end-to-end.
        
        Args:
            enquiry_content: The enquiry text
            from_email: Sender's email
            subject: Email subject
            customer_name: Customer name if known
            company: Company name if known
            
        Returns:
            EnquiryResponse with classification, response, and actions taken
        """
        import uuid
        enquiry_id = f"ENQ-{uuid.uuid4().hex[:8].upper()}"
        actions_taken = []
        
        # Step 1: Classify the enquiry
        classification = await self._classify_enquiry(
            content=enquiry_content,
            from_email=from_email,
            subject=subject
        )
        actions_taken.append(f"Classified as: {classification.enquiry_type.value}")
        
        # Step 2: Generate instant response if possible
        instant_response = None
        if classification.response_type in [ResponseType.INSTANT_FAQ, ResponseType.HYBRID]:
            instant_response = await self._generate_instant_response(
                content=enquiry_content,
                classification=classification
            )
            if instant_response:
                actions_taken.append("Generated instant response")
        
        # Step 3: Route if needed
        routed = False
        routed_to = None
        reference_number = enquiry_id
        
        if classification.response_type in [ResponseType.ROUTE_TO_TEAM, ResponseType.HYBRID, ResponseType.ESCALATE]:
            route_result = await self.email_tool.execute(
                query="Route enquiry",
                parameters={
                    "action": "route",
                    "category": self._map_destination_to_email_category(classification.routing_destination),
                    "priority": classification.priority.value,
                    "customer_email": from_email,
                    "subject": subject
                }
            )
            
            if route_result.success:
                routed = True
                routed_to = route_result.data.get("routed_to")
                reference_number = route_result.data.get("reference_id", enquiry_id)
                actions_taken.append(f"Routed to: {routed_to}")
        
        # Step 4: Log to CRM
        crm_logged = False
        crm_result = await self.crm_tool.execute(
            query="Log enquiry",
            parameters={
                "action": "log_interaction",
                "customer_email": from_email,
                "company": company,
                "interaction_type": "enquiry",
                "subject": subject,
                "summary": enquiry_content[:500],
                "priority": classification.priority.value,
                "products": classification.entities.get("products", [])
            }
        )
        
        if crm_result.success:
            crm_logged = True
            actions_taken.append("Logged to CRM")
        
        # Step 5: Send acknowledgment
        acknowledgment_sent = False
        if classification.response_type != ResponseType.INSTANT_FAQ:
            ack_result = await self.email_tool.execute(
                query="Send acknowledgment",
                parameters={
                    "action": "send_acknowledgment",
                    "customer_name": customer_name or "Valued Customer",
                    "customer_email": from_email,
                    "subject": subject,
                    "reference_id": reference_number,
                    "department": classification.routing_destination.value if classification.routing_destination else "Customer Service",
                    "response_time_hours": self._get_response_time(classification.priority)
                }
            )
            
            if ack_result.success:
                acknowledgment_sent = True
                actions_taken.append("Sent acknowledgment email")
        
        # Step 6: Schedule follow-up for high priority
        follow_up_scheduled = False
        if classification.priority in [EmailPriority.HIGH, EmailPriority.URGENT]:
            followup_result = await self.crm_tool.execute(
                query="Schedule follow-up",
                parameters={
                    "action": "schedule_followup",
                    "customer_email": from_email,
                    "due_date": self._calculate_followup_date(classification.priority),
                    "notes": f"High priority enquiry: {subject}",
                    "priority": classification.priority.value
                }
            )
            
            if followup_result.success:
                follow_up_scheduled = True
                actions_taken.append("Scheduled follow-up task")
        
        return EnquiryResponse(
            enquiry_id=enquiry_id,
            classification=classification,
            instant_response=instant_response,
            routed=routed,
            routed_to=routed_to,
            reference_number=reference_number,
            crm_logged=crm_logged,
            acknowledgment_sent=acknowledgment_sent,
            follow_up_scheduled=follow_up_scheduled,
            actions_taken=actions_taken
        )
    
    async def _classify_enquiry(
        self,
        content: str,
        from_email: str,
        subject: str
    ) -> EnquiryClassification:
        """Classify an enquiry using LLM."""
        prompt = self.CLASSIFICATION_PROMPT.format(
            from_email=from_email,
            subject=subject,
            content=content
        )
        
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        
        # Parse response
        response_content = response.content
        if "{" in response_content:
            start = response_content.index("{")
            end = response_content.rindex("}") + 1
            result = json.loads(response_content[start:end])
            
            return EnquiryClassification(
                enquiry_type=EnquiryType(result.get("enquiry_type", "general")),
                confidence=result.get("confidence", 0.8),
                response_type=ResponseType(result.get("response_type", "route_to_team")),
                routing_destination=RoutingDestination(result["routing_destination"]) if result.get("routing_destination") else None,
                priority=EmailPriority(result.get("priority", "normal")),
                keywords=result.get("keywords", []),
                entities=result.get("entities", {}),
                requires_technical_review=result.get("requires_technical_review", False),
                estimated_complexity=result.get("complexity", "medium")
            )
        
        # Default classification
        return EnquiryClassification(
            enquiry_type=EnquiryType.GENERAL,
            confidence=0.5,
            response_type=ResponseType.ROUTE_TO_TEAM,
            routing_destination=RoutingDestination.CUSTOMER_SERVICE,
            priority=EmailPriority.NORMAL,
            keywords=[],
            entities={},
            requires_technical_review=False,
            estimated_complexity="medium"
        )
    
    async def _generate_instant_response(
        self,
        content: str,
        classification: EnquiryClassification
    ) -> Optional[str]:
        """Generate an instant response for simple queries."""
        content_lower = content.lower()
        
        # Check FAQ database
        for faq_key, faq_data in self.FAQ_RESPONSES.items():
            if any(kw in content_lower for kw in faq_data["keywords"]):
                return faq_data["response"]
        
        # If RAG response is needed, use retriever
        if classification.response_type == ResponseType.RAG_RESPONSE and self.retriever:
            try:
                from src.knowledge_base.retriever import UserRole
                results = self.retriever.retrieve(
                    query=content,
                    user_role=UserRole.EMPLOYEE,
                    n_results=3
                )
                
                if results.all_results:
                    # Format response from retrieved content
                    context = "\n".join([r.content[:300] for r in results.all_results[:2]])
                    
                    response_prompt = f"""Based on this context, provide a helpful response to the customer enquiry.
Be concise and professional.

Context:
{context}

Enquiry:
{content}

Response:"""
                    
                    messages = [HumanMessage(content=response_prompt)]
                    llm_response = await self.llm.ainvoke(messages)
                    return llm_response.content
            except Exception:
                pass
        
        return None
    
    def _map_destination_to_email_category(
        self,
        destination: Optional[RoutingDestination]
    ) -> str:
        """Map routing destination to email category."""
        mapping = {
            RoutingDestination.SALES: "sales",
            RoutingDestination.TECHNICAL: "technical",
            RoutingDestination.ENGINEERING: "engineering",
            RoutingDestination.CUSTOMER_SERVICE: "customer_service",
            RoutingDestination.MANAGEMENT: "general"
        }
        return mapping.get(destination, "customer_service")
    
    def _get_response_time(self, priority: EmailPriority) -> int:
        """Get expected response time in hours."""
        times = {
            EmailPriority.URGENT: 1,
            EmailPriority.HIGH: 4,
            EmailPriority.NORMAL: 8,
            EmailPriority.LOW: 24
        }
        return times.get(priority, 8)
    
    def _calculate_followup_date(self, priority: EmailPriority) -> str:
        """Calculate follow-up date based on priority."""
        from datetime import timedelta
        
        days = {
            EmailPriority.URGENT: 0,
            EmailPriority.HIGH: 1,
            EmailPriority.NORMAL: 3,
            EmailPriority.LOW: 7
        }
        
        followup = datetime.now() + timedelta(days=days.get(priority, 3))
        return followup.strftime("%Y-%m-%d")
    
    async def get_enquiry_stats(self) -> Dict[str, Any]:
        """Get enquiry processing statistics."""
        # In production, query database
        return {
            "today": {
                "total": 0,
                "by_type": {},
                "by_priority": {},
                "avg_response_time_hours": 0
            },
            "week": {
                "total": 0,
                "by_type": {},
                "routed_vs_instant": {"routed": 0, "instant": 0}
            }
        }

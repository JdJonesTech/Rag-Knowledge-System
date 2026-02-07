"""
CRM Tool
Integrates with Customer Relationship Management system.
This is an ACTION tool that can read and write CRM data.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

from src.agentic.tools.base_tool import BaseTool, ToolResult, ToolStatus


class LeadStatus(str, Enum):
    """Lead/opportunity status."""
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    WON = "won"
    LOST = "lost"


class InteractionType(str, Enum):
    """Customer interaction types."""
    ENQUIRY = "enquiry"
    QUOTE_REQUEST = "quote_request"
    TECHNICAL_SUPPORT = "technical_support"
    COMPLAINT = "complaint"
    FOLLOW_UP = "follow_up"
    ORDER = "order"


class CRMTool(BaseTool):
    """
    Tool for CRM operations.
    
    Capabilities:
    - Create/update customer records
    - Log interactions
    - Create leads/opportunities
    - Track enquiry history
    - Schedule follow-ups
    """
    
    # Sample CRM data (in production, use real CRM API)
    CUSTOMERS = {
        "CUST-001": {
            "company": "Saudi Aramco",
            "contact_name": "Ahmed Al-Farsi",
            "email": "afarsi@aramco.sa",
            "phone": "+966-12-345-6789",
            "industry": "Oil & Gas",
            "region": "Middle East",
            "tier": "enterprise",
            "total_orders": 15,
            "total_value": 450000,
            "last_order": "2024-10-15"
        },
        "CUST-002": {
            "company": "Shell Chemicals",
            "contact_name": "Jan de Vries",
            "email": "jdevries@shell.com",
            "phone": "+31-70-123-4567",
            "industry": "Petrochemical",
            "region": "Europe",
            "tier": "enterprise",
            "total_orders": 8,
            "total_value": 180000,
            "last_order": "2024-11-01"
        },
        "CUST-003": {
            "company": "Reliance Industries",
            "contact_name": "Priya Sharma",
            "email": "psharma@ril.com",
            "phone": "+91-22-4567-8901",
            "industry": "Petrochemical",
            "region": "India",
            "tier": "enterprise",
            "total_orders": 22,
            "total_value": 320000,
            "last_order": "2024-11-10"
        }
    }
    
    INTERACTIONS = []
    LEADS = []
    
    def __init__(self):
        """Initialize CRM tool."""
        super().__init__(
            name="crm",
            description="""
            CRM operations:
            - Look up customer information
            - Log interactions/enquiries
            - Create leads and opportunities
            - Update customer records
            - Get customer history
            
            This is an ACTION tool that modifies CRM data.
            """
        )
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """Execute CRM operation."""
        try:
            action = parameters.get("action", "lookup")
            
            if action == "lookup":
                return await self._lookup_customer(parameters)
            
            elif action == "log_interaction":
                return await self._log_interaction(parameters)
            
            elif action == "create_lead":
                return await self._create_lead(parameters)
            
            elif action == "update_customer":
                return await self._update_customer(parameters)
            
            elif action == "get_history":
                return await self._get_customer_history(parameters)
            
            elif action == "schedule_followup":
                return await self._schedule_followup(parameters)
            
            else:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.PARTIAL,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _lookup_customer(self, parameters: Dict[str, Any]) -> ToolResult:
        """Look up customer information."""
        customer_id = parameters.get("customer_id")
        email = parameters.get("email")
        company = parameters.get("company")
        
        # Search by ID
        if customer_id and customer_id in self.CUSTOMERS:
            customer = self.CUSTOMERS[customer_id]
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data={
                    "customer_id": customer_id,
                    "customer": customer,
                    "is_existing": True
                }
            )
        
        # Search by email or company
        for cid, customer in self.CUSTOMERS.items():
            if email and customer["email"].lower() == email.lower():
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.SUCCESS,
                    data={"customer_id": cid, "customer": customer, "is_existing": True}
                )
            if company and company.lower() in customer["company"].lower():
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.SUCCESS,
                    data={"customer_id": cid, "customer": customer, "is_existing": True}
                )
        
        # Not found
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.PARTIAL,
            data={
                "is_existing": False,
                "searched": {
                    "customer_id": customer_id,
                    "email": email,
                    "company": company
                }
            }
        )
    
    async def _log_interaction(self, parameters: Dict[str, Any]) -> ToolResult:
        """Log a customer interaction."""
        interaction_id = f"INT-{uuid.uuid4().hex[:8].upper()}"
        
        interaction = {
            "interaction_id": interaction_id,
            "customer_id": parameters.get("customer_id"),
            "customer_email": parameters.get("customer_email"),
            "company": parameters.get("company"),
            "type": parameters.get("interaction_type", "enquiry"),
            "channel": parameters.get("channel", "web"),
            "subject": parameters.get("subject"),
            "summary": parameters.get("summary"),
            "products_discussed": parameters.get("products", []),
            "assigned_to": parameters.get("assigned_to"),
            "priority": parameters.get("priority", "normal"),
            "status": "open",
            "created_at": datetime.now().isoformat()
        }
        
        self.INTERACTIONS.append(interaction)
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "action": "interaction_logged",
                "interaction_id": interaction_id,
                "status": "open"
            }
        )
    
    async def _create_lead(self, parameters: Dict[str, Any]) -> ToolResult:
        """Create a new lead/opportunity."""
        lead_id = f"LEAD-{uuid.uuid4().hex[:8].upper()}"
        
        lead = {
            "lead_id": lead_id,
            "company": parameters.get("company"),
            "contact_name": parameters.get("contact_name"),
            "email": parameters.get("email"),
            "phone": parameters.get("phone"),
            "source": parameters.get("source", "web_enquiry"),
            "status": LeadStatus.NEW.value,
            "estimated_value": parameters.get("estimated_value", 0),
            "products_interested": parameters.get("products", []),
            "industry": parameters.get("industry"),
            "notes": parameters.get("notes"),
            "assigned_to": parameters.get("assigned_to"),
            "created_at": datetime.now().isoformat()
        }
        
        self.LEADS.append(lead)
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "action": "lead_created",
                "lead_id": lead_id,
                "status": LeadStatus.NEW.value,
                "follow_up_required": True
            }
        )
    
    async def _update_customer(self, parameters: Dict[str, Any]) -> ToolResult:
        """Update customer record."""
        customer_id = parameters.get("customer_id")
        updates = parameters.get("updates", {})
        
        if customer_id not in self.CUSTOMERS:
            # Create new customer
            new_id = f"CUST-{uuid.uuid4().hex[:6].upper()}"
            self.CUSTOMERS[new_id] = {
                "company": parameters.get("company", "Unknown"),
                "contact_name": parameters.get("contact_name"),
                "email": parameters.get("email"),
                "phone": parameters.get("phone"),
                "industry": parameters.get("industry"),
                "region": parameters.get("region"),
                "tier": "standard",
                "total_orders": 0,
                "total_value": 0,
                "created_at": datetime.now().isoformat()
            }
            
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data={
                    "action": "customer_created",
                    "customer_id": new_id
                }
            )
        
        # Update existing
        self.CUSTOMERS[customer_id].update(updates)
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "action": "customer_updated",
                "customer_id": customer_id,
                "fields_updated": list(updates.keys())
            }
        )
    
    async def _get_customer_history(self, parameters: Dict[str, Any]) -> ToolResult:
        """Get customer interaction history."""
        customer_id = parameters.get("customer_id")
        email = parameters.get("email")
        
        # Filter interactions
        history = [
            i for i in self.INTERACTIONS
            if i.get("customer_id") == customer_id or i.get("customer_email") == email
        ]
        
        # Sort by date
        history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "customer_id": customer_id,
                "total_interactions": len(history),
                "recent_interactions": history[:10]
            }
        )
    
    async def _schedule_followup(self, parameters: Dict[str, Any]) -> ToolResult:
        """Schedule a follow-up task."""
        task_id = f"TASK-{uuid.uuid4().hex[:8].upper()}"
        
        followup = {
            "task_id": task_id,
            "customer_id": parameters.get("customer_id"),
            "interaction_id": parameters.get("interaction_id"),
            "type": "follow_up",
            "due_date": parameters.get("due_date"),
            "assigned_to": parameters.get("assigned_to"),
            "notes": parameters.get("notes"),
            "priority": parameters.get("priority", "normal"),
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "action": "followup_scheduled",
                "task_id": task_id,
                "due_date": parameters.get("due_date")
            }
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["lookup", "log_interaction", "create_lead", "update_customer", "get_history", "schedule_followup"]
                },
                "customer_id": {"type": "string"},
                "email": {"type": "string"},
                "company": {"type": "string"},
                "contact_name": {"type": "string"},
                "interaction_type": {
                    "type": "string",
                    "enum": ["enquiry", "quote_request", "technical_support", "complaint", "follow_up", "order"]
                },
                "subject": {"type": "string"},
                "summary": {"type": "string"},
                "products": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "string"},
                "assigned_to": {"type": "string"},
                "notes": {"type": "string"},
                "due_date": {"type": "string"}
            }
        }

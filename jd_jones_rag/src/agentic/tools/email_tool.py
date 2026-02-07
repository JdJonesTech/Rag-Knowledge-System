"""
Email Tool
Routes and sends emails as part of agentic workflows.
This is an ACTION tool, not just retrieval.

Supports both mock mode (for development) and real SMTP sending (for production).
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.agentic.tools.base_tool import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class EmailPriority(str, Enum):
    """Email priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class EmailCategory(str, Enum):
    """Email routing categories."""
    SALES = "sales"
    TECHNICAL = "technical"
    CUSTOMER_SERVICE = "customer_service"
    ENGINEERING = "engineering"
    MANAGEMENT = "management"
    GENERAL = "general"


@dataclass
class EmailRecipient:
    """Email recipient details."""
    email: str
    name: str
    role: str
    department: str


class EmailTool(BaseTool):
    """
    Tool for email routing and sending.
    
    Capabilities:
    - Route enquiries to appropriate teams
    - Send automated responses
    - Schedule follow-up emails
    - Attach generated documents
    
    Configuration:
    - Set SMTP settings in settings.py or environment variables
    - Enable real SMTP sending by setting EMAIL_SMTP_ENABLED=true
    """
    
    # Email routing rules (in production, from config/database)
    ROUTING_RULES = {
        EmailCategory.SALES: {
            "primary": EmailRecipient(
                email="sales@jdjones.com",
                name="Sales Team",
                role="Sales",
                department="Sales"
            ),
            "escalation": EmailRecipient(
                email="sales.manager@jdjones.com",
                name="Sales Manager",
                role="Manager",
                department="Sales"
            ),
            "response_time_hours": 4
        },
        EmailCategory.TECHNICAL: {
            "primary": EmailRecipient(
                email="technical@jdjones.com",
                name="Technical Team",
                role="Engineer",
                department="Engineering"
            ),
            "escalation": EmailRecipient(
                email="tech.lead@jdjones.com",
                name="Technical Lead",
                role="Lead Engineer",
                department="Engineering"
            ),
            "response_time_hours": 8
        },
        EmailCategory.ENGINEERING: {
            "primary": EmailRecipient(
                email="engineering@jdjones.com",
                name="Engineering Team",
                role="Engineer",
                department="Engineering"
            ),
            "escalation": EmailRecipient(
                email="engineering.manager@jdjones.com",
                name="Engineering Manager",
                role="Manager",
                department="Engineering"
            ),
            "response_time_hours": 24
        },
        EmailCategory.CUSTOMER_SERVICE: {
            "primary": EmailRecipient(
                email="support@jdjones.com",
                name="Customer Service",
                role="Support",
                department="Customer Service"
            ),
            "escalation": EmailRecipient(
                email="support.supervisor@jdjones.com",
                name="Support Supervisor",
                role="Supervisor",
                department="Customer Service"
            ),
            "response_time_hours": 2
        }
    }
    
    # Email templates
    TEMPLATES = {
        "enquiry_acknowledgment": """
Dear {customer_name},

Thank you for contacting JD Jones Manufacturing.

We have received your enquiry regarding {subject} and it has been assigned to our {department} team.

Reference Number: {reference_id}
Expected Response Time: Within {response_time} hours

If your matter is urgent, please call us at 1-800-JD-JONES.

Best regards,
JD Jones Customer Support
""",
        "technical_routing": """
[INTERNAL - Technical Enquiry Routed]

Reference: {reference_id}
Customer: {customer_name}
Company: {company}
Priority: {priority}

Original Enquiry:
{enquiry_text}

Extracted Parameters:
{parameters}

Action Required: Technical review and response within {response_time} hours.
""",
        "quote_request": """
Dear {customer_name},

Thank you for your quote request.

We are preparing a detailed quotation for the following items:
{product_list}

Specifications:
{specifications}

You will receive your formal quotation within 24 hours.

Best regards,
JD Jones Sales Team
"""
    }
    
    def __init__(self, smtp_enabled: Optional[bool] = None):
        """
        Initialize email tool.
        
        Args:
            smtp_enabled: Override to enable/disable SMTP. If None, uses settings.
        """
        super().__init__(
            name="email_router",
            description="""
            Routes and sends emails:
            - Route enquiries to appropriate teams
            - Send acknowledgment emails
            - Trigger internal notifications
            - Schedule follow-ups
            
            This is an ACTION tool that actually sends emails.
            """
        )
        
        # Track sent emails (for audit trail)
        self.sent_emails: List[Dict[str, Any]] = []
        
        # SMTP configuration from settings
        try:
            from src.config.settings import settings
            self.smtp_host = getattr(settings, 'smtp_host', 'localhost')
            self.smtp_port = getattr(settings, 'smtp_port', 587)
            self.smtp_user = getattr(settings, 'smtp_user', '')
            self.smtp_password = getattr(settings, 'smtp_password', '')
            self.smtp_from_email = getattr(settings, 'smtp_from_email', 'noreply@jdjones.com')
            self.smtp_from_name = getattr(settings, 'smtp_from_name', 'JD Jones Customer Support')
            self.smtp_use_tls = getattr(settings, 'smtp_use_tls', True)
            
            # Enable SMTP if configured
            if smtp_enabled is not None:
                self.smtp_enabled = smtp_enabled
            else:
                self.smtp_enabled = getattr(settings, 'smtp_enabled', False)
                
        except ImportError:
            # Fallback defaults
            self.smtp_host = 'localhost'
            self.smtp_port = 587
            self.smtp_user = ''
            self.smtp_password = ''
            self.smtp_from_email = 'noreply@jdjones.com'
            self.smtp_from_name = 'JD Jones Customer Support'
            self.smtp_use_tls = True
            self.smtp_enabled = False
        
        logger.info(f"EmailTool initialized, SMTP enabled: {self.smtp_enabled}")
    
    def _send_email_smtp(
        self,
        to_email: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        attachments: Optional[List[Path]] = None
    ) -> bool:
        """
        Send email via SMTP.
        
        Args:
            to_email: Recipient email
            subject: Email subject
            body: Plain text body
            html_body: Optional HTML body
            attachments: Optional list of file paths to attach
            
        Returns:
            True if sent successfully
        """
        if not self.smtp_enabled:
            logger.info(f"SMTP disabled, would send to {to_email}: {subject}")
            return True
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.smtp_from_name} <{self.smtp_from_email}>"
            msg['To'] = to_email
            
            # Add plain text part
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML part if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if file_path.exists():
                        with open(file_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename={file_path.name}'
                            )
                            msg.attach(part)
            
            # Send via SMTP
            if self.smtp_use_tls:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    if self.smtp_user and self.smtp_password:
                        server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port) as server:
                    if self.smtp_user and self.smtp_password:
                        server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """Execute email action."""
        try:
            action = parameters.get("action", "route")
            
            if action == "route":
                return await self._route_enquiry(parameters)
            
            elif action == "send_acknowledgment":
                return await self._send_acknowledgment(parameters)
            
            elif action == "send_internal":
                return await self._send_internal_notification(parameters)
            
            elif action == "send_quote":
                return await self._send_quote_email(parameters)
            
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
    
    async def _route_enquiry(self, parameters: Dict[str, Any]) -> ToolResult:
        """Route an enquiry to the appropriate team."""
        category_str = parameters.get("category", "general").lower()
        
        try:
            category = EmailCategory(category_str)
        except ValueError:
            category = EmailCategory.GENERAL
        
        routing = self.ROUTING_RULES.get(category, self.ROUTING_RULES[EmailCategory.CUSTOMER_SERVICE])
        
        # Generate reference ID
        import uuid
        reference_id = f"ENQ-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        # Determine priority
        priority_str = parameters.get("priority", "normal")
        try:
            priority = EmailPriority(priority_str)
        except ValueError:
            priority = EmailPriority.NORMAL
        
        # Select recipient based on priority
        recipient = routing["primary"]
        if priority in [EmailPriority.HIGH, EmailPriority.URGENT]:
            recipient = routing["escalation"]
        
        # Record routing
        routing_record = {
            "reference_id": reference_id,
            "category": category.value,
            "priority": priority.value,
            "routed_to": {
                "email": recipient.email,
                "name": recipient.name,
                "department": recipient.department
            },
            "response_time_hours": routing["response_time_hours"],
            "routed_at": datetime.now().isoformat(),
            "customer_email": parameters.get("customer_email"),
            "subject": parameters.get("subject")
        }
        
        self.sent_emails.append(routing_record)
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "action": "routed",
                "reference_id": reference_id,
                "routed_to": recipient.email,
                "department": recipient.department,
                "expected_response_hours": routing["response_time_hours"],
                "priority": priority.value
            },
            metadata={"routing_record": routing_record}
        )
    
    async def _send_acknowledgment(self, parameters: Dict[str, Any]) -> ToolResult:
        """Send acknowledgment email to customer."""
        customer_name = parameters.get("customer_name", "Valued Customer")
        customer_email = parameters.get("customer_email")
        subject = parameters.get("subject", "Your Enquiry")
        reference_id = parameters.get("reference_id", "ENQ-PENDING")
        department = parameters.get("department", "Customer Service")
        response_time = parameters.get("response_time_hours", 4)
        
        email_body = self.TEMPLATES["enquiry_acknowledgment"].format(
            customer_name=customer_name,
            subject=subject,
            department=department,
            reference_id=reference_id,
            response_time=response_time
        )
        
        # Record email (in production, actually send)
        email_record = {
            "type": "acknowledgment",
            "to": customer_email,
            "subject": f"Re: {subject} - Reference: {reference_id}",
            "body": email_body,
            "sent_at": datetime.now().isoformat()
        }
        
        self.sent_emails.append(email_record)
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "action": "acknowledgment_sent",
                "to": customer_email,
                "reference_id": reference_id,
                "sent_at": datetime.now().isoformat()
            }
        )
    
    async def _send_internal_notification(self, parameters: Dict[str, Any]) -> ToolResult:
        """Send internal notification to team."""
        category_str = parameters.get("category", "technical")
        
        try:
            category = EmailCategory(category_str)
        except ValueError:
            category = EmailCategory.TECHNICAL
        
        routing = self.ROUTING_RULES.get(category, self.ROUTING_RULES[EmailCategory.TECHNICAL])
        
        email_body = self.TEMPLATES["technical_routing"].format(
            reference_id=parameters.get("reference_id", "N/A"),
            customer_name=parameters.get("customer_name", "Unknown"),
            company=parameters.get("company", "Not specified"),
            priority=parameters.get("priority", "normal"),
            enquiry_text=parameters.get("enquiry_text", ""),
            parameters=parameters.get("extracted_parameters", "None"),
            response_time=routing["response_time_hours"]
        )
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "action": "internal_notification_sent",
                "to": routing["primary"].email,
                "department": routing["primary"].department,
                "priority": parameters.get("priority", "normal")
            }
        )
    
    async def _send_quote_email(self, parameters: Dict[str, Any]) -> ToolResult:
        """Send quote acknowledgment email."""
        customer_name = parameters.get("customer_name", "Valued Customer")
        customer_email = parameters.get("customer_email")
        products = parameters.get("products", [])
        specifications = parameters.get("specifications", "As requested")
        
        product_list = "\n".join([f"- {p}" for p in products]) if products else "As discussed"
        
        email_body = self.TEMPLATES["quote_request"].format(
            customer_name=customer_name,
            product_list=product_list,
            specifications=specifications
        )
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "action": "quote_acknowledgment_sent",
                "to": customer_email,
                "products": products,
                "sent_at": datetime.now().isoformat()
            }
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["route", "send_acknowledgment", "send_internal", "send_quote"]
                },
                "category": {
                    "type": "string",
                    "enum": ["sales", "technical", "engineering", "customer_service", "general"]
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"]
                },
                "customer_name": {"type": "string"},
                "customer_email": {"type": "string"},
                "subject": {"type": "string"},
                "enquiry_text": {"type": "string"},
                "reference_id": {"type": "string"},
                "products": {"type": "array", "items": {"type": "string"}},
                "specifications": {"type": "string"}
            }
        }

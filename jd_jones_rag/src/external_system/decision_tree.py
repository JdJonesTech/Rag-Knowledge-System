"""
Decision Tree for External Customer Portal.
Manages customer journey through structured paths with form collection.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.external_system.classifier import CustomerIntent


class NodeType(str, Enum):
    """Types of decision tree nodes."""
    ROOT = "root"
    QUESTION = "question"
    INFORMATION = "information"
    FORM = "form"
    ACTION = "action"
    TERMINAL = "terminal"
    HANDOFF = "handoff"


class FormFieldType(str, Enum):
    """Types of form fields."""
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    NUMBER = "number"
    SELECT = "select"
    MULTISELECT = "multiselect"
    TEXTAREA = "textarea"
    DATE = "date"
    FILE = "file"


@dataclass
class FormField:
    """Definition of a form field."""
    name: str
    label: str
    field_type: FormFieldType
    required: bool = True
    placeholder: str = ""
    options: List[str] = field(default_factory=list)  # For select/multiselect
    validation_pattern: Optional[str] = None
    help_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "label": self.label,
            "type": self.field_type.value,
            "required": self.required,
            "placeholder": self.placeholder,
            "options": self.options,
            "validation_pattern": self.validation_pattern,
            "help_text": self.help_text
        }


@dataclass
class TreeNode:
    """Node in the decision tree."""
    node_id: str
    node_type: NodeType
    title: str
    content: str
    options: List[Dict[str, str]] = field(default_factory=list)  # {label, next_node_id}
    form_fields: List[FormField] = field(default_factory=list)
    action_type: Optional[str] = None  # For action nodes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "title": self.title,
            "content": self.content,
            "options": self.options,
            "form_fields": [f.to_dict() for f in self.form_fields],
            "action_type": self.action_type,
            "metadata": self.metadata
        }


class DecisionTree:
    """
    Decision tree for customer portal navigation.
    Manages customer journey through structured question paths.
    """
    
    def __init__(self):
        """Initialize decision tree with default structure."""
        self.nodes: Dict[str, TreeNode] = {}
        self._build_default_tree()
    
    def _build_default_tree(self):
        """Build the default customer decision tree."""
        
        # ==========================================
        # ROOT NODE
        # ==========================================
        self.nodes["root"] = TreeNode(
            node_id="root",
            node_type=NodeType.ROOT,
            title="Welcome to JD Jones Customer Portal",
            content="Hello! I'm here to help you. What would you like assistance with today?",
            options=[
                {"label": "Product Information", "next_node_id": "product_main"},
                {"label": "Get a Quote", "next_node_id": "quote_main"},
                {"label": "Track My Order", "next_node_id": "order_tracking"},
                {"label": "Technical Support", "next_node_id": "support_main"},
                {"label": "Returns & Warranty", "next_node_id": "returns_main"},
                {"label": "Contact Us", "next_node_id": "contact_main"},
            ]
        )
        
        # ==========================================
        # PRODUCT INFORMATION BRANCH
        # ==========================================
        self.nodes["product_main"] = TreeNode(
            node_id="product_main",
            node_type=NodeType.QUESTION,
            title="Product Information",
            content="Which product category are you interested in?",
            options=[
                {"label": "Industrial Equipment", "next_node_id": "product_industrial"},
                {"label": "Custom Solutions", "next_node_id": "product_custom"},
                {"label": "Replacement Parts", "next_node_id": "product_parts"},
                {"label": "Browse All Products", "next_node_id": "product_catalog"},
                {"label": "Back to Main Menu", "next_node_id": "root"},
            ]
        )
        
        self.nodes["product_industrial"] = TreeNode(
            node_id="product_industrial",
            node_type=NodeType.INFORMATION,
            title="Industrial Equipment",
            content="""Our industrial equipment line includes:

• **Heavy Machinery** - Designed for manufacturing and production environments
• **Assembly Systems** - Automated and semi-automated assembly solutions
• **Material Handling** - Conveyors, lifts, and transport systems
• **Safety Equipment** - Guards, barriers, and safety systems

All equipment meets ISO 9001 quality standards and comes with a 2-year warranty.""",
            options=[
                {"label": "Request Specifications", "next_node_id": "product_specs_form"},
                {"label": "Get a Quote", "next_node_id": "quote_main"},
                {"label": "Back to Products", "next_node_id": "product_main"},
            ]
        )
        
        self.nodes["product_custom"] = TreeNode(
            node_id="product_custom",
            node_type=NodeType.INFORMATION,
            title="Custom Solutions",
            content="""We specialize in custom manufacturing solutions tailored to your specific needs.

Our engineering team can help with:
• Custom design and engineering
• Prototype development
• Small to large batch production
• Material selection consulting
• Integration with existing systems

Lead times vary based on complexity, typically 4-12 weeks.""",
            options=[
                {"label": "Request Custom Quote", "next_node_id": "quote_custom_form"},
                {"label": "Schedule Consultation", "next_node_id": "contact_sales_form"},
                {"label": "Back to Products", "next_node_id": "product_main"},
            ]
        )
        
        self.nodes["product_parts"] = TreeNode(
            node_id="product_parts",
            node_type=NodeType.INFORMATION,
            title="Replacement Parts",
            content="""We maintain inventory of replacement parts for all current and legacy products.

• Most common parts ship within 24-48 hours
• Extended warranty options available
• Technical support for installation
• Bulk discounts for maintenance contracts""",
            options=[
                {"label": "Search Parts Catalog", "next_node_id": "parts_search_form"},
                {"label": "Contact Parts Department", "next_node_id": "contact_parts"},
                {"label": "Back to Products", "next_node_id": "product_main"},
            ]
        )
        
        self.nodes["product_catalog"] = TreeNode(
            node_id="product_catalog",
            node_type=NodeType.INFORMATION,
            title="Product Catalog",
            content="""Download our complete product catalog or request a printed copy.

The catalog includes:
• Full product specifications
• Pricing information
• Technical drawings
• Ordering information""",
            options=[
                {"label": "Download PDF Catalog", "next_node_id": "catalog_download"},
                {"label": "Request Printed Copy", "next_node_id": "catalog_request_form"},
                {"label": "Back to Products", "next_node_id": "product_main"},
            ]
        )
        
        self.nodes["product_specs_form"] = TreeNode(
            node_id="product_specs_form",
            node_type=NodeType.FORM,
            title="Request Product Specifications",
            content="Please provide your contact information and we'll send the specifications.",
            form_fields=[
                FormField("name", "Full Name", FormFieldType.TEXT, required=True),
                FormField("email", "Email Address", FormFieldType.EMAIL, required=True),
                FormField("company", "Company Name", FormFieldType.TEXT, required=False),
                FormField("product_interest", "Products of Interest", FormFieldType.TEXTAREA,
                         placeholder="Which products would you like specifications for?"),
            ],
            options=[
                {"label": "Submit", "next_node_id": "form_submitted"},
                {"label": "Cancel", "next_node_id": "product_main"},
            ],
            action_type="send_specs"
        )
        
        # ==========================================
        # QUOTE REQUEST BRANCH
        # ==========================================
        self.nodes["quote_main"] = TreeNode(
            node_id="quote_main",
            node_type=NodeType.QUESTION,
            title="Quote Request",
            content="What type of quote do you need?",
            options=[
                {"label": "Standard Products", "next_node_id": "quote_standard_form"},
                {"label": "Custom Solution", "next_node_id": "quote_custom_form"},
                {"label": "Bulk/Volume Order", "next_node_id": "quote_bulk_form"},
                {"label": "Back to Main Menu", "next_node_id": "root"},
            ]
        )
        
        self.nodes["quote_standard_form"] = TreeNode(
            node_id="quote_standard_form",
            node_type=NodeType.FORM,
            title="Standard Product Quote",
            content="Please provide details for your quote request.",
            form_fields=[
                FormField("name", "Full Name", FormFieldType.TEXT, required=True),
                FormField("email", "Email Address", FormFieldType.EMAIL, required=True),
                FormField("phone", "Phone Number", FormFieldType.PHONE, required=True),
                FormField("company", "Company Name", FormFieldType.TEXT, required=True),
                FormField("products", "Products & Quantities", FormFieldType.TEXTAREA,
                         required=True, placeholder="List products and quantities needed"),
                FormField("delivery_date", "Requested Delivery Date", FormFieldType.DATE,
                         required=False),
                FormField("additional_notes", "Additional Notes", FormFieldType.TEXTAREA,
                         required=False),
            ],
            options=[
                {"label": "Submit Quote Request", "next_node_id": "quote_submitted"},
                {"label": "Cancel", "next_node_id": "quote_main"},
            ],
            action_type="create_quote"
        )
        
        self.nodes["quote_custom_form"] = TreeNode(
            node_id="quote_custom_form",
            node_type=NodeType.FORM,
            title="Custom Solution Quote",
            content="Tell us about your custom requirements.",
            form_fields=[
                FormField("name", "Full Name", FormFieldType.TEXT, required=True),
                FormField("email", "Email Address", FormFieldType.EMAIL, required=True),
                FormField("phone", "Phone Number", FormFieldType.PHONE, required=True),
                FormField("company", "Company Name", FormFieldType.TEXT, required=True),
                FormField("industry", "Industry", FormFieldType.SELECT, options=[
                    "Manufacturing", "Automotive", "Aerospace", "Medical",
                    "Construction", "Energy", "Other"
                ]),
                FormField("project_description", "Project Description", FormFieldType.TEXTAREA,
                         required=True, placeholder="Describe your project requirements"),
                FormField("budget_range", "Budget Range", FormFieldType.SELECT, options=[
                    "Under $10,000", "$10,000 - $50,000", "$50,000 - $100,000",
                    "$100,000 - $500,000", "Over $500,000"
                ]),
                FormField("timeline", "Project Timeline", FormFieldType.SELECT, options=[
                    "Urgent (< 1 month)", "1-3 months", "3-6 months", "6-12 months", "Flexible"
                ]),
                FormField("attachments", "Upload Specifications (optional)", FormFieldType.FILE,
                         required=False),
            ],
            options=[
                {"label": "Submit Request", "next_node_id": "quote_submitted"},
                {"label": "Cancel", "next_node_id": "quote_main"},
            ],
            action_type="create_custom_quote"
        )
        
        self.nodes["quote_bulk_form"] = TreeNode(
            node_id="quote_bulk_form",
            node_type=NodeType.FORM,
            title="Bulk Order Quote",
            content="Request pricing for volume orders.",
            form_fields=[
                FormField("name", "Full Name", FormFieldType.TEXT, required=True),
                FormField("email", "Email Address", FormFieldType.EMAIL, required=True),
                FormField("phone", "Phone Number", FormFieldType.PHONE, required=True),
                FormField("company", "Company Name", FormFieldType.TEXT, required=True),
                FormField("products", "Products", FormFieldType.TEXTAREA, required=True),
                FormField("quantity", "Estimated Quantity", FormFieldType.NUMBER, required=True),
                FormField("frequency", "Order Frequency", FormFieldType.SELECT, options=[
                    "One-time order", "Monthly", "Quarterly", "Annual contract"
                ]),
            ],
            options=[
                {"label": "Submit", "next_node_id": "quote_submitted"},
                {"label": "Cancel", "next_node_id": "quote_main"},
            ],
            action_type="create_bulk_quote"
        )
        
        self.nodes["quote_submitted"] = TreeNode(
            node_id="quote_submitted",
            node_type=NodeType.TERMINAL,
            title="Quote Request Received",
            content="""Thank you for your quote request!

**What happens next:**
1. Our sales team will review your request within 24 hours
2. You'll receive a detailed quote via email
3. A sales representative may contact you for clarification

**Quote Reference:** Your reference number will be sent to your email.

Questions? Call us at 1-800-JD-JONES""",
            options=[
                {"label": "Return to Main Menu", "next_node_id": "root"},
            ]
        )
        
        # ==========================================
        # ORDER TRACKING BRANCH
        # ==========================================
        self.nodes["order_tracking"] = TreeNode(
            node_id="order_tracking",
            node_type=NodeType.FORM,
            title="Track Your Order",
            content="Enter your order information to check status.",
            form_fields=[
                FormField("order_number", "Order Number", FormFieldType.TEXT, required=True,
                         placeholder="e.g., ORD-12345"),
                FormField("email", "Email Address", FormFieldType.EMAIL, required=True,
                         help_text="Email used when placing the order"),
            ],
            options=[
                {"label": "Track Order", "next_node_id": "order_status_result"},
                {"label": "Back to Main Menu", "next_node_id": "root"},
            ],
            action_type="track_order"
        )
        
        self.nodes["order_status_result"] = TreeNode(
            node_id="order_status_result",
            node_type=NodeType.INFORMATION,
            title="Order Status",
            content="[Order status will be displayed here based on lookup]",
            options=[
                {"label": "Track Another Order", "next_node_id": "order_tracking"},
                {"label": "Contact Support", "next_node_id": "support_main"},
                {"label": "Return to Main Menu", "next_node_id": "root"},
            ],
            metadata={"dynamic_content": True}
        )
        
        # ==========================================
        # TECHNICAL SUPPORT BRANCH
        # ==========================================
        self.nodes["support_main"] = TreeNode(
            node_id="support_main",
            node_type=NodeType.QUESTION,
            title="Technical Support",
            content="What type of support do you need?",
            options=[
                {"label": "Product Issue", "next_node_id": "support_product_issue"},
                {"label": "Installation Help", "next_node_id": "support_installation"},
                {"label": "Documentation", "next_node_id": "support_documentation"},
                {"label": "Schedule Service Call", "next_node_id": "support_service_form"},
                {"label": "Back to Main Menu", "next_node_id": "root"},
            ]
        )
        
        self.nodes["support_product_issue"] = TreeNode(
            node_id="support_product_issue",
            node_type=NodeType.QUESTION,
            title="Product Issue",
            content="Can you describe the issue?",
            options=[
                {"label": "Product Not Working", "next_node_id": "support_troubleshoot"},
                {"label": "Damaged Product", "next_node_id": "returns_damaged"},
                {"label": "Missing Parts", "next_node_id": "support_missing_parts"},
                {"label": "Other Issue", "next_node_id": "support_ticket_form"},
            ]
        )
        
        self.nodes["support_troubleshoot"] = TreeNode(
            node_id="support_troubleshoot",
            node_type=NodeType.INFORMATION,
            title="Troubleshooting Steps",
            content="""Please try these common troubleshooting steps:

1. **Check Power Connection** - Ensure all power connections are secure
2. **Review User Manual** - Check for setup instructions
3. **Reset Device** - If applicable, perform a factory reset
4. **Check for Updates** - Ensure firmware/software is current

If the issue persists after these steps, please submit a support ticket.""",
            options=[
                {"label": "Issue Resolved", "next_node_id": "support_resolved"},
                {"label": "Still Need Help", "next_node_id": "support_ticket_form"},
                {"label": "Back", "next_node_id": "support_main"},
            ]
        )
        
        self.nodes["support_ticket_form"] = TreeNode(
            node_id="support_ticket_form",
            node_type=NodeType.FORM,
            title="Submit Support Ticket",
            content="Please provide details about your issue.",
            form_fields=[
                FormField("name", "Full Name", FormFieldType.TEXT, required=True),
                FormField("email", "Email Address", FormFieldType.EMAIL, required=True),
                FormField("phone", "Phone Number", FormFieldType.PHONE, required=True),
                FormField("order_number", "Order Number (if applicable)", FormFieldType.TEXT,
                         required=False),
                FormField("product", "Product Name/Model", FormFieldType.TEXT, required=True),
                FormField("issue_description", "Describe the Issue", FormFieldType.TEXTAREA,
                         required=True, placeholder="Please describe the issue in detail"),
                FormField("urgency", "Urgency Level", FormFieldType.SELECT, options=[
                    "Low - General question", "Medium - Affecting work",
                    "High - Production stopped", "Critical - Safety concern"
                ]),
            ],
            options=[
                {"label": "Submit Ticket", "next_node_id": "support_ticket_submitted"},
                {"label": "Cancel", "next_node_id": "support_main"},
            ],
            action_type="create_support_ticket"
        )
        
        self.nodes["support_ticket_submitted"] = TreeNode(
            node_id="support_ticket_submitted",
            node_type=NodeType.TERMINAL,
            title="Support Ticket Created",
            content="""Your support ticket has been submitted.

**Response Times:**
- Critical: Within 2 hours
- High: Within 4 hours
- Medium: Within 1 business day
- Low: Within 2 business days

A support representative will contact you soon.""",
            options=[
                {"label": "Return to Main Menu", "next_node_id": "root"},
            ]
        )
        
        # ==========================================
        # RETURNS & WARRANTY BRANCH
        # ==========================================
        self.nodes["returns_main"] = TreeNode(
            node_id="returns_main",
            node_type=NodeType.QUESTION,
            title="Returns & Warranty",
            content="What would you like help with?",
            options=[
                {"label": "Start a Return", "next_node_id": "returns_start"},
                {"label": "Warranty Claim", "next_node_id": "warranty_claim"},
                {"label": "Check Return Policy", "next_node_id": "returns_policy"},
                {"label": "Back to Main Menu", "next_node_id": "root"},
            ]
        )
        
        self.nodes["returns_policy"] = TreeNode(
            node_id="returns_policy",
            node_type=NodeType.INFORMATION,
            title="Return Policy",
            content="""**JD Jones Return Policy**

**Standard Returns:**
- Returns accepted within 30 days of delivery
- Products must be unused and in original packaging
- 15% restocking fee may apply

**Defective Products:**
- Report within 7 days of receiving
- No restocking fee
- We cover return shipping

**Custom Orders:**
- Custom products are non-returnable unless defective

**Process:**
1. Submit return request
2. Receive RMA number
3. Ship product back
4. Refund processed within 5-7 business days""",
            options=[
                {"label": "Start a Return", "next_node_id": "returns_start"},
                {"label": "Back", "next_node_id": "returns_main"},
            ]
        )
        
        self.nodes["returns_start"] = TreeNode(
            node_id="returns_start",
            node_type=NodeType.FORM,
            title="Return Request",
            content="Please provide your order details to initiate a return.",
            form_fields=[
                FormField("order_number", "Order Number", FormFieldType.TEXT, required=True),
                FormField("email", "Email Address", FormFieldType.EMAIL, required=True),
                FormField("reason", "Reason for Return", FormFieldType.SELECT, required=True,
                         options=[
                             "Changed my mind", "Wrong item ordered", "Defective product",
                             "Damaged in shipping", "Not as described", "Other"
                         ]),
                FormField("description", "Additional Details", FormFieldType.TEXTAREA,
                         required=False),
                FormField("condition", "Product Condition", FormFieldType.SELECT, required=True,
                         options=["Unopened", "Opened - Unused", "Used"]),
            ],
            options=[
                {"label": "Submit Return Request", "next_node_id": "returns_submitted"},
                {"label": "Cancel", "next_node_id": "returns_main"},
            ],
            action_type="create_return"
        )
        
        self.nodes["returns_submitted"] = TreeNode(
            node_id="returns_submitted",
            node_type=NodeType.TERMINAL,
            title="Return Request Submitted",
            content="""Your return request has been submitted.

**Next Steps:**
1. You'll receive an RMA number via email within 24 hours
2. Print the return label (included in email)
3. Package the item securely
4. Drop off at any UPS location

Questions? Contact returns@jdjones.com""",
            options=[
                {"label": "Return to Main Menu", "next_node_id": "root"},
            ]
        )
        
        # ==========================================
        # CONTACT BRANCH
        # ==========================================
        self.nodes["contact_main"] = TreeNode(
            node_id="contact_main",
            node_type=NodeType.QUESTION,
            title="Contact Us",
            content="How would you like to reach us?",
            options=[
                {"label": "Request Call Back", "next_node_id": "contact_callback_form"},
                {"label": "Send Message", "next_node_id": "contact_message_form"},
                {"label": "View Contact Information", "next_node_id": "contact_info"},
                {"label": "Back to Main Menu", "next_node_id": "root"},
            ]
        )
        
        self.nodes["contact_info"] = TreeNode(
            node_id="contact_info",
            node_type=NodeType.INFORMATION,
            title="Contact Information",
            content="""**JD Jones Manufacturing**

**Phone:** 1-800-JD-JONES (1-800-535-6637)
**Email:** info@jdjones.com
**Fax:** 1-800-535-6638

**Hours of Operation:**
Monday - Friday: 8:00 AM - 6:00 PM EST
Saturday: 9:00 AM - 1:00 PM EST
Sunday: Closed

**Headquarters:**
123 Industrial Parkway
Manufacturing City, MC 12345

**Regional Offices:**
- West Coast: Los Angeles, CA
- Midwest: Chicago, IL  
- Southeast: Atlanta, GA""",
            options=[
                {"label": "Request Call Back", "next_node_id": "contact_callback_form"},
                {"label": "Send Message", "next_node_id": "contact_message_form"},
                {"label": "Back", "next_node_id": "contact_main"},
            ]
        )
        
        self.nodes["contact_callback_form"] = TreeNode(
            node_id="contact_callback_form",
            node_type=NodeType.FORM,
            title="Request Call Back",
            content="We'll call you at your preferred time.",
            form_fields=[
                FormField("name", "Full Name", FormFieldType.TEXT, required=True),
                FormField("phone", "Phone Number", FormFieldType.PHONE, required=True),
                FormField("email", "Email Address", FormFieldType.EMAIL, required=True),
                FormField("preferred_time", "Preferred Time", FormFieldType.SELECT, options=[
                    "Morning (8AM-12PM)", "Afternoon (12PM-4PM)", "Evening (4PM-6PM)"
                ]),
                FormField("topic", "Topic", FormFieldType.SELECT, options=[
                    "Sales Inquiry", "Technical Support", "Billing Question",
                    "Partnership Opportunity", "Other"
                ]),
                FormField("message", "Brief Message", FormFieldType.TEXTAREA, required=False),
            ],
            options=[
                {"label": "Request Call", "next_node_id": "contact_submitted"},
                {"label": "Cancel", "next_node_id": "contact_main"},
            ],
            action_type="request_callback"
        )
        
        self.nodes["contact_message_form"] = TreeNode(
            node_id="contact_message_form",
            node_type=NodeType.FORM,
            title="Send Us a Message",
            content="We'll respond within 1 business day.",
            form_fields=[
                FormField("name", "Full Name", FormFieldType.TEXT, required=True),
                FormField("email", "Email Address", FormFieldType.EMAIL, required=True),
                FormField("subject", "Subject", FormFieldType.TEXT, required=True),
                FormField("message", "Your Message", FormFieldType.TEXTAREA, required=True),
            ],
            options=[
                {"label": "Send Message", "next_node_id": "contact_submitted"},
                {"label": "Cancel", "next_node_id": "contact_main"},
            ],
            action_type="send_message"
        )
        
        self.nodes["contact_submitted"] = TreeNode(
            node_id="contact_submitted",
            node_type=NodeType.TERMINAL,
            title="Message Received",
            content="""Thank you for contacting JD Jones!

We've received your message and will respond within 1 business day.

For urgent matters, please call 1-800-JD-JONES.""",
            options=[
                {"label": "Return to Main Menu", "next_node_id": "root"},
            ]
        )
        
        # ==========================================
        # UTILITY NODES
        # ==========================================
        self.nodes["form_submitted"] = TreeNode(
            node_id="form_submitted",
            node_type=NodeType.TERMINAL,
            title="Request Submitted",
            content="Thank you! Your request has been received. We'll be in touch soon.",
            options=[
                {"label": "Return to Main Menu", "next_node_id": "root"},
            ]
        )
        
        self.nodes["support_resolved"] = TreeNode(
            node_id="support_resolved",
            node_type=NodeType.TERMINAL,
            title="Glad We Could Help!",
            content="Great! We're happy your issue is resolved. Is there anything else we can help with?",
            options=[
                {"label": "Yes, I have another question", "next_node_id": "root"},
                {"label": "No, thank you", "next_node_id": "goodbye"},
            ]
        )
        
        self.nodes["goodbye"] = TreeNode(
            node_id="goodbye",
            node_type=NodeType.TERMINAL,
            title="Thank You!",
            content="Thank you for visiting JD Jones. Have a great day!",
            options=[
                {"label": "Start Over", "next_node_id": "root"},
            ]
        )
    
    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            TreeNode or None
        """
        return self.nodes.get(node_id)
    
    def get_root(self) -> TreeNode:
        """Get the root node."""
        return self.nodes["root"]
    
    def navigate(self, current_node_id: str, option_index: int) -> Optional[TreeNode]:
        """
        Navigate to next node based on selected option.
        
        Args:
            current_node_id: Current node ID
            option_index: Index of selected option
            
        Returns:
            Next TreeNode or None
        """
        current_node = self.get_node(current_node_id)
        if not current_node:
            return None
        
        if option_index < 0 or option_index >= len(current_node.options):
            return None
        
        next_node_id = current_node.options[option_index].get("next_node_id")
        return self.get_node(next_node_id)
    
    def get_node_for_intent(self, intent: CustomerIntent) -> TreeNode:
        """
        Get the appropriate starting node for a classified intent.
        
        Args:
            intent: Classified customer intent
            
        Returns:
            Appropriate TreeNode
        """
        intent_to_node = {
            CustomerIntent.PRODUCT_INFO: "product_main",
            CustomerIntent.PRICING_QUOTE: "quote_main",
            CustomerIntent.ORDER_STATUS: "order_tracking",
            CustomerIntent.TECHNICAL_SUPPORT: "support_main",
            CustomerIntent.RETURNS_WARRANTY: "returns_main",
            CustomerIntent.GENERAL_INQUIRY: "root",
            CustomerIntent.COMPLAINT: "support_ticket_form",
            CustomerIntent.PARTNERSHIP: "contact_main",
            CustomerIntent.CONTACT_SALES: "contact_callback_form",
            CustomerIntent.UNKNOWN: "root",
        }
        
        node_id = intent_to_node.get(intent, "root")
        return self.nodes[node_id]
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """Get the complete tree structure."""
        return {
            node_id: node.to_dict()
            for node_id, node in self.nodes.items()
        }

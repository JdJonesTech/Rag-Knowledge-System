"""
Approval Manager
Handles human-in-the-loop approval workflows for sensitive actions.

Required for:
- Legal summaries
- Financial approvals
- Email sending
- Database modifications
- Customer communications
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ActionType(str, Enum):
    """Types of actions requiring approval."""
    SEND_EMAIL = "send_email"
    MODIFY_DATABASE = "modify_database"
    FINANCIAL_TRANSACTION = "financial_transaction"
    LEGAL_DOCUMENT = "legal_document"
    CUSTOMER_COMMUNICATION = "customer_communication"
    PRICE_OVERRIDE = "price_override"
    DATA_EXPORT = "data_export"
    SYSTEM_CONFIGURATION = "system_configuration"


class ApprovalLevel(str, Enum):
    """Approval authority levels."""
    STANDARD = "standard"      # Any authorized user
    SUPERVISOR = "supervisor"  # Supervisor or above
    MANAGER = "manager"        # Manager level
    EXECUTIVE = "executive"    # Executive approval needed


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    request_id: str
    action_type: ActionType
    action_description: str
    action_payload: Dict[str, Any]
    approval_level: ApprovalLevel
    requestor_id: str
    requestor_context: Dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_notes: str = ""
    callback_id: Optional[str] = None
    priority: str = "normal"
    
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action_type": self.action_type.value,
            "action_description": self.action_description,
            "action_payload": self.action_payload,
            "approval_level": self.approval_level.value,
            "requestor_id": self.requestor_id,
            "requestor_context": self.requestor_context,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_notes": self.review_notes,
            "priority": self.priority
        }


@dataclass
class ApprovalPolicy:
    """Policy defining when approval is required."""
    action_type: ActionType
    approval_level: ApprovalLevel
    conditions: Dict[str, Any] = field(default_factory=dict)
    expiry_hours: int = 24
    notify_on_create: bool = True
    auto_reject_on_expiry: bool = True


class ApprovalManager:
    """
    Manages human approval workflows.
    
    Features:
    - Configurable approval policies
    - Multi-level approval chains
    - Automatic expiration
    - Notification integration
    - Audit logging
    """
    
    # Default policies for sensitive actions
    DEFAULT_POLICIES = {
        ActionType.SEND_EMAIL: ApprovalPolicy(
            action_type=ActionType.SEND_EMAIL,
            approval_level=ApprovalLevel.STANDARD,
            conditions={"external_recipient": True},
            expiry_hours=4
        ),
        ActionType.FINANCIAL_TRANSACTION: ApprovalPolicy(
            action_type=ActionType.FINANCIAL_TRANSACTION,
            approval_level=ApprovalLevel.MANAGER,
            conditions={"amount_threshold": 10000},
            expiry_hours=48
        ),
        ActionType.LEGAL_DOCUMENT: ApprovalPolicy(
            action_type=ActionType.LEGAL_DOCUMENT,
            approval_level=ApprovalLevel.MANAGER,
            expiry_hours=72
        ),
        ActionType.MODIFY_DATABASE: ApprovalPolicy(
            action_type=ActionType.MODIFY_DATABASE,
            approval_level=ApprovalLevel.SUPERVISOR,
            conditions={"table_whitelist": ["orders", "customers"]},
            expiry_hours=24
        ),
        ActionType.PRICE_OVERRIDE: ApprovalPolicy(
            action_type=ActionType.PRICE_OVERRIDE,
            approval_level=ApprovalLevel.MANAGER,
            conditions={"discount_threshold": 15},
            expiry_hours=24
        ),
        ActionType.DATA_EXPORT: ApprovalPolicy(
            action_type=ActionType.DATA_EXPORT,
            approval_level=ApprovalLevel.SUPERVISOR,
            conditions={"pii_included": True},
            expiry_hours=24
        )
    }
    
    def __init__(
        self,
        policies: Optional[Dict[ActionType, ApprovalPolicy]] = None,
        notification_handler: Optional[Callable] = None,
        audit_handler: Optional[Callable] = None
    ):
        """
        Initialize approval manager.
        
        Args:
            policies: Custom approval policies
            notification_handler: Function to send notifications
            audit_handler: Function to log audit events
        """
        self.policies = {**self.DEFAULT_POLICIES, **(policies or {})}
        self.notification_handler = notification_handler
        self.audit_handler = audit_handler
        
        # Approval request storage
        self.requests: Dict[str, ApprovalRequest] = {}
        
        # Callbacks for when approvals are processed
        self.approval_callbacks: Dict[str, Callable] = {}
        
        # Pending futures for async waiting
        self.pending_futures: Dict[str, asyncio.Future] = {}
    
    def requires_approval(
        self,
        action_type: ActionType,
        action_context: Dict[str, Any]
    ) -> tuple[bool, Optional[ApprovalLevel]]:
        """
        Check if an action requires approval.
        
        Args:
            action_type: Type of action
            action_context: Context/parameters of the action
            
        Returns:
            Tuple of (requires_approval, approval_level)
        """
        if action_type not in self.policies:
            return False, None
        
        policy = self.policies[action_type]
        
        # Check conditions
        for condition_key, condition_value in policy.conditions.items():
            if condition_key == "amount_threshold":
                amount = action_context.get("amount", 0)
                if amount < condition_value:
                    return False, None
            
            elif condition_key == "discount_threshold":
                discount = action_context.get("discount_percent", 0)
                if discount < condition_value:
                    return False, None
            
            elif condition_key == "external_recipient":
                if condition_value and not action_context.get("is_external", False):
                    return False, None
            
            elif condition_key == "pii_included":
                if condition_value and not action_context.get("contains_pii", False):
                    return False, None
            
            elif condition_key == "table_whitelist":
                table = action_context.get("table")
                if table and table in condition_value:
                    return False, None
        
        return True, policy.approval_level
    
    async def request_approval(
        self,
        action_type: ActionType,
        action_description: str,
        action_payload: Dict[str, Any],
        requestor_id: str,
        requestor_context: Optional[Dict[str, Any]] = None,
        priority: str = "normal",
        callback: Optional[Callable] = None
    ) -> ApprovalRequest:
        """
        Create an approval request.
        
        Args:
            action_type: Type of action
            action_description: Human-readable description
            action_payload: Data for the action
            requestor_id: Who is requesting
            requestor_context: Additional context
            priority: Request priority
            callback: Function to call when approved/rejected
            
        Returns:
            ApprovalRequest object
        """
        policy = self.policies.get(action_type)
        expiry_hours = policy.expiry_hours if policy else 24
        approval_level = policy.approval_level if policy else ApprovalLevel.STANDARD
        
        request_id = f"apr_{uuid.uuid4().hex[:12]}"
        
        request = ApprovalRequest(
            request_id=request_id,
            action_type=action_type,
            action_description=action_description,
            action_payload=action_payload,
            approval_level=approval_level,
            requestor_id=requestor_id,
            requestor_context=requestor_context or {},
            expires_at=datetime.now() + timedelta(hours=expiry_hours),
            priority=priority
        )
        
        self.requests[request_id] = request
        
        # Store callback
        if callback:
            self.approval_callbacks[request_id] = callback
        
        # Send notification
        if self.notification_handler and (not policy or policy.notify_on_create):
            await self._send_notification(request, "created")
        
        # Log audit event
        if self.audit_handler:
            await self._log_audit(request, "created")
        
        return request
    
    async def approve(
        self,
        request_id: str,
        reviewer_id: str,
        notes: str = ""
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Approve a request.
        
        Args:
            request_id: Request to approve
            reviewer_id: Who is approving
            notes: Review notes
            
        Returns:
            Tuple of (success, action_result)
        """
        request = self.requests.get(request_id)
        if not request:
            return False, {"error": "Request not found"}
        
        if request.status != ApprovalStatus.PENDING:
            return False, {"error": f"Request is {request.status.value}"}
        
        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            return False, {"error": "Request has expired"}
        
        # Update request
        request.status = ApprovalStatus.APPROVED
        request.reviewed_by = reviewer_id
        request.reviewed_at = datetime.now()
        request.review_notes = notes
        
        # Execute callback
        result = None
        if request_id in self.approval_callbacks:
            try:
                callback = self.approval_callbacks[request_id]
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(request.action_payload)
                else:
                    result = callback(request.action_payload)
            except Exception as e:
                result = {"error": str(e)}
        
        # Resolve pending future
        if request_id in self.pending_futures:
            self.pending_futures[request_id].set_result(("approved", result))
        
        # Notifications and audit
        if self.notification_handler:
            await self._send_notification(request, "approved")
        if self.audit_handler:
            await self._log_audit(request, "approved", reviewer_id)
        
        return True, result
    
    async def reject(
        self,
        request_id: str,
        reviewer_id: str,
        reason: str = ""
    ) -> bool:
        """
        Reject a request.
        
        Args:
            request_id: Request to reject
            reviewer_id: Who is rejecting
            reason: Rejection reason
            
        Returns:
            Success status
        """
        request = self.requests.get(request_id)
        if not request:
            return False
        
        if request.status != ApprovalStatus.PENDING:
            return False
        
        # Update request
        request.status = ApprovalStatus.REJECTED
        request.reviewed_by = reviewer_id
        request.reviewed_at = datetime.now()
        request.review_notes = reason
        
        # Resolve pending future
        if request_id in self.pending_futures:
            self.pending_futures[request_id].set_result(("rejected", None))
        
        # Notifications and audit
        if self.notification_handler:
            await self._send_notification(request, "rejected")
        if self.audit_handler:
            await self._log_audit(request, "rejected", reviewer_id)
        
        return True
    
    async def wait_for_approval(
        self,
        request_id: str,
        timeout_seconds: int = 3600
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Wait for an approval decision.
        
        Args:
            request_id: Request to wait for
            timeout_seconds: Maximum wait time
            
        Returns:
            Tuple of (status, result)
        """
        if request_id not in self.requests:
            return "not_found", None
        
        request = self.requests[request_id]
        
        # Already resolved
        if request.status != ApprovalStatus.PENDING:
            return request.status.value, None
        
        # Create future to wait on
        future = asyncio.get_event_loop().create_future()
        self.pending_futures[request_id] = future
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
            return result
        except asyncio.TimeoutError:
            # Check if expired
            if request.is_expired():
                request.status = ApprovalStatus.EXPIRED
                return "expired", None
            return "timeout", None
        finally:
            if request_id in self.pending_futures:
                del self.pending_futures[request_id]
    
    def get_pending_requests(
        self,
        approval_level: Optional[ApprovalLevel] = None,
        action_type: Optional[ActionType] = None
    ) -> List[ApprovalRequest]:
        """Get pending approval requests."""
        pending = [
            r for r in self.requests.values()
            if r.status == ApprovalStatus.PENDING and not r.is_expired()
        ]
        
        if approval_level:
            pending = [r for r in pending if r.approval_level == approval_level]
        
        if action_type:
            pending = [r for r in pending if r.action_type == action_type]
        
        return sorted(pending, key=lambda r: r.created_at)
    
    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a specific request."""
        return self.requests.get(request_id)
    
    async def cleanup_expired(self) -> int:
        """Clean up expired requests."""
        count = 0
        for request in self.requests.values():
            if request.status == ApprovalStatus.PENDING and request.is_expired():
                policy = self.policies.get(request.action_type)
                if policy and policy.auto_reject_on_expiry:
                    request.status = ApprovalStatus.EXPIRED
                    
                    if request.request_id in self.pending_futures:
                        self.pending_futures[request.request_id].set_result(("expired", None))
                    
                    if self.audit_handler:
                        await self._log_audit(request, "expired")
                    
                    count += 1
        
        return count
    
    async def _send_notification(self, request: ApprovalRequest, event: str):
        """Send notification for approval event."""
        if self.notification_handler:
            try:
                if asyncio.iscoroutinefunction(self.notification_handler):
                    await self.notification_handler(request, event)
                else:
                    self.notification_handler(request, event)
            except Exception as e:
                logger.error(f"Notification error: {e}")
    
    async def _log_audit(
        self,
        request: ApprovalRequest,
        event: str,
        reviewer_id: Optional[str] = None
    ):
        """Log audit event."""
        if self.audit_handler:
            try:
                audit_entry = {
                    "request_id": request.request_id,
                    "event": event,
                    "action_type": request.action_type.value,
                    "requestor_id": request.requestor_id,
                    "reviewer_id": reviewer_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                if asyncio.iscoroutinefunction(self.audit_handler):
                    await self.audit_handler(audit_entry)
                else:
                    self.audit_handler(audit_entry)
            except Exception as e:
                logger.error(f"Audit error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get approval statistics."""
        all_requests = list(self.requests.values())
        
        return {
            "total_requests": len(all_requests),
            "pending": sum(1 for r in all_requests if r.status == ApprovalStatus.PENDING),
            "approved": sum(1 for r in all_requests if r.status == ApprovalStatus.APPROVED),
            "rejected": sum(1 for r in all_requests if r.status == ApprovalStatus.REJECTED),
            "expired": sum(1 for r in all_requests if r.status == ApprovalStatus.EXPIRED),
            "by_type": {
                t.value: sum(1 for r in all_requests if r.action_type == t)
                for t in ActionType
            }
        }

"""
Agent Registry
Central registry for managing agent configurations and instances.
"""

from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    name: str
    agent_type: str
    role: str
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: int = 4096
    tools: List[str] = field(default_factory=list)
    system_prompt: str = ""
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type,
            "role": self.role,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": self.tools,
            "system_prompt": self.system_prompt[:200] + "..." if len(self.system_prompt) > 200 else self.system_prompt,
            "enabled": self.enabled,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class AgentStats:
    """Statistics for an agent."""
    agent_id: str
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_tokens_used: int = 0
    avg_latency_ms: float = 0
    last_invoked: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "total_invocations": self.total_invocations,
            "successful_invocations": self.successful_invocations,
            "failed_invocations": self.failed_invocations,
            "success_rate": self.successful_invocations / self.total_invocations if self.total_invocations > 0 else 0,
            "total_tokens_used": self.total_tokens_used,
            "avg_latency_ms": self.avg_latency_ms,
            "last_invoked": self.last_invoked.isoformat() if self.last_invoked else None
        }


class AgentRegistry:
    """
    Central registry for agent management.
    
    Features:
    - Agent configuration storage
    - Instance management
    - Statistics tracking
    - Health monitoring
    """
    
    # Default agent configurations
    DEFAULT_AGENTS = {
        "router": AgentConfig(
            agent_id="agent_router",
            name="Query Router",
            agent_type="router",
            role="router",
            model="gpt-4-turbo-preview",
            temperature=0,
            tools=["vector_search"],
            system_prompt="You are a query router. Analyze queries and determine the best agent/pipeline to handle them."
        ),
        "researcher": AgentConfig(
            agent_id="agent_researcher",
            name="Research Agent",
            agent_type="react",
            role="researcher",
            model="gpt-4-turbo-preview",
            temperature=0.1,
            tools=["vector_search", "product_database", "external_api"],
            system_prompt="You are a research agent. Gather comprehensive information to answer queries."
        ),
        "writer": AgentConfig(
            agent_id="agent_writer",
            name="Writer Agent",
            agent_type="react",
            role="writer",
            model="gpt-4-turbo-preview",
            temperature=0.3,
            tools=["document_generator"],
            system_prompt="You are a writing agent. Synthesize information into clear, professional responses."
        ),
        "reviewer": AgentConfig(
            agent_id="agent_reviewer",
            name="Reviewer Agent",
            agent_type="validation",
            role="reviewer",
            model="gpt-4-turbo-preview",
            temperature=0,
            tools=["compliance_checker", "vector_search"],
            system_prompt="You are a review agent. Validate information accuracy and compliance."
        ),
        "executor": AgentConfig(
            agent_id="agent_executor",
            name="Executor Agent",
            agent_type="react",
            role="executor",
            model="gpt-4-turbo-preview",
            temperature=0,
            tools=["erp_query", "email_router", "crm"],
            system_prompt="You are an executor agent. Take actions like sending emails, updating CRM, querying databases."
        ),
        "product_selector": AgentConfig(
            agent_id="agent_product_selector",
            name="Product Selection Agent",
            agent_type="guided",
            role="specialist",
            model="gpt-4-turbo-preview",
            temperature=0.1,
            tools=["product_database", "compliance_checker"],
            system_prompt="You are a product selection specialist. Guide users through product selection with targeted questions."
        ),
        "enquiry_manager": AgentConfig(
            agent_id="agent_enquiry_manager",
            name="Enquiry Management Agent",
            agent_type="classifier",
            role="specialist",
            model="gpt-4-turbo-preview",
            temperature=0,
            tools=["email_router", "crm", "vector_search"],
            system_prompt="You manage incoming enquiries - classify, route, and respond appropriately."
        )
    }
    
    def __init__(self):
        """Initialize registry."""
        self.configs: Dict[str, AgentConfig] = {}
        self.instances: Dict[str, Any] = {}
        self.stats: Dict[str, AgentStats] = {}
        
        # Load default agents
        for key, config in self.DEFAULT_AGENTS.items():
            self.register(config)
    
    def register(self, config: AgentConfig) -> bool:
        """
        Register an agent configuration.
        
        Args:
            config: Agent configuration
            
        Returns:
            Success status
        """
        self.configs[config.agent_id] = config
        self.stats[config.agent_id] = AgentStats(agent_id=config.agent_id)
        return True
    
    def unregister(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id in self.configs:
            del self.configs[agent_id]
            if agent_id in self.instances:
                del self.instances[agent_id]
            if agent_id in self.stats:
                del self.stats[agent_id]
            return True
        return False
    
    def get_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent configuration."""
        return self.configs.get(agent_id)
    
    def get_by_role(self, role: str) -> List[AgentConfig]:
        """Get all agents with a specific role."""
        return [c for c in self.configs.values() if c.role == role and c.enabled]
    
    def get_by_type(self, agent_type: str) -> List[AgentConfig]:
        """Get all agents of a specific type."""
        return [c for c in self.configs.values() if c.agent_type == agent_type and c.enabled]
    
    def set_instance(self, agent_id: str, instance: Any):
        """Store an agent instance."""
        self.instances[agent_id] = instance
    
    def get_instance(self, agent_id: str) -> Optional[Any]:
        """Get an agent instance."""
        return self.instances.get(agent_id)
    
    def record_invocation(
        self,
        agent_id: str,
        success: bool,
        tokens_used: int = 0,
        latency_ms: float = 0
    ):
        """Record an agent invocation for statistics."""
        if agent_id not in self.stats:
            self.stats[agent_id] = AgentStats(agent_id=agent_id)
        
        stats = self.stats[agent_id]
        stats.total_invocations += 1
        
        if success:
            stats.successful_invocations += 1
        else:
            stats.failed_invocations += 1
        
        stats.total_tokens_used += tokens_used
        stats.last_invoked = datetime.now()
        
        # Update rolling average latency
        if stats.total_invocations == 1:
            stats.avg_latency_ms = latency_ms
        else:
            stats.avg_latency_ms = (
                stats.avg_latency_ms * (stats.total_invocations - 1) + latency_ms
            ) / stats.total_invocations
    
    def get_stats(self, agent_id: str) -> Optional[AgentStats]:
        """Get agent statistics."""
        return self.stats.get(agent_id)
    
    def get_all_stats(self) -> Dict[str, AgentStats]:
        """Get all agent statistics."""
        return self.stats
    
    def enable(self, agent_id: str) -> bool:
        """Enable an agent."""
        if agent_id in self.configs:
            self.configs[agent_id].enabled = True
            return True
        return False
    
    def disable(self, agent_id: str) -> bool:
        """Disable an agent."""
        if agent_id in self.configs:
            self.configs[agent_id].enabled = False
            return True
        return False
    
    def update_config(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update agent configuration."""
        if agent_id not in self.configs:
            return False
        
        config = self.configs[agent_id]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return True
    
    def list_agents(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """List all registered agents."""
        agents = self.configs.values()
        if enabled_only:
            agents = [a for a in agents if a.enabled]
        
        return [
            {
                **a.to_dict(),
                "stats": self.stats[a.agent_id].to_dict() if a.agent_id in self.stats else None
            }
            for a in agents
        ]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all agents."""
        statuses = {}
        
        for agent_id, config in self.configs.items():
            stats = self.stats.get(agent_id)
            
            # Determine health
            health = "healthy"
            if not config.enabled:
                health = "disabled"
            elif stats:
                success_rate = stats.successful_invocations / stats.total_invocations if stats.total_invocations > 0 else 1
                if success_rate < 0.5:
                    health = "unhealthy"
                elif success_rate < 0.9:
                    health = "degraded"
            
            statuses[agent_id] = {
                "name": config.name,
                "role": config.role,
                "health": health,
                "enabled": config.enabled,
                "invocations": stats.total_invocations if stats else 0,
                "success_rate": stats.successful_invocations / stats.total_invocations if stats and stats.total_invocations > 0 else None
            }
        
        return statuses

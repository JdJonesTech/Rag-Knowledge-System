"""
Multi-Agent Coordinator
Coordinates multiple specialized agents for complex workflows.

Implements:
- Agent role assignment
- Task distribution
- Result aggregation
- Inter-agent communication
"""

from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid

from src.agentic.agents.react_agent import ReActAgent, ReActResult
from src.agentic.agents.query_planner import QueryPlannerAgent, ExecutionPlan, SubTask
from src.agentic.agents.validation_agent import ValidationAgent
from src.agentic.observability.tracer import AgentTracer, SpanType


class AgentRole(str, Enum):
    """Roles for specialized agents."""
    ROUTER = "router"           # Routes queries to appropriate pipelines
    PLANNER = "planner"         # Decomposes complex tasks
    RESEARCHER = "researcher"   # Gathers information
    WRITER = "writer"           # Synthesizes responses
    REVIEWER = "reviewer"       # Validates outputs
    EXECUTOR = "executor"       # Takes actions
    SPECIALIST = "specialist"   # Domain-specific agent


class TaskStatus(str, Enum):
    """Status of a distributed task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """A task assigned to an agent."""
    task_id: str
    agent_role: AgentRole
    task_type: str
    input_data: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_role": self.agent_role.value,
            "task_type": self.task_type,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "dependencies": self.dependencies
        }


@dataclass
class WorkflowResult:
    """Result of a multi-agent workflow."""
    workflow_id: str
    success: bool
    final_output: Any
    tasks_completed: int
    tasks_failed: int
    total_time_ms: float
    agent_contributions: Dict[str, Any]
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "success": self.success,
            "final_output": self.final_output,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_time_ms": self.total_time_ms,
            "agent_contributions": self.agent_contributions,
            "trace_id": self.trace_id
        }


class MultiAgentCoordinator:
    """
    Coordinates multiple specialized agents.
    
    Architecture:
    1. Router Agent: Analyzes query and determines which agents needed
    2. Planner Agent: Decomposes into sub-tasks
    3. Worker Agents: Execute tasks in parallel where possible
    4. Reviewer Agent: Validates combined output
    5. Executor Agent: Takes final actions
    """
    
    def __init__(
        self,
        tracer: Optional[AgentTracer] = None,
        max_parallel_tasks: int = 5
    ):
        """
        Initialize coordinator.
        
        Args:
            tracer: Optional tracer for observability
            max_parallel_tasks: Maximum concurrent tasks
        """
        self.tracer = tracer or AgentTracer()
        self.max_parallel_tasks = max_parallel_tasks
        
        # Specialized agents
        self.planner = QueryPlannerAgent()
        self.validator = ValidationAgent()
        
        # Worker agents by role
        self.workers: Dict[AgentRole, ReActAgent] = {}
        
        # Active workflows
        self.workflows: Dict[str, Dict[str, AgentTask]] = {}
        
        # Task queue
        self.task_semaphore = asyncio.Semaphore(max_parallel_tasks)
    
    def register_worker(self, role: AgentRole, agent: ReActAgent):
        """Register a worker agent for a role."""
        self.workers[role] = agent
    
    async def execute_workflow(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> WorkflowResult:
        """
        Execute a multi-agent workflow.
        
        Args:
            query: User query
            context: Additional context
            user_id: User identifier
            
        Returns:
            WorkflowResult with combined output
        """
        import time
        start_time = time.time()
        
        workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
        self.workflows[workflow_id] = {}
        
        # Start trace
        trace = self.tracer.start_trace(
            name=f"workflow_{workflow_id}",
            user_id=user_id
        )
        
        try:
            # Step 1: Plan the workflow
            with self.tracer.span_context("planning", SpanType.REASONING):
                plan = await self.planner.create_plan(query, context)
            
            # Step 2: Create tasks from plan
            tasks = self._create_tasks_from_plan(workflow_id, plan)
            
            # Step 3: Execute tasks respecting dependencies
            results = await self._execute_tasks(workflow_id, tasks)
            
            # Step 4: Aggregate results
            with self.tracer.span_context("aggregation", SpanType.REASONING):
                aggregated = self._aggregate_results(results)
            
            # Step 5: Validate output
            with self.tracer.span_context("validation", SpanType.VALIDATION):
                validation = await self.validator.validate_response(
                    response=str(aggregated.get("final_response", "")),
                    context=context
                )
            
            # Compile final result
            tasks_completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
            tasks_failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
            
            return WorkflowResult(
                workflow_id=workflow_id,
                success=tasks_failed == 0,
                final_output={
                    "response": aggregated.get("final_response"),
                    "sources": aggregated.get("sources", []),
                    "validation": validation.to_dict()
                },
                tasks_completed=tasks_completed,
                tasks_failed=tasks_failed,
                total_time_ms=(time.time() - start_time) * 1000,
                agent_contributions=aggregated.get("contributions", {}),
                trace_id=trace.trace_id
            )
            
        except Exception as e:
            return WorkflowResult(
                workflow_id=workflow_id,
                success=False,
                final_output={"error": str(e)},
                tasks_completed=0,
                tasks_failed=1,
                total_time_ms=(time.time() - start_time) * 1000,
                agent_contributions={},
                trace_id=trace.trace_id if trace else None
            )
        
        finally:
            self.tracer.end_trace()
            # Cleanup
            if workflow_id in self.workflows:
                del self.workflows[workflow_id]
    
    def _create_tasks_from_plan(
        self,
        workflow_id: str,
        plan: ExecutionPlan
    ) -> List[AgentTask]:
        """Create agent tasks from execution plan."""
        tasks = []
        
        for subtask in plan.tasks:
            # Map task type to agent role
            role = self._map_task_to_role(subtask)
            
            task = AgentTask(
                task_id=f"{workflow_id}_{subtask.task_id}",
                agent_role=role,
                task_type=subtask.task_type.value,
                input_data={
                    "description": subtask.description,
                    "tool": subtask.tool_to_use,
                    "parameters": subtask.parameters
                },
                dependencies=[f"{workflow_id}_{d}" for d in subtask.dependencies]
            )
            
            tasks.append(task)
            self.workflows[workflow_id][task.task_id] = task
        
        return tasks
    
    def _map_task_to_role(self, subtask: SubTask) -> AgentRole:
        """Map a subtask to an agent role."""
        task_type = subtask.task_type.value
        
        if task_type == "retrieve":
            return AgentRole.RESEARCHER
        elif task_type in ["query_database", "action"]:
            return AgentRole.EXECUTOR
        elif task_type == "validate":
            return AgentRole.REVIEWER
        elif task_type in ["generate", "aggregate"]:
            return AgentRole.WRITER
        else:
            return AgentRole.SPECIALIST
    
    async def _execute_tasks(
        self,
        workflow_id: str,
        tasks: List[AgentTask]
    ) -> List[AgentTask]:
        """Execute tasks respecting dependencies."""
        # Build dependency graph
        task_map = {t.task_id: t for t in tasks}
        pending = set(t.task_id for t in tasks)
        completed = set()
        
        while pending:
            # Find tasks ready to execute (dependencies met)
            ready = [
                task_map[tid] for tid in pending
                if all(dep in completed for dep in task_map[tid].dependencies)
            ]
            
            if not ready:
                # Circular dependency or stuck
                break
            
            # Execute ready tasks in parallel
            execute_tasks = []
            for task in ready[:self.max_parallel_tasks]:
                execute_tasks.append(self._execute_single_task(task))
            
            await asyncio.gather(*execute_tasks)
            
            # Update sets
            for task in ready[:self.max_parallel_tasks]:
                pending.discard(task.task_id)
                if task.status == TaskStatus.COMPLETED:
                    completed.add(task.task_id)
        
        return tasks
    
    async def _execute_single_task(self, task: AgentTask):
        """Execute a single task."""
        async with self.task_semaphore:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()
            
            try:
                with self.tracer.span_context(
                    f"task_{task.task_type}",
                    SpanType.AGENT,
                    {"task_id": task.task_id, "role": task.agent_role.value}
                ):
                    # Get appropriate worker
                    worker = self.workers.get(task.agent_role)
                    
                    if worker:
                        result = await worker.execute(
                            query=task.input_data.get("description", ""),
                            context=task.input_data
                        )
                        task.result = result.to_dict() if hasattr(result, "to_dict") else result
                    else:
                        # No worker, use default execution
                        task.result = {"status": "executed", "data": task.input_data}
                    
                    task.status = TaskStatus.COMPLETED
                    
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
            
            finally:
                task.completed_at = datetime.now()
    
    def _aggregate_results(self, tasks: List[AgentTask]) -> Dict[str, Any]:
        """Aggregate results from all tasks."""
        contributions = {}
        all_sources = []
        responses = []
        
        for task in tasks:
            if task.status == TaskStatus.COMPLETED and task.result:
                role = task.agent_role.value
                
                if role not in contributions:
                    contributions[role] = []
                contributions[role].append(task.result)
                
                # Extract sources
                if isinstance(task.result, dict):
                    sources = task.result.get("sources", [])
                    all_sources.extend(sources)
                    
                    if "final_answer" in task.result:
                        responses.append(task.result["final_answer"])
                    elif "response" in task.result:
                        responses.append(task.result["response"])
        
        # Combine responses
        final_response = "\n\n".join(responses) if responses else "No results generated."
        
        return {
            "final_response": final_response,
            "sources": all_sources,
            "contributions": contributions
        }
    
    async def execute_simple(
        self,
        query: str,
        agent_role: AgentRole,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a simple single-agent task.
        
        Args:
            query: Query to execute
            agent_role: Which agent to use
            context: Additional context
            
        Returns:
            Agent result
        """
        worker = self.workers.get(agent_role)
        if not worker:
            raise ValueError(f"No worker registered for role: {agent_role}")
        
        with self.tracer.span_context(f"simple_{agent_role.value}", SpanType.AGENT):
            result = await worker.execute(query, context)
        
        return result
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow."""
        if workflow_id not in self.workflows:
            return None
        
        tasks = self.workflows[workflow_id]
        return {
            "workflow_id": workflow_id,
            "total_tasks": len(tasks),
            "pending": sum(1 for t in tasks.values() if t.status == TaskStatus.PENDING),
            "in_progress": sum(1 for t in tasks.values() if t.status == TaskStatus.IN_PROGRESS),
            "completed": sum(1 for t in tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in tasks.values() if t.status == TaskStatus.FAILED),
            "tasks": [t.to_dict() for t in tasks.values()]
        }

"""
Query Planner Agent
Decomposes complex requests into executable sub-tasks.
Handles multi-step queries that require data from multiple sources.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings


class TaskType(str, Enum):
    """Types of sub-tasks."""
    RETRIEVE = "retrieve"           # Fetch data from knowledge base
    QUERY_DATABASE = "query_database"  # Query ERP/SQL
    CALCULATE = "calculate"         # Perform calculations
    COMPARE = "compare"            # Compare data sets
    VALIDATE = "validate"          # Validate against standards
    GENERATE = "generate"          # Generate document/content
    ACTION = "action"              # Take an action (email, update)
    AGGREGATE = "aggregate"        # Combine results


class TaskPriority(str, Enum):
    """Task execution priority."""
    CRITICAL = "critical"    # Must complete for answer
    HIGH = "high"           # Important but not blocking
    MEDIUM = "medium"       # Nice to have
    LOW = "low"             # Optional enhancement


@dataclass
class SubTask:
    """A decomposed sub-task."""
    task_id: str
    task_type: TaskType
    description: str
    tool_to_use: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)  # IDs of tasks this depends on
    priority: TaskPriority = TaskPriority.HIGH
    estimated_complexity: str = "low"  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "description": self.description,
            "tool_to_use": self.tool_to_use,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "priority": self.priority.value,
            "estimated_complexity": self.estimated_complexity
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan for a query."""
    plan_id: str
    original_query: str
    tasks: List[SubTask]
    execution_order: List[str]  # Task IDs in execution order
    parallel_groups: List[List[str]]  # Groups that can run in parallel
    estimated_total_time_ms: int
    complexity_score: float
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "original_query": self.original_query,
            "tasks": [t.to_dict() for t in self.tasks],
            "execution_order": self.execution_order,
            "parallel_groups": self.parallel_groups,
            "estimated_total_time_ms": self.estimated_total_time_ms,
            "complexity_score": self.complexity_score,
            "created_at": self.created_at.isoformat()
        }


class QueryPlannerAgent:
    """
    Decomposes complex queries into executable sub-tasks.
    
    Capabilities:
    - Analyze query to identify required data sources
    - Break down multi-step requests
    - Identify dependencies between tasks
    - Optimize execution order (parallelization)
    - Handle queries like "Compare Q3 sales to last year"
    """
    
    PLANNING_PROMPT = """You are a query planning agent for JD Jones Manufacturing.

Your job is to decompose complex queries into executable sub-tasks.

AVAILABLE TOOLS:
- vector_search: Search knowledge base for documents
- product_database: Query product catalog
- erp_query: Query ERP for stock, orders, pricing
- compliance_checker: Check standards compliance
- document_generator: Generate documents
- email_router: Send/route emails
- crm: CRM operations
- external_api: Exchange rates, shipping, etc.

TASK TYPES:
- retrieve: Fetch data from knowledge base
- query_database: Query ERP/SQL databases
- calculate: Perform calculations
- compare: Compare data sets
- validate: Validate against standards
- generate: Generate document/content
- action: Take an action (email, CRM update)
- aggregate: Combine results from multiple tasks

ANALYSIS:
For the given query, identify:
1. What information is needed?
2. Which tools can provide it?
3. What order should tasks run?
4. Which tasks can run in parallel?
5. What are the dependencies?

QUERY: {query}
CONTEXT: {context}

Respond in JSON format:
{{
    "analysis": "Brief analysis of what's needed",
    "tasks": [
        {{
            "task_id": "task_1",
            "task_type": "retrieve|query_database|calculate|compare|validate|generate|action|aggregate",
            "description": "What this task does",
            "tool_to_use": "tool_name",
            "parameters": {{}},
            "dependencies": [],
            "priority": "critical|high|medium|low",
            "estimated_complexity": "low|medium|high"
        }}
    ],
    "parallel_groups": [["task_1", "task_2"], ["task_3"]],
    "estimated_time_ms": 5000,
    "complexity_score": 0.7
}}
"""

    def __init__(self):
        """Initialize query planner."""
        from src.config.settings import get_llm
        self.llm = get_llm(temperature=0)
    
    async def create_plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan for a query.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            ExecutionPlan with decomposed tasks
        """
        import uuid
        
        prompt = self.PLANNING_PROMPT.format(
            query=query,
            context=json.dumps(context or {}, indent=2)
        )
        
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        
        # Parse response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        plan_data = json.loads(content.strip())
        
        # Build tasks
        tasks = []
        for task_data in plan_data.get("tasks", []):
            tasks.append(SubTask(
                task_id=task_data["task_id"],
                task_type=TaskType(task_data["task_type"]),
                description=task_data["description"],
                tool_to_use=task_data["tool_to_use"],
                parameters=task_data.get("parameters", {}),
                dependencies=task_data.get("dependencies", []),
                priority=TaskPriority(task_data.get("priority", "high")),
                estimated_complexity=task_data.get("estimated_complexity", "medium")
            ))
        
        # Determine execution order
        execution_order = self._compute_execution_order(tasks)
        
        return ExecutionPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            original_query=query,
            tasks=tasks,
            execution_order=execution_order,
            parallel_groups=plan_data.get("parallel_groups", [[t.task_id for t in tasks]]),
            estimated_total_time_ms=plan_data.get("estimated_time_ms", 5000),
            complexity_score=plan_data.get("complexity_score", 0.5)
        )
    
    def _compute_execution_order(self, tasks: List[SubTask]) -> List[str]:
        """Compute execution order based on dependencies (topological sort)."""
        # Build adjacency list
        in_degree = {task.task_id: 0 for task in tasks}
        graph = {task.task_id: [] for task in tasks}
        
        for task in tasks:
            for dep in task.dependencies:
                if dep in graph:
                    graph[dep].append(task.task_id)
                    in_degree[task.task_id] += 1
        
        # Kahn's algorithm
        queue = [tid for tid, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority within current level
            task_map = {t.task_id: t for t in tasks}
            queue.sort(key=lambda x: (
                0 if task_map[x].priority == TaskPriority.CRITICAL else
                1 if task_map[x].priority == TaskPriority.HIGH else
                2 if task_map[x].priority == TaskPriority.MEDIUM else 3
            ))
            
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    async def simplify_query(self, query: str) -> str:
        """
        Simplify/rewrite a query for better retrieval.
        Query enhancement technique.
        """
        prompt = f"""Rewrite this query to be more specific and search-friendly.
Keep the core intent but make it clearer.

Original: {query}

Rewritten (just the query, nothing else):"""
        
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        return response.content.strip()
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into multiple search queries.
        Useful for comprehensive retrieval.
        """
        prompt = f"""Generate 3 alternative search queries that could help answer this question.
Each should approach the topic from a different angle.

Original: {query}

Return as JSON array:
["query1", "query2", "query3"]"""
        
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        
        content = response.content
        if "[" in content:
            start = content.index("[")
            end = content.rindex("]") + 1
            return json.loads(content[start:end])
        
        return [query]

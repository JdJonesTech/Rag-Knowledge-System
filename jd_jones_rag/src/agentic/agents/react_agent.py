"""
ReAct Agent
Implements the Reason + Act loop pattern for iterative problem solving.
The agent reasons through a task, takes action, observes results, and refines approach.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)

from src.config.settings import settings
from src.agentic.tools.base_tool import BaseTool, ToolResult


class ReActStep(str, Enum):
    """Steps in the ReAct loop."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class ReActTrace:
    """A single step in the ReAct trace."""
    step_type: ReActStep
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_type": self.step_type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ReActResult:
    """Result from ReAct execution."""
    success: bool
    final_answer: str
    trace: List[ReActTrace] = field(default_factory=list)
    iterations: int = 0
    total_time_ms: float = 0
    tools_used: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_answer": self.final_answer,
            "trace": [t.to_dict() for t in self.trace],
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "tools_used": self.tools_used,
            "sources": self.sources
        }


class ReActAgent:
    """
    ReAct Agent implementing the Reason + Act loop.
    
    Pattern:
    1. THOUGHT: Reason about what to do next
    2. ACTION: Choose and execute a tool
    3. OBSERVATION: Observe the result
    4. Repeat until FINAL_ANSWER or max iterations
    
    This allows the agent to self-correct if initial retrieval is insufficient.
    """
    
    REACT_SYSTEM_PROMPT = """You are a ReAct agent for JD Jones Manufacturing. You solve problems by reasoning step-by-step and using tools.

AVAILABLE TOOLS:
{tools_description}

RESPONSE FORMAT:
You must respond in this exact format:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"param1": "value1", "param2": "value2"}}

OR if you have enough information:

Thought: [Your final reasoning]
Final Answer: [Your complete answer to the user]

RULES:
1. Always start with a Thought
2. Use tools to gather information - don't guess
3. If a tool returns insufficient info, try a different approach
4. If retrieved documents don't answer the question, say so
5. Maximum {max_iterations} iterations before providing best answer
6. Include sources in your final answer
7. For product recommendations, ALWAYS verify specifications meet requirements

CONTEXT:
User Query: {query}
Additional Context: {context}
"""

    def __init__(
        self,
        tools: Optional[Dict[str, BaseTool]] = None,
        max_iterations: int = 5,
        verbose: bool = False
    ):
        """
        Initialize ReAct agent.
        
        Args:
            tools: Dictionary of available tools
            max_iterations: Maximum reasoning iterations
            verbose: Whether to print trace
        """
        self.tools = tools or {}
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        from src.config.settings import get_llm
        self.llm = get_llm(temperature=0.1)
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool for the agent to use."""
        self.tools[tool.name] = tool
    
    def _get_tools_description(self) -> str:
        """Generate tools description for the prompt."""
        descriptions = []
        for name, tool in self.tools.items():
            schema = tool.get_schema()
            params = ", ".join(schema.get("properties", {}).keys())
            descriptions.append(f"- {name}: {tool.description}\n  Parameters: {params}")
        return "\n".join(descriptions)
    
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReActResult:
        """
        Execute the ReAct loop.
        
        Args:
            query: User query
            context: Additional context (parameters, history, etc.)
            
        Returns:
            ReActResult with final answer and trace
        """
        import time
        start_time = time.time()
        
        trace: List[ReActTrace] = []
        tools_used: List[str] = []
        sources: List[Dict[str, Any]] = []
        
        # Build system prompt
        system_prompt = self.REACT_SYSTEM_PROMPT.format(
            tools_description=self._get_tools_description(),
            max_iterations=self.max_iterations,
            query=query,
            context=json.dumps(context or {}, indent=2)
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please help with: {query}")
        ]
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.info(f"--- Iteration {iteration + 1} ---")
            
            # Get LLM response
            response = await self.llm.ainvoke(messages)
            response_text = response.content
            
            if self.verbose:
                logger.info(f"LLM Response:\n{response_text}")
            
            # Parse response
            parsed = self._parse_response(response_text)
            
            # Record thought
            if parsed.get("thought"):
                trace.append(ReActTrace(
                    step_type=ReActStep.THOUGHT,
                    content=parsed["thought"]
                ))
            
            # Check for final answer
            if parsed.get("final_answer"):
                trace.append(ReActTrace(
                    step_type=ReActStep.FINAL_ANSWER,
                    content=parsed["final_answer"]
                ))
                
                return ReActResult(
                    success=True,
                    final_answer=parsed["final_answer"],
                    trace=trace,
                    iterations=iteration + 1,
                    total_time_ms=(time.time() - start_time) * 1000,
                    tools_used=list(set(tools_used)),
                    sources=sources
                )
            
            # Execute action
            if parsed.get("action"):
                action_name = parsed["action"]
                action_input = parsed.get("action_input", {})
                
                trace.append(ReActTrace(
                    step_type=ReActStep.ACTION,
                    content=f"Executing {action_name}",
                    tool_name=action_name,
                    tool_input=action_input
                ))
                
                # Execute tool
                if action_name in self.tools:
                    tool = self.tools[action_name]
                    tool_result = await tool(
                        query=query,
                        parameters=action_input,
                        intent=context.get("intent") if context else None
                    )
                    
                    tools_used.append(action_name)
                    
                    # Collect sources
                    if tool_result.sources:
                        sources.extend(tool_result.sources)
                    
                    # Format observation
                    if tool_result.success:
                        observation = json.dumps(tool_result.data, indent=2)
                    else:
                        observation = f"Error: {tool_result.error}"
                    
                    trace.append(ReActTrace(
                        step_type=ReActStep.OBSERVATION,
                        content=observation,
                        tool_name=action_name,
                        tool_output=tool_result.data if tool_result.success else tool_result.error
                    ))
                    
                    # Add to messages for next iteration
                    messages.append(AIMessage(content=response_text))
                    messages.append(HumanMessage(content=f"Observation: {observation}"))
                else:
                    # Tool not found
                    observation = f"Error: Tool '{action_name}' not found. Available tools: {list(self.tools.keys())}"
                    trace.append(ReActTrace(
                        step_type=ReActStep.OBSERVATION,
                        content=observation
                    ))
                    messages.append(AIMessage(content=response_text))
                    messages.append(HumanMessage(content=f"Observation: {observation}"))
            else:
                # No action or final answer - prompt to continue
                messages.append(AIMessage(content=response_text))
                messages.append(HumanMessage(content="Please continue with your reasoning and provide either an Action or Final Answer."))
        
        # Max iterations reached - force final answer
        messages.append(HumanMessage(content="Maximum iterations reached. Please provide your best Final Answer based on the information gathered."))
        
        response = await self.llm.ainvoke(messages)
        final_answer = self._extract_final_answer(response.content)
        
        return ReActResult(
            success=True,
            final_answer=final_answer,
            trace=trace,
            iterations=self.max_iterations,
            total_time_ms=(time.time() - start_time) * 1000,
            tools_used=list(set(tools_used)),
            sources=sources
        )
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract thought, action, or final answer."""
        result = {}
        
        # Extract thought
        if "Thought:" in response:
            thought_start = response.index("Thought:") + len("Thought:")
            # Find where thought ends
            thought_end = len(response)
            for marker in ["Action:", "Final Answer:"]:
                if marker in response:
                    marker_pos = response.index(marker)
                    if marker_pos > thought_start:
                        thought_end = min(thought_end, marker_pos)
            result["thought"] = response[thought_start:thought_end].strip()
        
        # Extract final answer
        if "Final Answer:" in response:
            answer_start = response.index("Final Answer:") + len("Final Answer:")
            result["final_answer"] = response[answer_start:].strip()
            return result
        
        # Extract action
        if "Action:" in response:
            action_start = response.index("Action:") + len("Action:")
            action_end = response.index("Action Input:") if "Action Input:" in response else len(response)
            result["action"] = response[action_start:action_end].strip()
            
            # Extract action input
            if "Action Input:" in response:
                input_start = response.index("Action Input:") + len("Action Input:")
                input_text = response[input_start:].strip()
                
                # Try to parse as JSON
                try:
                    # Find JSON object
                    if "{" in input_text:
                        json_start = input_text.index("{")
                        json_end = input_text.rindex("}") + 1
                        result["action_input"] = json.loads(input_text[json_start:json_end])
                    else:
                        result["action_input"] = {"query": input_text}
                except json.JSONDecodeError:
                    result["action_input"] = {"query": input_text}
        
        return result
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response."""
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()
        return response.strip()

"""
Agent Orchestrator
The central "brain" that coordinates all agentic operations.
Uses a generalized LLM (GPT-4/Claude) for high-level reasoning and planning.

Integrates:
- RouterAgent: Query analysis and routing
- ReActAgent: Iterative reasoning with tool use
- ValidationAgent: Fact verification against trusted sources
- ReflectionLoop: Self-correction when retrieval is insufficient
- MultiAgentCoordinator: Complex multi-step workflows
- ConversationMemory: Session context management
- Guardrails: Input/output validation
- PIIFilter: Privacy protection
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.config.settings import settings
from src.agentic.router_agent import RouterAgent, QueryAnalysis, QueryIntent, QueryComplexity
from src.agentic.reflection_loop import ReflectionLoop, ReflectionResult


class AgentState(str, Enum):
    """Current state of the agent."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    GATHERING_INFO = "gathering_info"
    RETRIEVING = "retrieving"
    REASONING = "reasoning"
    VALIDATING = "validating"
    EXECUTING_ACTION = "executing_action"
    RESPONDING = "responding"
    AWAITING_USER_INPUT = "awaiting_user_input"
    ERROR = "error"


@dataclass
class OrchestratorContext:
    """Context maintained by the orchestrator during a session."""
    # OPTIMIZATION: Bounded tool results to prevent memory bloat
    MAX_TOOL_RESULTS = 5
    MAX_CONVERSATION_HISTORY = 50
    
    session_id: str
    user_id: Optional[str]
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    collected_parameters: Dict[str, Any] = field(default_factory=dict)
    missing_parameters: List[str] = field(default_factory=list)
    current_intent: Optional[QueryIntent] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    state: AgentState = AgentState.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def prune_tool_results(self):
        """OPTIMIZATION: Keep only the last N tool results to prevent memory bloat."""
        if len(self.tool_results) > self.MAX_TOOL_RESULTS:
            self.tool_results = self.tool_results[-self.MAX_TOOL_RESULTS:]
    
    def prune_conversation_history(self):
        """OPTIMIZATION: Keep only the last N messages to prevent memory bloat."""
        if len(self.conversation_history) > self.MAX_CONVERSATION_HISTORY:
            self.conversation_history = self.conversation_history[-self.MAX_CONVERSATION_HISTORY:]
    
    def add_tool_result(self, result: Dict[str, Any]):
        """Add tool result with automatic pruning."""
        self.tool_results.append(result)
        self.prune_tool_results()
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history with automatic pruning."""
        self.conversation_history.append({"role": role, "content": content})
        self.prune_conversation_history()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversation_history": self.conversation_history,
            "collected_parameters": self.collected_parameters,
            "missing_parameters": self.missing_parameters,
            "current_intent": self.current_intent.value if self.current_intent else None,
            "state": self.state.value,
            "tool_results_count": len(self.tool_results),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }



@dataclass
class OrchestratorResponse:
    """Response from the orchestrator."""
    response_text: str
    requires_user_input: bool = False
    clarifying_question: Optional[str] = None
    suggested_options: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    state: AgentState = AgentState.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response_text": self.response_text,
            "requires_user_input": self.requires_user_input,
            "clarifying_question": self.clarifying_question,
            "suggested_options": self.suggested_options,
            "sources": self.sources,
            "actions_taken": self.actions_taken,
            "validation_warnings": self.validation_warnings,
            "next_steps": self.next_steps,
            "state": self.state.value,
            "metadata": self.metadata
        }


class AgentOrchestrator:
    """
    Central orchestrator that coordinates all agent operations.
    
    Architecture:
    - Uses a generalized LLM (GPT-4/Claude) as the "brain" for reasoning
    - Delegates specialized tasks to SLM workers or tools
    - Maintains state across multi-turn conversations
    - Implements the "Brain-and-Tool" pattern
    """
    
    ORCHESTRATOR_SYSTEM_PROMPT = """You are an intelligent orchestrator for JD Jones Manufacturing's AI system.

Your role is to:
1. ANALYZE user queries to understand intent and identify missing information
2. COORDINATE tools and agents to gather necessary data
3. VALIDATE results against industry standards and safety requirements
4. PROVIDE accurate, helpful responses

CONTEXT:
- JD Jones manufactures industrial sealing solutions (gaskets, packings, expansion joints)
- Products must meet various industry standards (API 622/624, Shell SPE, Saudi Aramco)
- Technical queries often require multiple parameters (temperature, pressure, media, industry)

DECISION MAKING:
- If critical parameters are missing, ASK clarifying questions before proceeding
- If multiple products could work, present OPTIONS with trade-offs
- If a standard/certification is mentioned, VERIFY compliance
- If you're uncertain, be TRANSPARENT about limitations

RESPONSE FORMAT:
- Be concise but complete
- Include relevant technical specifications
- Cite sources when using retrieved information
- Warn about any safety considerations

Current collected parameters: {parameters}
Missing critical parameters: {missing_params}
Current conversation state: {state}
"""

    def __init__(self):
        """Initialize the orchestrator with all integrated components."""
        # Initialize LLM based on provider setting
        self.reasoning_llm = self._init_llm()
        
        # === Core Agents ===
        # Router for query analysis
        self.router = RouterAgent()
        
        # Reflection loop for self-correction
        self.reflection = ReflectionLoop()
        
        # === Specialized Agents (lazy initialized) ===
        self._react_agent = None      # For iterative reasoning
        self._validation_agent = None # For fact verification
        self._multi_agent_coordinator = None  # For complex workflows
        self._specialist_registry = None  # For domain-specific specialists
        
        # === SOTA Enhancements (lazy initialized) ===
        self._tiered_intelligence = None  # LLM → SLM → sklearn routing
        self._sota_integration = None     # Full SOTA pipeline
        
        # === Memory & Context ===
        # Conversation memory per session
        self._memory_store: Dict[str, "ConversationMemory"] = {}
        
        # === Safety & Compliance ===
        # Guardrails for input/output validation
        self._guardrails = None  # Lazy init
        
        # PII filter for privacy
        self._pii_filter = None  # Lazy init
        
        # === Tools & Sessions ===
        # Registered tools
        self.tools: Dict[str, Callable] = {}
        
        # Active sessions
        self.sessions: Dict[str, OrchestratorContext] = {}
        
        # Session locks to prevent race conditions
        self._session_locks: Dict[str, asyncio.Lock] = {}
    
    # === Lazy Initialization Methods ===
    
    def _init_llm(self):
        """Initialize LLM based on provider setting."""
        provider = getattr(settings, 'llm_provider', 'openai').lower()
        
        if provider == 'ollama':
            # Ollama exposes OpenAI-compatible API, so we can use ChatOpenAI
            # pointing to Ollama's endpoint
            ollama_base = getattr(settings, 'ollama_base_url', 'http://localhost:11434')
            return ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                base_url=f"{ollama_base}/v1",
                api_key="ollama"  # Ollama doesn't need a real key
            )
        else:
            # Default to OpenAI
            return ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                openai_api_key=settings.openai_api_key
            )
    
    def _get_react_agent(self):
        """Lazy initialization of ReAct agent for iterative reasoning."""
        if self._react_agent is None:
            from src.agentic.agents.react_agent import ReActAgent
            self._react_agent = ReActAgent(
                tools=self.tools
            )
        return self._react_agent
    
    def _get_validation_agent(self):
        """Lazy initialization of Validation agent for fact checking."""
        if self._validation_agent is None:
            from src.agentic.agents.validation_agent import ValidationAgent
            self._validation_agent = ValidationAgent()
        return self._validation_agent
    
    def _get_multi_agent_coordinator(self):
        """Lazy initialization of Multi-Agent Coordinator for complex workflows."""
        if self._multi_agent_coordinator is None:
            from src.agentic.multi_agent.coordinator import MultiAgentCoordinator
            self._multi_agent_coordinator = MultiAgentCoordinator()
            
            # Register specialized agents
            self._multi_agent_coordinator.register_agent(
                "react", self._get_react_agent()
            )
            self._multi_agent_coordinator.register_agent(
                "validation", self._get_validation_agent()
            )
        return self._multi_agent_coordinator
    
    def _get_specialist_registry(self):
        """Lazy initialization of specialist agent registry."""
        if self._specialist_registry is None:
            from src.agentic.agents.specialized_agents import SpecialistAgentRegistry
            self._specialist_registry = SpecialistAgentRegistry()
            
            # Register tools with specialists
            tools = {}
            for name, tool_info in self.tools.items():
                if isinstance(tool_info, dict) and "function" in tool_info:
                    tools[name] = tool_info["function"]
            self._specialist_registry.register_tools(tools)
        return self._specialist_registry
    
    def _get_guardrails(self):
        """Lazy initialization of guardrails."""
        if self._guardrails is None:
            from src.agentic.hitl.guardrails import Guardrails
            self._guardrails = Guardrails()
        return self._guardrails
    
    def _get_pii_filter(self):
        """Lazy initialization of PII filter."""
        if self._pii_filter is None:
            from src.agentic.slm.pii_filter import PIIFilter
            self._pii_filter = PIIFilter()
        return self._pii_filter
    
    def _get_memory(self, session_id: str) -> "ConversationMemory":
        """Get or create conversation memory for session."""
        if session_id not in self._memory_store:
            from src.agentic.memory.conversation_memory import ConversationMemory
            self._memory_store[session_id] = ConversationMemory(
                max_messages=20,
                context_window=10,
                ttl_hours=24
            )
        return self._memory_store[session_id]
    
    def _get_tiered_intelligence(self):
        """Lazy initialization of SOTA Tiered Intelligence for fast query routing."""
        if self._tiered_intelligence is None:
            try:
                from src.sota.tiered_intelligence import get_tiered_intelligence
                self._tiered_intelligence = get_tiered_intelligence()
            except ImportError:
                logger.debug("SOTA Tiered Intelligence not available")
        return self._tiered_intelligence
    
    def _get_sota_integration(self):
        """Lazy initialization of SOTA Integration layer."""
        if self._sota_integration is None:
            try:
                from src.sota.integration import get_sota_integration
                self._sota_integration = get_sota_integration()
            except ImportError:
                logger.debug("SOTA Integration not available")
        return self._sota_integration
    
    @property
    def guardrails(self):
        """Public access to guardrails for testing and external use."""
        return self._get_guardrails()
    
    def register_tool(self, name: str, tool: Callable, description: str = ""):
        """
        Register a tool for the orchestrator to use.
        
        Args:
            name: Tool identifier
            tool: Callable tool function
            description: Tool description for the LLM
        """
        self.tools[name] = {
            "function": tool,
            "description": description
        }
        
        # Update ReAct agent if already initialized
        if self._react_agent is not None:
            self._react_agent.tools = self.tools
    
    def get_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> OrchestratorContext:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = OrchestratorContext(
                session_id=session_id,
                user_id=user_id
            )
        return self.sessions[session_id]
    
    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a session to prevent race conditions."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]
    
    async def process(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> OrchestratorResponse:
        """
        Process a user query through the full agentic pipeline.
        
        Pipeline:
        1. Input validation (guardrails)
        2. Query analysis (router)
        3. Parameter collection
        4. Tool execution
        5. Response generation
        6. Output validation (guardrails)
        7. Memory update
        
        Args:
            query: User's input
            session_id: Session identifier
            user_id: Optional user identifier
            user_context: Additional context (role, department, etc.)
            
        Returns:
            OrchestratorResponse with result and metadata
        """
        # Acquire session lock to prevent race conditions
        session_lock = self._get_session_lock(session_id)
        
        async with session_lock:
            return await self._process_with_lock(
                query=query,
                session_id=session_id,
                user_id=user_id,
                user_context=user_context
            )
    
    async def _process_with_lock(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> OrchestratorResponse:
        """Internal processing method called with session lock held."""
        # Get or create session context
        context = self.get_or_create_session(session_id, user_id)
        context.last_activity = datetime.now()
        context.state = AgentState.ANALYZING
        
        # Get memory for this session
        memory = self._get_memory(session_id)
        
        # Step 0: Input validation with guardrails
        guardrails = self._get_guardrails()
        input_check = guardrails.check_input(query)
        if not input_check.passed:
            return OrchestratorResponse(
                response_text=f"I cannot process this request: {input_check.message}",
                validation_warnings=input_check.warnings,
                state=AgentState.FAILED
            )
        
        # Add query to conversation history and memory
        context.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        memory.add_message(session_id, "user", query)
        
        try:
            # === SOTA Fast Path: Try Tiered Intelligence for simple queries ===
            # This can handle 70% of queries in <100ms using sklearn/SLM
            tiered = self._get_tiered_intelligence()
            if tiered:
                try:
                    # Build context for tiered intelligence
                    tiered_context = {
                        "collected_parameters": context.collected_parameters,
                        "has_conversation_history": len(context.conversation_history) > 1
                    }
                    
                    tiered_response = await tiered.process(query, tiered_context)
                    
                    # If tiered intelligence handled it with high confidence, return fast
                    if tiered_response.content and tiered_response.confidence >= 0.85:
                        logger.info(f"SOTA fast path: {tiered_response.tier_used.name} ({tiered_response.latency_ms:.0f}ms)")
                        
                        # Add response to history
                        response_text = tiered_response.content
                        context.conversation_history.append({
                            "role": "assistant",
                            "content": response_text,
                            "timestamp": datetime.now().isoformat()
                        })
                        memory.add_message(session_id, "assistant", response_text)
                        
                        return OrchestratorResponse(
                            response_text=response_text,
                            state=AgentState.IDLE,
                            metadata={
                                "tier_used": tiered_response.tier_used.name,
                                "latency_ms": tiered_response.latency_ms,
                                "confidence": tiered_response.confidence,
                                "fast_path": True
                            }
                        )
                except Exception as e:
                    logger.debug(f"SOTA fast path skipped: {e}")
            
            # Step 1: Analyze the query (full pipeline)
            analysis = await self.router.analyze(
                query=query,
                context=context.collected_parameters,
                conversation_history=context.conversation_history
            )
            context.current_intent = analysis.intent
            
            # Step 2: Check for missing parameters
            if analysis.missing_parameters:
                context.missing_parameters = analysis.missing_parameters
                context.state = AgentState.AWAITING_USER_INPUT
                
                # Generate clarifying question
                clarifying_q = await self._generate_clarifying_question(
                    analysis, context
                )
                
                return OrchestratorResponse(
                    response_text=clarifying_q,
                    requires_user_input=True,
                    clarifying_question=clarifying_q,
                    suggested_options=analysis.suggested_values.get(
                        analysis.missing_parameters[0], []
                    ) if analysis.suggested_values else [],
                    state=AgentState.AWAITING_USER_INPUT
                )
            
            # Step 3: Update collected parameters
            context.collected_parameters.update(analysis.extracted_parameters)
            context.state = AgentState.RETRIEVING
            
            # Step 4: Determine execution strategy based on query complexity
            use_react = analysis.intent in [
                QueryIntent.TECHNICAL_SPECIFICATION,
                QueryIntent.COMPARISON,
                QueryIntent.COMPLIANCE
            ] or analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]
            
            use_multi_agent = user_context and user_context.get("use_multi_agent", False)
            
            # Step 4.5: Check if a specialist agent should handle this
            specialist_registry = self._get_specialist_registry()
            specialist_domain = specialist_registry.determine_specialist(
                query=query,
                intent=analysis.intent.value if analysis.intent else ""
            )
            use_specialist = specialist_domain is not None
            
            # Step 5: Execute based on strategy (priority: multi-agent > specialist > react > standard)
            if use_multi_agent and analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
                # Use Multi-Agent Coordinator for complex workflows
                coordinator = self._get_multi_agent_coordinator()
                workflow_result = await coordinator.execute_workflow(
                    query=query,
                    context={
                        "intent": analysis.intent.value if analysis.intent else "general",
                        "parameters": context.collected_parameters,
                        **(user_context or {})
                    },
                    user_id=user_context.get("user_id") if user_context else None
                )
                tool_results = [{
                    "tool": "multi_agent_workflow",
                    "success": workflow_result.success,
                    "result": workflow_result.to_dict()
                }]
            elif use_specialist:
                # Delegate to specialist agent for domain-specific precision
                specialist = specialist_registry.get_specialist(specialist_domain)
                specialist_name = specialist_domain.value
                
                logger.info(f"Delegating to specialist: {specialist_name}")
                
                specialist_result = await specialist.execute(
                    query=query,
                    context={
                        "intent": analysis.intent.value if analysis.intent else "general",
                        "parameters": context.collected_parameters,
                        "user_context": user_context or {}
                    }
                )
                tool_results = [{
                    "tool": f"specialist_{specialist_name}",
                    "success": specialist_result.success,
                    "result": specialist_result.to_dict(),
                    "sources": specialist_result.sources,
                    "specialist": specialist_name,
                    "trace": [t.to_dict() for t in specialist_result.trace]
                }]
            elif use_react:
                # Use ReAct agent for iterative reasoning
                react_agent = self._get_react_agent()
                react_result = await react_agent.execute(
                    query=query,
                    context={
                        "intent": analysis.intent.value if analysis.intent else "general",
                        "parameters": context.collected_parameters,
                        "user_context": user_context or {}
                    }
                )
                tool_results = [{
                    "tool": "react_agent",
                    "success": react_result.success,
                    "result": react_result.to_dict(),
                    "sources": react_result.sources
                }]
            else:
                # Standard tool execution
                tool_results = await self._execute_tools(analysis, context, user_context)
            
            context.tool_results.extend(tool_results)
            
            # Step 6: Validate results with ValidationAgent
            context.state = AgentState.VALIDATING
            
            # Format tool results for validation
            response_text_for_validation = json.dumps(tool_results, indent=2, default=str)
            
            validation_agent = self._get_validation_agent()
            validation_report = await validation_agent.validate_response(
                response=response_text_for_validation,
                context=context.collected_parameters
            )
            
            # Step 7: Reflection and self-correction
            reflection_result = await self.reflection.validate(
                query=query,
                intent=analysis.intent,
                tool_results=tool_results,
                parameters=context.collected_parameters
            )
            context.validation_results.append(reflection_result.to_dict())
            
            # Add validation warnings from ValidationAgent
            if validation_report.overall_reliability < 0.7:
                reflection_result.warnings.extend(validation_report.recommendations)
            
            # Step 8: If validation fails, iterate with corrections
            if not reflection_result.is_valid and reflection_result.corrective_actions:
                # Re-query with corrections using ReAct for better results
                react_agent = self._get_react_agent()
                corrected_result = await react_agent.execute(
                    query=f"Correction needed: {reflection_result.corrective_actions[0]}. Original query: {query}",
                    context={
                        "intent": analysis.intent.value if analysis.intent else "general",
                        "parameters": context.collected_parameters,
                        "previous_results": tool_results
                    }
                )
                tool_results.append({
                    "tool": "react_agent_correction",
                    "success": corrected_result.success,
                    "result": corrected_result.to_dict()
                })
            
            # Step 9: Generate final response
            context.state = AgentState.RESPONDING
            response = await self._generate_response(
                query=query,
                analysis=analysis,
                tool_results=tool_results,
                reflection=reflection_result,
                context=context,
                user_context=user_context
            )
            
            # Update memory with response
            memory.add_message(session_id, "assistant", response.response_text)
            
            # Add response to history
            context.conversation_history.append({
                "role": "assistant",
                "content": response.response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            context.state = AgentState.IDLE
            return response
            
        except Exception as e:
            context.state = AgentState.ERROR
            return OrchestratorResponse(
                response_text=f"I encountered an issue processing your request. Please try rephrasing or contact support.",
                state=AgentState.ERROR,
                metadata={"error": str(e)}
            )
    
    async def _generate_clarifying_question(
        self,
        analysis: QueryAnalysis,
        context: OrchestratorContext
    ) -> str:
        """Generate a clarifying question for missing parameters."""
        missing_param = analysis.missing_parameters[0]
        suggested = analysis.suggested_values.get(missing_param, [])
        
        # Map parameters to natural language questions
        param_questions = {
            "temperature": "What is the operating temperature range for your application?",
            "pressure": "What pressure will the sealing solution need to withstand?",
            "media": "What fluid or media will be in contact with the seal?",
            "industry": "Which industry is this for? (e.g., Oil & Gas, Chemical, Pharmaceutical)",
            "equipment_type": "What type of equipment needs sealing?",
            "size": "What size/dimensions are required?",
            "certification": "Are there any specific certifications required (e.g., API 622, FDA)?",
            "ph_level": "What is the pH level of the media?",
        }
        
        base_question = param_questions.get(
            missing_param,
            f"Could you please provide the {missing_param.replace('_', ' ')}?"
        )
        
        # Add suggested options if available
        if suggested:
            options_text = ", ".join(suggested[:5])
            base_question += f"\n\nCommon options include: {options_text}"
        
        return base_question
    
    async def _execute_tools(
        self,
        analysis: QueryAnalysis,
        context: OrchestratorContext,
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute appropriate tools based on query analysis.
        
        OPTIMIZATION: Tools are executed in parallel using asyncio.gather,
        reducing latency significantly for multi-tool queries.
        """
        # Merge user context into parameters for RBAC
        parameters = context.collected_parameters.copy()
        if user_context:
            parameters["user_context"] = user_context
            parameters["user_role"] = user_context.get("role", "employee")
            parameters["department"] = user_context.get("department")
            parameters["user_id"] = user_context.get("user_id")
        
        # Collect tools to execute
        tools_to_execute = [
            (tool_name, self.tools[tool_name])
            for tool_name in analysis.required_tools
            if tool_name in self.tools
        ]
        
        if not tools_to_execute:
            return []
        
        # Execute tools in parallel
        async def execute_single_tool(tool_name: str, tool: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = await tool["function"](
                    query=analysis.original_query,
                    parameters=parameters,
                    intent=analysis.intent
                )
                return {
                    "tool": tool_name,
                    "success": True,
                    "result": result
                }
            except Exception as e:
                return {
                    "tool": tool_name,
                    "success": False,
                    "error": str(e)
                }
        
        # Run all tools concurrently
        results = await asyncio.gather(*[
            execute_single_tool(name, tool) 
            for name, tool in tools_to_execute
        ], return_exceptions=True)
        
        # Handle any unexpected exceptions from gather
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_name = tools_to_execute[i][0]
                final_results.append({
                    "tool": tool_name,
                    "success": False,
                    "error": str(result)
                })
            else:
                final_results.append(result)
        
        return final_results
    
    async def _apply_corrections(
        self,
        reflection: ReflectionResult,
        context: OrchestratorContext
    ) -> List[Dict[str, Any]]:
        """Apply corrective actions from reflection."""
        results = []
        
        for action in reflection.corrective_actions:
            if action.get("type") == "re_query":
                # Re-execute a tool with modified parameters
                tool_name = action.get("tool")
                if tool_name and tool_name in self.tools:
                    try:
                        tool = self.tools[tool_name]
                        result = await tool["function"](
                            query=action.get("modified_query", ""),
                            parameters=action.get("parameters", {}),
                            intent=context.current_intent
                        )
                        results.append({
                            "tool": tool_name,
                            "correction": True,
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "tool": tool_name,
                            "correction": True,
                            "error": str(e)
                        })
        
        return results
    
    async def _generate_response(
        self,
        query: str,
        analysis: QueryAnalysis,
        tool_results: List[Dict[str, Any]],
        reflection: ReflectionResult,
        context: OrchestratorContext,
        user_context: Optional[Dict[str, Any]]
    ) -> OrchestratorResponse:
        """Generate the final response using the reasoning LLM."""
        
        # Build context for the LLM
        system_prompt = self.ORCHESTRATOR_SYSTEM_PROMPT.format(
            parameters=json.dumps(context.collected_parameters, indent=2),
            missing_params=", ".join(context.missing_parameters) or "None",
            state=context.state.value
        )
        
        # Compile tool results
        tool_context = ""
        sources = []
        for result in tool_results:
            if result.get("success") and result.get("result"):
                tool_context += f"\n[{result['tool']}]: {json.dumps(result['result'], indent=2)}"
                if isinstance(result['result'], dict) and 'sources' in result['result']:
                    sources.extend(result['result']['sources'])
        
        # Build the prompt
        user_prompt = f"""
User Query: {query}

Intent: {analysis.intent.value}
Confidence: {analysis.confidence}

Retrieved Information:
{tool_context}

Validation Notes:
- Valid: {reflection.is_valid}
- Issues: {', '.join(reflection.issues) if reflection.issues else 'None'}
- Warnings: {', '.join(reflection.warnings) if reflection.warnings else 'None'}

User Context: {json.dumps(user_context or {}, indent=2)}

Please provide a helpful, accurate response. If recommending products, include specifications and any relevant certifications.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Get response from reasoning LLM
        response = await self.reasoning_llm.ainvoke(messages)
        
        return OrchestratorResponse(
            response_text=response.content,
            sources=sources,
            actions_taken=[r["tool"] for r in tool_results if r.get("success")],
            validation_warnings=reflection.warnings,
            next_steps=analysis.follow_up_questions,
            state=AgentState.IDLE,
            metadata={
                "intent": analysis.intent.value,
                "confidence": analysis.confidence,
                "parameters_used": context.collected_parameters
            }
        )
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session and its associated lock."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            # Also clean up the session lock if it exists
            if session_id in self._session_locks:
                del self._session_locks[session_id]
            return True
        return False
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state."""
        if session_id in self.sessions:
            return self.sessions[session_id].to_dict()
        return None

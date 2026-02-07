# JD Jones RAG System - Unified Implementation Plan

**Created:** 2026-02-06  
**Unified:** 2026-02-07  
**Status:** PLANNING (Not Implemented)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Phase 1: LangFuse Observability](#2-phase-1-langfuse-observability)
3. [Phase 2: LangGraph Agent](#3-phase-2-langgraph-agent)
4. [Phase 3: QLoRA Fine-Tuning Pipeline](#4-phase-3-qlora-fine-tuning-pipeline)
5. [Complete File Change Summary](#5-complete-file-change-summary)
6. [Dependencies & Environment](#6-dependencies--environment)
7. [Testing & Verification](#7-testing--verification)
8. [Timeline](#8-timeline)

---

## 1. Executive Summary

### Context
The JD Jones RAG system is a 95-file production-grade agentic RAG platform for JD Jones Manufacturing (industrial sealing solutions). Three major enhancements are planned:

1. **LangFuse Observability** — Wire existing tracer stubs, add self-hosted Docker services, evaluation hooks
2. **LangGraph Agent** — Convert ReAct loop to StateGraph with debugging/breakpoints/HITL
3. **QLoRA Fine-Tuning** — Extend trainer with LoRA/QLoRA/DoRA/QDoRA, GGUF export, Ollama registration

### Implementation Order Rationale
Observability first = all subsequent work gets automatic tracing. LangGraph second = needs tracer. Fine-tuning third = depends on RAG data capture from observability.

### Existing Infrastructure (Already Present)
- `AgentTracer` in `observability/tracer.py` — Has placeholder `_export_to_langsmith()` and `_export_to_langfuse()` methods
- `AgentMonitor` in `observability/monitor.py` — Performance metrics
- `ModelTrainer` in `fine_tuning/trainer.py` — Basic LoRA support
- `AgentOrchestrator` in `orchestrator.py` — Current orchestration
- `ReActAgent` in `agents/react_agent.py` — Current ReAct loop

---

## 2. Phase 1: LangFuse Observability (~7 files)

### 2.1 New Files

| File | Purpose |
|------|---------|
| `src/agentic/observability/langfuse_integration.py` | `LangFuseExporter` class: maps Trace/Span model to LangFuse API |
| `src/agentic/observability/evaluation_hooks.py` | `EvaluationHooks` class: hallucination detection, factual grounding, domain accuracy |

### 2.2 Modified Files

| File | Changes |
|------|---------|
| `docker-compose.yml` | Add `langfuse-postgres` + `langfuse-server` services |
| `src/config/settings.py` | Add observability settings (backend selector, keys, feature flags) |
| `src/agentic/observability/tracer.py` | Wire `_export_langfuse()` to `LangFuseExporter.export_trace()`; wire `_export_langsmith()` for env-based auto-tracing |
| `src/agentic/observability/__init__.py` | Export new classes |
| `src/agentic/orchestrator.py` | Call `EvaluationHooks.evaluate_response()` after response generation |

### 2.3 LangFuseExporter

```python
class LangFuseExporter:
    """
    Exports AgentTracer traces to LangFuse.
    Maps our trace/span model to LangFuse's model:
    - Trace -> langfuse.trace()
    - SpanType.LLM -> langfuse generation() (better analytics)
    - SpanType.TOOL -> langfuse span()
    - SpanType.RETRIEVAL -> langfuse span() with retrieval metadata
    - SpanType.AGENT -> langfuse span()
    """

    def __init__(self, public_key, secret_key, host="http://localhost:3002",
                 project_name="jd_jones_rag", enabled=True):
        if not enabled:
            self._client = None
            return
        from langfuse import Langfuse
        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

    def export_trace(self, trace: Trace) -> Optional[str]:
        """Export full trace to LangFuse. Returns LangFuse trace ID."""
        if not self._client:
            return None
        try:
            lf_trace = self._client.trace(
                name=trace.name,
                id=trace.trace_id,
                user_id=trace.user_id,
                session_id=trace.session_id,
                metadata=trace.metadata,
            )
            for span in trace.spans:
                self._export_span(lf_trace, span)
            self._client.flush()
            logger.debug(f"Exported trace {trace.trace_id} to LangFuse")
            return lf_trace.id
        except Exception as e:
            logger.warning(f"Failed to export to LangFuse: {e}")
            return None

    def _export_span(self, lf_trace, span: TraceSpan):
        """Export single span, using generation() for LLM calls."""
        if span.span_type == SpanType.LLM:
            lf_trace.generation(
                name=span.name,
                input=span.input_data,
                output=span.output,
                start_time=span.started_at,
                end_time=span.ended_at,
                metadata=span.metadata,
            )
        else:
            lf_trace.span(
                name=span.name,
                input=span.input_data,
                output=span.output,
                start_time=span.started_at,
                end_time=span.ended_at,
                metadata={"type": span.span_type.value, **span.metadata},
            )

    def log_score(self, trace_id, name, value, comment=""):
        """Log evaluation score to LangFuse."""
        if self._client:
            self._client.score(trace_id=trace_id, name=name, value=value, comment=comment)

    def log_cost(self, trace_id, model, input_tokens, output_tokens,
                 cost_per_1k_input, cost_per_1k_output):
        """Log token cost to LangFuse."""
        if self._client:
            total = (input_tokens / 1000 * cost_per_1k_input +
                     output_tokens / 1000 * cost_per_1k_output)
            self._client.score(trace_id=trace_id, name="cost_usd", value=total)

    def shutdown(self):
        """Flush and close client."""
        if self._client:
            self._client.flush()
```

### 2.4 LangSmith Export (Tracer Enhancement)

```python
def _export_to_langsmith(self, trace: Trace):
    """Export trace to LangSmith for visualization and debugging."""
    if not settings.langsmith_api_key:
        return
    try:
        from langsmith import Client
        client = Client(api_key=settings.langsmith_api_key)
        run = client.create_run(
            name=trace.name,
            run_type="chain",
            project_name=settings.langsmith_project,
            inputs={"query": trace.metadata.get("query", "")},
            extra={"trace_id": trace.trace_id, "user_id": trace.user_id},
        )
        for span in trace.spans:
            client.create_run(
                name=span.name,
                run_type=self._map_span_type(span.span_type),
                parent_run_id=run.id,
                inputs=span.input_data or {},
                outputs={"result": span.output} if span.output else {},
                error=span.error,
                start_time=span.started_at,
                end_time=span.ended_at,
            )
        client.update_run(run.id, outputs={"response": trace.metadata.get("response", "")},
                          end_time=trace.ended_at)
    except ImportError:
        logger.warning("langsmith package not installed")
    except Exception as e:
        logger.warning(f"Failed to export to LangSmith: {e}")

def _map_span_type(self, span_type: SpanType) -> str:
    mapping = {
        SpanType.LLM: "llm", SpanType.TOOL: "tool",
        SpanType.RETRIEVAL: "retriever", SpanType.AGENT: "chain",
        SpanType.DECISION: "chain", SpanType.VALIDATION: "chain",
    }
    return mapping.get(span_type, "chain")
```

### 2.5 EvaluationHooks

```python
class EvaluationHooks:
    """Automated evaluation hooks for agent responses. Integrates with LangFuse scoring API."""

    CRITICAL_DOMAIN_TERMS = {
        "api 622": "packing testing standard valve",
        "api 624": "valve fugitive emission testing",
        "pacmaan": "jd jones gasket product line",
        "flexseal": "jd jones packing product",
        "expansoflex": "jd jones expansion joint",
        "fugitive emissions": "unintentional gas vapor release leakage",
        "spiral wound": "gasket type metallic winding",
        "ptfe": "polytetrafluoroethylene sealing material",
    }

    def __init__(self, langfuse_exporter=None)
    async def evaluate_response(self, query, response, sources, trace_id=None) -> List[EvaluationScore]
    async def _check_hallucination(self, query, response, sources) -> EvaluationScore
    async def _check_factual_grounding(self, query, response, sources) -> EvaluationScore
    async def _check_relevance(self, query, response, sources) -> EvaluationScore
    async def _check_domain_accuracy(self, query, response, sources) -> EvaluationScore
```

### 2.6 Docker Services (docker-compose.yml additions)

```yaml
langfuse-postgres:
  image: postgres:16-alpine
  container_name: jd_jones_langfuse_postgres
  environment:
    POSTGRES_USER: langfuse
    POSTGRES_PASSWORD: ${LANGFUSE_POSTGRES_PASSWORD:-langfuse_secret}
    POSTGRES_DB: langfuse
  ports:
    - "5433:5432"
  volumes:
    - langfuse_postgres_data:/var/lib/postgresql/data
  networks:
    - jd_jones_network
  restart: unless-stopped
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U langfuse -d langfuse"]
    interval: 10s
    timeout: 5s
    retries: 5

langfuse-server:
  image: langfuse/langfuse:2
  container_name: jd_jones_langfuse
  ports:
    - "3002:3000"
  environment:
    DATABASE_URL: postgresql://langfuse:${LANGFUSE_POSTGRES_PASSWORD:-langfuse_secret}@langfuse-postgres:5432/langfuse
    NEXTAUTH_URL: http://localhost:3002
    NEXTAUTH_SECRET: ${LANGFUSE_NEXTAUTH_SECRET:-langfuse-next-auth-secret}
    SALT: ${LANGFUSE_SALT:-langfuse-salt-value}
    TELEMETRY_ENABLED: "false"
  depends_on:
    langfuse-postgres:
      condition: service_healthy
  networks:
    - jd_jones_network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:3000/api/public/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 30s
```

### 2.7 Settings Additions (settings.py)

```python
# Observability Settings
observability_backend: str = Field(default="memory")        # "memory", "langfuse", "langsmith"
langfuse_host: str = Field(default="http://localhost:3002")
langfuse_public_key: str = Field(default="")
langfuse_secret_key: str = Field(default="")
langsmith_api_key: Optional[str] = Field(default=None)
langsmith_project: str = Field(default="jd_jones_rag")
enable_cost_tracking: bool = Field(default=True)
enable_hallucination_detection: bool = Field(default=True)
```

---

## 3. Phase 2: LangGraph Agent (~5 files)

### 3.1 Design Decision: ReAct-Level Graph

The LangGraph wraps the **ReAct reasoning loop** (thought -> action -> observation cycle), NOT the outer pipeline. This is where debugging, breakpoints, and time-travel matter most.

**Key Design: Backward Compatibility**
- `LangGraphReActAgent.execute(query, context) -> ReActResult` (same return type as `ReActAgent`)
- `register_tool(tool: BaseTool)` (same interface)
- Orchestrator swaps via `use_langgraph_agent=True/False` setting
- Reuses `REACT_SYSTEM_PROMPT` verbatim from existing `ReActAgent`

### 3.2 New Files

| File | Purpose |
|------|---------|
| `src/agentic/agents/graph_state.py` | `AgentGraphState` TypedDict, `GraphNodeName` enum |
| `src/agentic/agents/langgraph_agent.py` | `LangGraphReActAgent` class with StateGraph |
| `src/agentic/agents/graph_config.py` | `GraphConfig` dataclass for checkpointer, breakpoints, HITL |
| `langgraph.json` | LangGraph Studio config (project root) |

### 3.3 Modified Files

| File | Changes |
|------|---------|
| `src/config/settings.py` | Add LangGraph settings |
| `src/agentic/orchestrator.py` | Modify `_get_react_agent()` to conditionally return `LangGraphReActAgent` |
| `src/agentic/agents/__init__.py` | Export `LangGraphReActAgent` |

### 3.4 AgentGraphState

```python
class AgentGraphState(TypedDict):
    query: str                          # Original user query
    context: Dict[str, Any]             # Retrieved context from RAG pipeline
    messages: Annotated[List[BaseMessage], add_messages]  # LLM conversation history
    current_thought: str                # Current reasoning step
    selected_action: Optional[str]      # Tool name to execute
    action_input: Optional[Dict]        # Tool input parameters
    observation: Optional[str]          # Tool execution result
    steps: List[Dict[str, Any]]         # Full ReAct trace (thought/action/observation triples)
    step_count: int                     # Current step number
    max_steps: int                      # Maximum allowed steps (default: 10)
    final_answer: Optional[str]         # Final response when reasoning complete
    needs_human_review: bool            # Whether HITL is triggered
    human_decision: Optional[str]       # "approved" or "rejected"
    error: Optional[str]               # Error message if any
    trace_id: Optional[str]            # For observability integration
```

### 3.5 GraphConfig

```python
@dataclass
class GraphConfig:
    checkpoint_backend: str = "sqlite"               # "sqlite" or "memory"
    checkpoint_path: str = "./data/langgraph_checkpoints.db"
    enable_breakpoints: bool = False                  # Pause before tool execution
    enable_human_in_loop: bool = False                # Route to HITL on sensitive ops
    breakpoint_nodes: List[str] = field(default_factory=lambda: ["tool_executor"])
    max_steps: int = 10
    sensitive_tools: List[str] = field(default_factory=lambda: [
        "generate_quotation", "submit_enquiry"
    ])
```

### 3.6 Graph Structure

```
THOUGHT -> [final_answer] -> FINAL_ANSWER -> END
THOUGHT -> [action] -> ACTION_SELECTOR -> TOOL_EXECUTOR -> OBSERVATION -> THOUGHT (loop)
                          |
                          +-> [human_review] -> HUMAN_REVIEW -> [approved] -> TOOL_EXECUTOR
                                                              -> [rejected] -> FINAL_ANSWER
```

### 3.7 LangGraphReActAgent

```python
class LangGraphReActAgent:
    """
    LangGraph-based ReAct agent with StateGraph.
    Drop-in replacement for ReActAgent with identical interface.
    Adds: breakpoints, time-travel debugging, HITL, visualization.
    """

    def __init__(self, llm, system_prompt, config: GraphConfig = None, tracer=None)
    def register_tool(self, tool: BaseTool)           # Same interface as ReActAgent
    async def execute(self, query, context) -> ReActResult  # Same return type
    def _build_graph(self) -> CompiledStateGraph      # Build the StateGraph

    # Node functions
    async def _thought_node(self, state: AgentGraphState) -> AgentGraphState
    async def _action_selector_node(self, state: AgentGraphState) -> AgentGraphState
    async def _tool_executor_node(self, state: AgentGraphState) -> AgentGraphState
    async def _observation_node(self, state: AgentGraphState) -> AgentGraphState
    async def _final_answer_node(self, state: AgentGraphState) -> AgentGraphState
    async def _human_review_node(self, state: AgentGraphState) -> AgentGraphState

    # Edge routing functions
    def _thought_router(self, state) -> str  # "action" | "final_answer"
    def _action_router(self, state) -> str   # "execute" | "human_review"
    def _human_router(self, state) -> str    # "approved" | "rejected"

    # Debugging features
    async def get_state_history(self, thread_id: str) -> List[StateSnapshot]
    async def replay_from_checkpoint(self, thread_id, checkpoint_id, updated_state=None) -> ReActResult
    def get_graph_visualization(self) -> str            # Returns Mermaid diagram
    def get_graph_png(self) -> bytes                    # Returns PNG image bytes
```

### 3.8 Debugging Features
- **Breakpoints**: `interrupt_before=["tool_executor"]` pauses before tool calls
- **Time-travel**: `get_state_history(thread_id)` returns all checkpointed states
- **Replay**: `replay_from_checkpoint(thread_id, checkpoint_id)` re-executes from any point
- **Visualization**: Mermaid diagram + PNG export

### 3.9 Settings Additions

```python
# LangGraph Settings
use_langgraph_agent: bool = Field(default=False)
agent_human_in_loop: bool = Field(default=False)
langgraph_checkpoint_backend: str = Field(default="sqlite")
langgraph_checkpoint_path: str = Field(default="./data/langgraph_checkpoints.db")
```

### 3.10 langgraph.json (project root)

```json
{
  "dependencies": ["."],
  "graphs": {
    "react_agent": "./src/agentic/agents/langgraph_agent.py:create_graph"
  },
  "env": ".env"
}
```

---

## 4. Phase 3: QLoRA Fine-Tuning Pipeline (~5 files)

### 4.1 New Files

| File | Purpose |
|------|---------|
| `src/fine_tuning/qlora_trainer.py` | `QLoRATrainer`: LoRA/QLoRA/DoRA/QDoRA via PEFT+bitsandbytes or Unsloth |
| `src/fine_tuning/gguf_exporter.py` | `GGUFExporter`: HF->GGUF conversion, quantization, Ollama registration |
| `src/fine_tuning/rag_data_generator.py` | `RAGDataGenerator`: quality-tiered data capture from RAG interactions |

### 4.2 Modified Files

| File | Changes |
|------|---------|
| `src/fine_tuning/trainer.py` | Add `LOCAL` provider dispatch to call `QLoRATrainer`; extend `TrainingConfig` |
| `src/fine_tuning/data_preparation.py` | Add `SHAREGPT` to `DataFormat` enum; add ShareGPT export |
| `src/fine_tuning/__init__.py` | Export new classes |
| `src/config/settings.py` | Add fine-tuning settings |
| `requirements.txt` | Add peft, bitsandbytes, trl, accelerate, datasets |

### 4.3 QLoRAConfig

```python
@dataclass
class QLoRAConfig:
    base_model: str = "unsloth/llama-3.2-3b-instruct-bnb-4bit"
    output_dir: str = "./fine_tuning_output"
    fine_tuning_method: str = "qlora"      # "lora", "qlora", "dora", "qdora"

    # LoRA hyperparameters
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_dora: bool = False                  # Weight-Decomposed LoRA (DoRA)

    # Quantization
    quantization_bits: int = 4              # 0=fp16, 4=4bit NF4, 8=8bit
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    lr_scheduler_type: str = "cosine"

    # Hardware
    use_unsloth: bool = True                # 2x faster training via Unsloth
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    @property
    def is_quantized(self) -> bool:
        return self.quantization_bits in (4, 8)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
```

### 4.4 QLoRATrainer

```python
class QLoRATrainer:
    """
    Trains local LLMs using LoRA/QLoRA/DoRA/QDoRA.
    Supports both standard PEFT+BitsAndBytes and Unsloth (2x faster) backends.
    """

    def __init__(self, config: QLoRAConfig)

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load base model with optional quantization."""
        if self.config.use_unsloth:
            from unsloth import FastLanguageModel
            return FastLanguageModel.from_pretrained(self.config.base_model)
        # Standard: AutoModelForCausalLM + BitsAndBytesConfig
        quant_config = self._get_bnb_config() if self.config.is_quantized else None
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.config.is_quantized:
            model = prepare_model_for_kbit_training(model)
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def apply_adapters(self, model) -> PeftModel:
        """Apply LoRA/DoRA adapters."""
        peft_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=self.config.use_dora,
        )
        return get_peft_model(model, peft_config)

    def train(self, dataset: List[Dict]) -> TrainingResult:
        """Full training pipeline: load -> adapt -> train -> save."""
        model, tokenizer = self.load_model()
        model = self.apply_adapters(model)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            fp16=True,
            optim="paged_adamw_32bit" if self.config.is_quantized else "adamw_torch",
            lr_scheduler_type=self.config.lr_scheduler_type,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_grad_norm=0.3,
            group_by_length=True,
        )

        trainer = SFTTrainer(
            model=model, args=training_args,
            train_dataset=dataset, tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
        )
        trainer.train()
        model.save_pretrained(f"{self.config.output_dir}/adapter")
        tokenizer.save_pretrained(f"{self.config.output_dir}/adapter")
        return TrainingResult(status="completed", output_dir=self.config.output_dir)

    def merge_and_save(self) -> str:
        """Merge LoRA adapters into base model. Returns path to merged model."""
        # Load base + adapter, merge, save to output_dir/merged
        pass

    def _get_bnb_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=(self.config.quantization_bits == 4),
            load_in_8bit=(self.config.quantization_bits == 8),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=self.config.use_double_quant,
        )
```

### 4.5 GGUFExporter

```python
class GGUFExporter:
    """Converts HuggingFace models to GGUF format and registers with Ollama."""

    @dataclass
    class Config:
        merged_model_path: str
        output_path: str = "./fine_tuning_output/gguf"
        quantization_type: str = "q4_k_m"       # q4_0, q4_k_m, q5_k_m, q8_0, f16
        llama_cpp_path: Optional[str] = None
        ollama_model_name: str = "jdjones-llama3.2"
        ollama_system_prompt: str = "You are JD Jones Manufacturing's expert assistant..."
        ollama_parameters: Dict[str, Any] = field(default_factory=lambda: {
            "temperature": 0.1, "top_p": 0.9, "num_ctx": 4096,
        })

    def __init__(self, config: Config)
    def export(self) -> GGUFExportResult:
        """Pipeline: HF -> GGUF -> Quantize -> Modelfile -> ollama create"""
    def _ensure_llama_cpp(self) -> str
    def _convert_to_gguf(self, model_path: str) -> str
    def _quantize(self, gguf_path: str) -> str
    def _generate_modelfile(self, quantized_path: str) -> str
    def _register_ollama(self, modelfile_path: str) -> bool
```

### 4.6 RAGDataGenerator

```python
class RAGDataGenerator:
    """
    Captures and curates RAG interactions for fine-tuning data.
    Quality tiers: Gold (human-verified), Silver (auto-validated), Bronze (raw).
    """

    def __init__(self, min_quality_score=0.7)
    def capture_interaction(self, query, response, sources, validation_score=None, user_feedback=None)
    def assess_quality(self, interaction) -> QualityTier:
        # Gold: user_feedback >= 0.8 | Silver: validation >= 0.85 AND sources >= 2 | Bronze: validation >= 0.7
    def generate_dataset(self, min_tier="silver") -> List[Dict]:
        # Format as ShareGPT: [{"from": "human", "value": query}, {"from": "gpt", "value": response}]
    def add_domain_seed_data(self):
        # 20+ JD Jones domain-specific Q&A (API 622/624, PACMAAN, FlexSeal, etc.)
    def export_to_jsonl(self, output_path, min_tier="silver")
```

### 4.7 Supported Methods Comparison

| Method | quantization_bits | use_dora | VRAM (3B) | VRAM (8B) | Quality |
|--------|-------------------|----------|-----------|-----------|---------|
| **LoRA** | 0 (fp16) | False | ~7 GB | ~18 GB | 95-98% |
| **QLoRA** | 4 | False | ~5 GB | ~10 GB | 93-97% |
| **DoRA** | 0 (fp16) | True | ~8 GB | ~17 GB | 98-101% |
| **QDoRA** | 4 | True | ~5.5 GB | ~11 GB | 96-100% |

### 4.8 End-to-End Pipeline

```
1. RAGDataGenerator captures quality-tiered interactions
2. QLoRATrainer loads base model (4-bit NF4 via BitsAndBytes)
3. Apply LoRA/DoRA adapters to target modules
4. Train with SFTTrainer (from trl)
5. Save adapter weights
6. Merge adapters into base model
7. GGUFExporter converts to GGUF via llama.cpp
8. Quantize to q4_k_m
9. Generate Ollama Modelfile
10. Register: ollama create jdjones-llama3.2 -f Modelfile
```

### 4.9 Ollama Modelfile Template

```
FROM ./jdjones-llama3.2-q4_k_m.gguf

TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

SYSTEM """You are JD Jones Manufacturing's expert assistant specializing in industrial sealing solutions.
You have deep knowledge of gaskets (PACMAAN line), packing (FlexSeal), expansion joints (ExpansoFlex),
and industry standards (API 622, API 624). Always cite specific product specifications and standards."""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

### 4.10 Settings Additions

```python
# Fine-tuning Settings
fine_tuning_output_dir: str = Field(default="./fine_tuning_output")
fine_tuning_base_model: str = Field(default="unsloth/llama-3.2-3b-instruct-bnb-4bit")
fine_tuning_method: str = Field(default="qlora")
fine_tuning_quantization_bits: int = Field(default=4)
fine_tuning_lora_rank: int = Field(default=64)
fine_tuning_lora_alpha: int = Field(default=16)
fine_tuning_lora_dropout: float = Field(default=0.0)
fine_tuning_target_modules: str = Field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
gguf_quantization_type: str = Field(default="q4_k_m")
ollama_model_name: str = Field(default="jdjones-llama3.2")
```

---

## 5. Complete File Change Summary

### New Files (9)
| # | File | Purpose |
|---|------|---------|
| 1 | `src/agentic/observability/langfuse_integration.py` | LangFuse exporter |
| 2 | `src/agentic/observability/evaluation_hooks.py` | Evaluation hooks (hallucination, grounding, relevance, domain) |
| 3 | `src/agentic/agents/graph_state.py` | LangGraph state TypedDict and enums |
| 4 | `src/agentic/agents/langgraph_agent.py` | LangGraph ReAct agent with StateGraph |
| 5 | `src/agentic/agents/graph_config.py` | Graph configuration dataclass |
| 6 | `src/fine_tuning/qlora_trainer.py` | QLoRA/DoRA trainer |
| 7 | `src/fine_tuning/gguf_exporter.py` | GGUF exporter with Ollama registration |
| 8 | `src/fine_tuning/rag_data_generator.py` | RAG data generator with quality tiers |
| 9 | `langgraph.json` | LangGraph Studio configuration |

### Modified Files (9)
| # | File | Changes |
|---|------|---------|
| 1 | `docker-compose.yml` | Add LangFuse services (langfuse-postgres + langfuse-server) |
| 2 | `requirements.txt` | Add langgraph, langfuse, peft, bitsandbytes, trl, accelerate, datasets |
| 3 | `src/config/settings.py` | Add ~20 new settings fields across 3 sections |
| 4 | `src/agentic/observability/tracer.py` | Wire langfuse/langsmith export stubs |
| 5 | `src/agentic/observability/__init__.py` | Export LangFuseExporter, EvaluationHooks |
| 6 | `src/agentic/orchestrator.py` | Swap to LangGraph agent, add evaluation hooks |
| 7 | `src/agentic/agents/__init__.py` | Export LangGraphReActAgent |
| 8 | `src/fine_tuning/trainer.py` | Add LOCAL provider dispatch, extend TrainingConfig |
| 9 | `src/fine_tuning/data_preparation.py` | Add ShareGPT format support |

**Total: 9 new files + 9 modified files = 18 file operations**

---

## 6. Dependencies & Environment

### New Packages (requirements.txt)

```
# LangFuse Observability
langfuse>=2.40.0

# LangGraph Agent Orchestration
langgraph>=0.2.0
langgraph-checkpoint>=2.0.0
langgraph-checkpoint-sqlite>=2.0.0

# QLoRA Fine-tuning Pipeline
peft>=0.12.0
bitsandbytes>=0.43.0
trl>=0.9.0
accelerate>=0.30.0
datasets>=2.20.0
```

### Environment Variables (.env additions)

```bash
# === LangFuse Observability ===
OBSERVABILITY_BACKEND=langfuse              # "memory", "langfuse", "langsmith"
LANGFUSE_HOST=http://localhost:3002
LANGFUSE_PUBLIC_KEY=pk-lf-...               # Generated in LangFuse UI
LANGFUSE_SECRET_KEY=sk-lf-...               # Generated in LangFuse UI
LANGFUSE_POSTGRES_PASSWORD=langfuse_secret
LANGFUSE_NEXTAUTH_SECRET=langfuse-next-auth-secret
LANGFUSE_SALT=langfuse-salt-value
ENABLE_COST_TRACKING=true
ENABLE_HALLUCINATION_DETECTION=true

# === LangSmith (alternative to LangFuse) ===
# LANGSMITH_API_KEY=ls-...
# LANGSMITH_PROJECT=jd_jones_rag

# === LangGraph Agent ===
USE_LANGGRAPH_AGENT=false                   # Set true to use LangGraph instead of basic ReAct
AGENT_HUMAN_IN_LOOP=false
LANGGRAPH_CHECKPOINT_BACKEND=sqlite
LANGGRAPH_CHECKPOINT_PATH=./data/langgraph_checkpoints.db

# === QLoRA Fine-tuning ===
FINE_TUNING_OUTPUT_DIR=./fine_tuning_output
FINE_TUNING_BASE_MODEL=unsloth/llama-3.2-3b-instruct-bnb-4bit
FINE_TUNING_METHOD=qlora
FINE_TUNING_QUANTIZATION_BITS=4
FINE_TUNING_LORA_RANK=64
FINE_TUNING_LORA_ALPHA=16
FINE_TUNING_LORA_DROPOUT=0.0
FINE_TUNING_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
GGUF_QUANTIZATION_TYPE=q4_k_m
OLLAMA_MODEL_NAME=jdjones-llama3.2
```

---

## 7. Testing & Verification

### Phase 1 Verification (LangFuse)
1. `docker compose up langfuse-postgres langfuse-server` — Verify UI at localhost:3002
2. Create API keys in LangFuse UI, add to .env
3. Set `OBSERVABILITY_BACKEND=langfuse`
4. Run a test query through the API
5. Verify traces appear with spans, generations, and evaluation scores

### Phase 2 Verification (LangGraph)
1. Set `USE_LANGGRAPH_AGENT=true`
2. Run a test query — verify `ReActResult` returned with identical fields
3. Call `get_graph_visualization()` — verify Mermaid diagram renders
4. Call `get_state_history(thread_id)` — verify checkpoints exist
5. Test `replay_from_checkpoint()` for time-travel debugging
6. If HITL enabled: verify execution pauses at `tool_executor` node

### Phase 3 Verification (QLoRA)
1. Prepare test dataset: `RAGDataGenerator.add_domain_seed_data()`
2. Run training (requires GPU): `QLoRATrainer(config).train(dataset)`
3. Verify adapter saved to `output_dir/adapter/`
4. Verify merged model saved to `output_dir/merged/`
5. Run GGUF export: `GGUFExporter(config).export()` (requires llama.cpp)
6. Verify Ollama model created: `ollama list` shows `jdjones-llama3.2`
7. Test inference: `ollama run jdjones-llama3.2 "What is API 622?"`

---

## 8. Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Observability** | Week 1 | LangFuse Docker + exporter + evaluation hooks + LangSmith fallback |
| **Phase 2: LangGraph** | Week 2-3 | LangGraph ReAct agent, debugging features, HITL, Studio config |
| **Phase 3: Fine-Tuning** | Week 3-4 | QLoRA trainer, GGUF exporter, RAG data generator, Ollama integration |
| **Phase 4: Testing** | Week 4-5 | Integration tests, documentation, verification |

---

**End of Unified Implementation Plan**

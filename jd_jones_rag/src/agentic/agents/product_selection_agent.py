"""
Product Selection Agent
Guides customers through product selection with targeted questions.
GUIDED approach - not open-ended chatbot.

This implements the "Product Selection Assistant" requirement:
- Recognizes missing parameters
- Iterates with user (multi-turn dialogue)
- Verifies recommendations against safety standards (API 622/624)
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings


class SelectionStage(str, Enum):
    """Stages of the product selection process."""
    INDUSTRY_IDENTIFICATION = "industry_identification"
    APPLICATION_TYPE = "application_type"
    OPERATING_CONDITIONS = "operating_conditions"
    MEDIA_IDENTIFICATION = "media_identification"
    COMPLIANCE_REQUIREMENTS = "compliance_requirements"
    SIZE_SPECIFICATION = "size_specification"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    VALIDATION = "validation"
    COMPLETE = "complete"


class ParameterPriority(str, Enum):
    """Priority of parameters."""
    REQUIRED = "required"      # Must have before recommendation
    IMPORTANT = "important"    # Strongly recommended
    OPTIONAL = "optional"      # Nice to have


@dataclass
class SelectionParameter:
    """A parameter in the selection process."""
    name: str
    display_name: str
    description: str
    priority: ParameterPriority
    stage: SelectionStage
    value_type: str  # text, number, select, multi_select
    options: List[str] = field(default_factory=list)
    validation_pattern: Optional[str] = None
    help_text: str = ""
    depends_on: Optional[str] = None  # Parameter this depends on
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "priority": self.priority.value,
            "stage": self.stage.value,
            "value_type": self.value_type,
            "options": self.options,
            "help_text": self.help_text
        }


@dataclass
class SelectionState:
    """Current state of product selection."""
    session_id: str
    current_stage: SelectionStage
    collected_parameters: Dict[str, Any]
    missing_required: List[str]
    missing_important: List[str]
    current_question: Optional[str] = None
    current_options: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "current_stage": self.current_stage.value,
            "collected_parameters": self.collected_parameters,
            "missing_required": self.missing_required,
            "missing_important": self.missing_important,
            "current_question": self.current_question,
            "current_options": self.current_options,
            "recommendations": self.recommendations,
            "validation_results": self.validation_results,
            "progress_percent": self._calculate_progress()
        }
    
    def _calculate_progress(self) -> int:
        """Calculate selection progress percentage."""
        stages = list(SelectionStage)
        current_idx = stages.index(self.current_stage)
        return int((current_idx / len(stages)) * 100)


@dataclass
class SelectionResponse:
    """Response from the selection agent."""
    state: SelectionState
    message: str
    question: Optional[str] = None
    options: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    is_complete: bool = False
    requires_input: bool = True
    validation_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "message": self.message,
            "question": self.question,
            "options": self.options,
            "recommendations": self.recommendations,
            "is_complete": self.is_complete,
            "requires_input": self.requires_input,
            "validation_warnings": self.validation_warnings
        }


class ProductSelectionAgent:
    """
    Guided product selection agent.
    
    Key Principles:
    1. GUIDED experience - not open-ended
    2. Ask ONE question at a time
    3. Provide options where possible
    4. Validate recommendations against standards
    5. Multi-turn dialogue with state management
    """
    
    # Define the selection parameters
    PARAMETERS = [
        # Stage 1: Industry
        SelectionParameter(
            name="industry",
            display_name="Industry",
            description="Which industry is this application for?",
            priority=ParameterPriority.REQUIRED,
            stage=SelectionStage.INDUSTRY_IDENTIFICATION,
            value_type="select",
            options=[
                "Oil & Gas (Upstream)",
                "Oil & Gas (Downstream/Refinery)",
                "Petrochemical",
                "Chemical Processing",
                "Pharmaceutical",
                "Food & Beverage",
                "Power Generation",
                "Pulp & Paper",
                "Water/Wastewater",
                "Marine",
                "Other"
            ],
            help_text="Select the primary industry for this application"
        ),
        
        # Stage 2: Application Type
        SelectionParameter(
            name="equipment_type",
            display_name="Equipment Type",
            description="What type of equipment needs sealing?",
            priority=ParameterPriority.REQUIRED,
            stage=SelectionStage.APPLICATION_TYPE,
            value_type="select",
            options=[
                "Valve (Gate/Globe/Ball)",
                "Pump",
                "Compressor",
                "Flange/Pipe Joint",
                "Heat Exchanger",
                "Reactor/Vessel",
                "Agitator/Mixer",
                "Expansion Joint",
                "Other"
            ]
        ),
        SelectionParameter(
            name="sealing_type",
            display_name="Sealing Type Needed",
            description="What type of sealing solution are you looking for?",
            priority=ParameterPriority.REQUIRED,
            stage=SelectionStage.APPLICATION_TYPE,
            value_type="select",
            options=[
                "Gasket (Static Seal)",
                "Packing (Dynamic Seal)",
                "Expansion Joint",
                "O-Ring/Seal",
                "Not Sure - Need Guidance"
            ]
        ),
        
        # Stage 3: Operating Conditions
        SelectionParameter(
            name="max_temperature",
            display_name="Maximum Operating Temperature",
            description="What is the maximum temperature in °C?",
            priority=ParameterPriority.REQUIRED,
            stage=SelectionStage.OPERATING_CONDITIONS,
            value_type="select",
            options=[
                "Below -40°C (Cryogenic)",
                "-40 to 0°C (Sub-zero)",
                "0 to 100°C (Ambient)",
                "100 to 250°C (Moderate)",
                "250 to 400°C (High)",
                "400 to 600°C (Very High)",
                "Above 600°C (Extreme)"
            ],
            help_text="Include any temperature excursions or cycling"
        ),
        SelectionParameter(
            name="max_pressure",
            display_name="Maximum Operating Pressure",
            description="What is the maximum pressure?",
            priority=ParameterPriority.REQUIRED,
            stage=SelectionStage.OPERATING_CONDITIONS,
            value_type="select",
            options=[
                "Low (<10 bar / 150 psi)",
                "Medium (10-50 bar / 150-750 psi)",
                "High (50-150 bar / 750-2200 psi)",
                "Very High (150-400 bar / 2200-5800 psi)",
                "Ultra High (>400 bar / >5800 psi)"
            ]
        ),
        
        # Stage 4: Media
        SelectionParameter(
            name="media_type",
            display_name="Process Media",
            description="What fluid/media will be in contact with the seal?",
            priority=ParameterPriority.REQUIRED,
            stage=SelectionStage.MEDIA_IDENTIFICATION,
            value_type="select",
            options=[
                "Hydrocarbons (Oil, Gas, Fuel)",
                "Steam",
                "Water/Treated Water",
                "Acids",
                "Alkalis/Caustics",
                "Solvents",
                "Gases (Inert)",
                "Gases (Corrosive - H2S, CO2)",
                "Slurries/Abrasive Media",
                "Food/Beverage Products",
                "Pharmaceutical/Sterile",
                "Other (Please specify)"
            ]
        ),
        SelectionParameter(
            name="media_details",
            display_name="Media Details",
            description="Please provide specific media name or composition if known",
            priority=ParameterPriority.IMPORTANT,
            stage=SelectionStage.MEDIA_IDENTIFICATION,
            value_type="text",
            help_text="E.g., 'Crude oil with 5% H2S' or 'Concentrated sulfuric acid'"
        ),
        
        # Stage 5: Compliance
        SelectionParameter(
            name="certifications_required",
            display_name="Required Certifications",
            description="Which certifications/standards must the product meet?",
            priority=ParameterPriority.IMPORTANT,
            stage=SelectionStage.COMPLIANCE_REQUIREMENTS,
            value_type="multi_select",
            options=[
                "API 622 (Fugitive Emissions - Packing)",
                "API 624 (Fugitive Emissions - Valves)",
                "Shell SPE 77/312",
                "Saudi Aramco SAES",
                "FDA Food Contact",
                "USP Class VI",
                "3A Sanitary",
                "ASME",
                "PED (Pressure Equipment Directive)",
                "Fire-Safe (API 607/ISO 10497)",
                "None Specific",
                "Not Sure"
            ]
        ),
        
        # Stage 6: Size
        SelectionParameter(
            name="size",
            display_name="Size/Dimensions",
            description="What size is required?",
            priority=ParameterPriority.IMPORTANT,
            stage=SelectionStage.SIZE_SPECIFICATION,
            value_type="text",
            help_text="E.g., 'DN100', '4 inch', or specific dimensions in mm"
        ),
        SelectionParameter(
            name="quantity",
            display_name="Quantity Required",
            description="How many units do you need?",
            priority=ParameterPriority.OPTIONAL,
            stage=SelectionStage.SIZE_SPECIFICATION,
            value_type="number",
            help_text="For pricing estimation"
        )
    ]
    
    # Product database - loaded dynamically from JSON files via JDJonesDataLoader
    # No longer hardcoded - see get_products_fallback() method
    PRODUCTS = {}  # Populated on first access via _ensure_products_loaded()
    
    @classmethod
    def _ensure_products_loaded(cls):
        """Load products from JSON files if not already loaded."""
        if cls.PRODUCTS:
            return  # Already loaded
            
        try:
            from src.data_ingestion.jd_jones_data_loader import get_data_loader
            loader = get_data_loader()
            products = loader.get_all_products()
            
            # Transform to expected format - ProductData is a dataclass, not dict
            for code, prod in products.items():
                # Access dataclass attributes directly
                specs = getattr(prod, 'specs', None) or {}
                if hasattr(specs, '__dict__'):
                    specs = specs.__dict__
                elif not isinstance(specs, dict):
                    specs = {}
                
                cls.PRODUCTS[code] = {
                    "name": getattr(prod, 'name', '') or '',
                    "type": "packing",
                    "category": getattr(prod, 'category', '') or '',
                    "max_temp": specs.get('temperature_max') if isinstance(specs, dict) else getattr(specs, 'temperature_max', None),
                    "min_temp": specs.get('temperature_min') if isinstance(specs, dict) else getattr(specs, 'temperature_min', None),
                    "max_pressure": specs.get('pressure_static') if isinstance(specs, dict) else getattr(specs, 'pressure_static', None),
                    "pressure_rotary": specs.get('pressure_rotary') if isinstance(specs, dict) else getattr(specs, 'pressure_rotary', None),
                    "pressure_reciprocating": specs.get('pressure_reciprocating') if isinstance(specs, dict) else getattr(specs, 'pressure_reciprocating', None),
                    "shaft_speed": specs.get('shaft_speed_rotary') if isinstance(specs, dict) else getattr(specs, 'shaft_speed_rotary', None),
                    "media": getattr(prod, 'service_media', []) or [],
                    "certifications": getattr(prod, 'certifications', []) or [],
                    "industries": getattr(prod, 'industries', []) or [],
                    "applications": getattr(prod, 'applications', []) or [],
                    "url": getattr(prod, 'source_url', '') or '',
                }
                
            import logging
            logging.info(f"Loaded {len(cls.PRODUCTS)} products from JSON files")
            
        except ImportError:
            import logging
            logging.warning("JDJonesDataLoader not available, PRODUCTS will remain empty")
        except Exception as e:
            import logging
            logging.error(f"Error loading products from JSON: {e}")


    def __init__(self):
        """Initialize product selection agent."""
        # Ensure products are loaded from JSON files
        self._ensure_products_loaded()
        
        from src.config.settings import get_llm
        self.llm = get_llm(temperature=0.1)
        self.sessions: Dict[str, SelectionState] = {}
        
        # Initialize product catalog retriever for real product data
        try:
            from src.data_ingestion.product_catalog_retriever import get_product_retriever
            self.product_retriever = get_product_retriever()
            self.use_real_catalog = True
        except Exception as e:
            # Fallback to static products if retriever not available
            import logging
            logging.warning(f"ProductCatalogRetriever not available, using fallback: {e}")
            self.product_retriever = None
            self.use_real_catalog = False
    
    
    def start_selection(self, session_id: str) -> SelectionResponse:
        """Start a new product selection session."""
        state = SelectionState(
            session_id=session_id,
            current_stage=SelectionStage.INDUSTRY_IDENTIFICATION,
            collected_parameters={},
            missing_required=self._get_required_params(),
            missing_important=self._get_important_params()
        )
        
        self.sessions[session_id] = state
        
        # Get first question
        return self._generate_next_question(state)
    
    def process_input(
        self,
        session_id: str,
        user_input: str
    ) -> SelectionResponse:
        """
        Process user input and advance the selection.
        
        Args:
            session_id: Session identifier
            user_input: User's response to current question
            
        Returns:
            SelectionResponse with next question or recommendations
        """
        if session_id not in self.sessions:
            return self.start_selection(session_id)
        
        state = self.sessions[session_id]
        
        # Determine which parameter this input is for
        current_param = self._get_current_parameter(state)
        
        if current_param:
            # Validate and store input
            validated = self._validate_input(current_param, user_input)
            if validated["valid"]:
                state.collected_parameters[current_param.name] = validated["value"]
                
                # Remove from missing lists
                if current_param.name in state.missing_required:
                    state.missing_required.remove(current_param.name)
                if current_param.name in state.missing_important:
                    state.missing_important.remove(current_param.name)
        
        # Check if we can make recommendations
        if self._can_recommend(state):
            return self._generate_recommendations(state)
        
        # Advance stage if needed
        self._advance_stage(state)
        
        # Generate next question
        return self._generate_next_question(state)
    
    def _get_required_params(self) -> List[str]:
        """Get list of required parameter names."""
        return [p.name for p in self.PARAMETERS if p.priority == ParameterPriority.REQUIRED]
    
    def _get_important_params(self) -> List[str]:
        """Get list of important parameter names."""
        return [p.name for p in self.PARAMETERS if p.priority == ParameterPriority.IMPORTANT]
    
    def _get_current_parameter(self, state: SelectionState) -> Optional[SelectionParameter]:
        """Get the parameter for current stage that needs input."""
        # First check required params
        for param in self.PARAMETERS:
            if param.name in state.missing_required and param.stage == state.current_stage:
                return param
        
        # Then important params for current stage
        for param in self.PARAMETERS:
            if param.name in state.missing_important and param.stage == state.current_stage:
                return param
        
        return None
    
    def _validate_input(
        self,
        param: SelectionParameter,
        user_input: str
    ) -> Dict[str, Any]:
        """Validate user input for a parameter."""
        user_input = user_input.strip()
        
        if param.value_type == "select":
            # Check if input matches an option (case-insensitive)
            for option in param.options:
                if user_input.lower() in option.lower() or option.lower() in user_input.lower():
                    return {"valid": True, "value": option}
            
            # Try to match by number
            try:
                idx = int(user_input) - 1
                if 0 <= idx < len(param.options):
                    return {"valid": True, "value": param.options[idx]}
            except ValueError:
                pass
            
            return {"valid": False, "error": f"Please select from the available options"}
        
        elif param.value_type == "multi_select":
            # Handle multiple selections
            selected = []
            for option in param.options:
                if option.lower() in user_input.lower():
                    selected.append(option)
            return {"valid": True, "value": selected if selected else ["None Specific"]}
        
        elif param.value_type == "number":
            try:
                return {"valid": True, "value": int(user_input)}
            except ValueError:
                return {"valid": False, "error": "Please enter a number"}
        
        else:  # text
            return {"valid": True, "value": user_input}
    
    def _can_recommend(self, state: SelectionState) -> bool:
        """Check if we have enough info to make recommendations."""
        # Must have all required parameters
        return len(state.missing_required) == 0
    
    def _advance_stage(self, state: SelectionState) -> None:
        """Advance to next stage if current stage is complete."""
        stages = list(SelectionStage)
        current_idx = stages.index(state.current_stage)
        
        # Check if current stage has any remaining required params
        current_stage_params = [
            p for p in self.PARAMETERS
            if p.stage == state.current_stage and p.name in state.missing_required
        ]
        
        if not current_stage_params and current_idx < len(stages) - 1:
            state.current_stage = stages[current_idx + 1]
    
    def _generate_next_question(self, state: SelectionState) -> SelectionResponse:
        """Generate the next question to ask."""
        param = self._get_current_parameter(state)
        
        if not param:
            # No more questions in current stage, try advancing
            self._advance_stage(state)
            param = self._get_current_parameter(state)
        
        if not param:
            # Can we recommend?
            if self._can_recommend(state):
                return self._generate_recommendations(state)
            else:
                # Move to next stage with missing params
                for p in self.PARAMETERS:
                    if p.name in state.missing_required:
                        state.current_stage = p.stage
                        param = p
                        break
        
        if param:
            state.current_question = param.description
            state.current_options = param.options if param.options else []
            
            message = self._format_question_message(param, state)
            
            return SelectionResponse(
                state=state,
                message=message,
                question=param.description,
                options=param.options if param.value_type in ["select", "multi_select"] else [],
                requires_input=True
            )
        
        # Fallback
        return SelectionResponse(
            state=state,
            message="I have all the information needed. Let me find the best products for you.",
            requires_input=False
        )
    
    def _format_question_message(
        self,
        param: SelectionParameter,
        state: SelectionState
    ) -> str:
        """Format a question message with context."""
        collected = state.collected_parameters
        
        # Build context summary
        context_parts = []
        if collected.get("industry"):
            context_parts.append(f"Industry: {collected['industry']}")
        if collected.get("equipment_type"):
            context_parts.append(f"Equipment: {collected['equipment_type']}")
        if collected.get("max_temperature"):
            context_parts.append(f"Temperature: {collected['max_temperature']}")
        
        context = " | ".join(context_parts) if context_parts else ""
        
        message = f"**{param.display_name}**\n\n{param.description}"
        
        if param.help_text:
            message += f"\n\n*{param.help_text}*"
        
        if context:
            message = f"[{context}]\n\n{message}"
        
        if param.options:
            message += "\n\nOptions:\n"
            for i, opt in enumerate(param.options, 1):
                message += f"{i}. {opt}\n"
        
        return message
    
    def _generate_recommendations(self, state: SelectionState) -> SelectionResponse:
        """Generate product recommendations based on collected parameters."""
        params = state.collected_parameters
        
        scored_products = []
        
        # Try to use real product catalog if available
        if self.use_real_catalog and self.product_retriever:
            try:
                # Use the product catalog retriever for real data
                real_recommendations = self.product_retriever.get_recommendations_for_selection(params)
                
                for rec in real_recommendations:
                    scored_products.append({
                        "product_id": rec.get("product_code", ""),
                        "product_name": rec.get("product_name", rec.get("product_code", "")),
                        "type": rec.get("category", "packing"),
                        "score": rec.get("score", 50),
                        "specifications": rec.get("specifications", {}),
                        "warnings": [],
                        "match_reasons": rec.get("match_reasons", []),
                        "confidence": rec.get("confidence", "medium"),
                        "certifications": rec.get("certifications", []),
                        "applications": rec.get("applications", []),
                        "source_url": rec.get("source_url", ""),
                        "features": rec.get("features", []),
                        "material": rec.get("material", ""),
                        "industries": rec.get("industries", []),
                        "recommendation_text": rec.get("recommendation_text", ""),
                    })
                
            except Exception as e:
                import logging
                logging.warning(f"Error fetching from catalog, falling back to static: {e}")
                # Fall through to static products
        
        # Fallback to static products if nothing from catalog
        if not scored_products:
            for prod_id, product in self.PRODUCTS.items():
                score, warnings = self._score_product(product, params)
                if score > 0:
                    scored_products.append({
                        "product_id": prod_id,
                        "product_name": product["name"],
                        "type": product["type"],
                        "score": score,
                        "specifications": {
                            "max_temperature": f"{product.get('max_temp', 'N/A')}°C",
                            "max_pressure": f"{product.get('max_pressure', 'N/A')} bar",
                            "certifications": product.get("certifications", [])
                        },
                        "warnings": warnings,
                        "match_reasons": self._get_match_reasons(product, params)
                    })
        
        # Sort by score
        scored_products.sort(key=lambda x: x["score"], reverse=True)
        
        state.recommendations = scored_products[:5]  # Top 5 recommendations
        state.current_stage = SelectionStage.COMPLETE
        
        # Collect validation warnings
        all_warnings = []
        for prod in scored_products[:5]:
            all_warnings.extend(prod.get("warnings", []))
        
        message = self._format_recommendations_message(scored_products[:5], params)
        
        return SelectionResponse(
            state=state,
            message=message,
            recommendations=scored_products[:5],
            is_complete=True,
            requires_input=False,
            validation_warnings=list(set(all_warnings))
        )
    
    def _score_product(
        self,
        product: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Score a product against requirements using weighted multi-signal scoring.
        
        Returns a (score, warnings) tuple where score is 0-100.
        Uses additive scoring across independent dimensions so that a product
        matching 5 criteria always outranks one matching only 2.
        """
        # Weights mirror the retriever's scoring
        W_TEMP = 25
        W_PRESSURE = 20
        W_CERTIFICATION = 15
        W_INDUSTRY = 15
        W_APPLICATION = 15
        W_MEDIA = 10
        MAX_RAW = W_TEMP + W_PRESSURE + W_CERTIFICATION + W_INDUSTRY + W_APPLICATION + W_MEDIA
        
        raw_score = 0.0
        warnings = []
        
        # ------ Temperature ------
        temp_str = params.get("temperature", params.get("max_temperature", ""))
        required_temp = self._parse_temp_range(str(temp_str)) if temp_str else None
        
        max_temp = product.get("max_temp")
        min_temp = product.get("min_temp")
        
        if required_temp is not None and max_temp is not None:
            if max_temp >= required_temp:
                raw_score += W_TEMP
                # Tight margin warning
                if max_temp < required_temp * 1.15:
                    warnings.append("Temperature margin is tight - consider higher rated option")
            elif max_temp >= required_temp * 0.85:
                raw_score += W_TEMP * 0.3  # Marginal
                warnings.append(f"Max temp ({max_temp}°C) is below requirement ({required_temp}°C)")
            else:
                return 0, ["Temperature far below requirement"]  # Disqualify
        elif required_temp is None:
            raw_score += W_TEMP * 0.3  # No temp constraint -> partial credit
        
        # ------ Pressure ------
        pressure_str = params.get("pressure", params.get("max_pressure", ""))
        required_pressure = self._parse_pressure_range(str(pressure_str)) if pressure_str else None
        
        max_pressure = product.get("max_pressure") or 0
        p_rotary = product.get("pressure_rotary") or 0
        p_recip = product.get("pressure_reciprocating") or 0
        effective_pressure = max(max_pressure, p_rotary, p_recip)
        
        if required_pressure is not None and effective_pressure > 0:
            if effective_pressure >= required_pressure:
                raw_score += W_PRESSURE
            elif effective_pressure >= required_pressure * 0.7:
                raw_score += W_PRESSURE * 0.4
                warnings.append(f"Pressure rating ({effective_pressure} bar) may be marginal")
            else:
                raw_score -= W_PRESSURE * 0.3  # Penalty
        elif required_pressure is None:
            raw_score += W_PRESSURE * 0.3  # No pressure constraint
        
        # ------ Certifications ------
        required_certs = params.get("certifications_required", [])
        if isinstance(required_certs, list) and required_certs:
            cert_matches = 0
            cert_total = 0
            for cert in required_certs:
                cert_clean = str(cert).split("(")[0].strip().lower()
                if cert_clean in ("none", "none specific", "not sure", "none required"):
                    raw_score += W_CERTIFICATION * 0.3
                    break
                cert_total += 1
                if any(cert_clean in pc.lower() for pc in product.get("certifications", [])):
                    cert_matches += 1
            if cert_total > 0:
                raw_score += W_CERTIFICATION * (cert_matches / cert_total)
        else:
            raw_score += W_CERTIFICATION * 0.3  # No cert constraint
        
        # ------ Industry ------
        industry = params.get("industry", "").lower()
        if industry:
            ind_words = set(industry.replace('&', ' ').replace('/', ' ').split())
            product_industries = [i.lower() for i in product.get("industries", [])]
            matched = False
            for pi in product_industries:
                pi_words = set(pi.replace('&', ' ').replace('/', ' ').split())
                if ind_words & pi_words:
                    raw_score += W_INDUSTRY
                    matched = True
                    break
            if not matched:
                raw_score += W_INDUSTRY * 0.15  # Minimal credit
        else:
            raw_score += W_INDUSTRY * 0.3
        
        # ------ Application / Equipment ------
        app_type = params.get("application_type", params.get("equipment_type", "")).lower()
        if app_type:
            app_words = set(app_type.replace('/', ' ').replace('(', ' ').replace(')', ' ').split())
            product_apps = [a.lower() for a in product.get("applications", [])]
            matched = False
            for pa in product_apps:
                pa_words = set(pa.replace('/', ' ').split())
                if app_words & pa_words:
                    raw_score += W_APPLICATION
                    matched = True
                    break
            if not matched:
                # Check product type
                if any(w in product.get("type", "").lower() for w in app_words):
                    raw_score += W_APPLICATION * 0.5
                else:
                    raw_score += W_APPLICATION * 0.1
        else:
            raw_score += W_APPLICATION * 0.3
        
        # ------ Media ------
        media = params.get("media", params.get("media_type", "")).lower()
        if media:
            product_media = " ".join(
                product.get("media", []) + product.get("applications", [])
            ).lower()
            if media in product_media or any(m in media for m in product.get("media", [])):
                raw_score += W_MEDIA
            else:
                raw_score += W_MEDIA * 0.1
        else:
            raw_score += W_MEDIA * 0.3
        
        # ------ Normalize to 0-100 ------
        final_score = max(0, min(100, (raw_score / MAX_RAW) * 100))
        
        return round(final_score, 1), warnings
    
    def _parse_temp_range(self, temp_str: str) -> Optional[int]:
        """Parse temperature from selection string."""
        import re
        if "Above 600" in temp_str or "Extreme" in temp_str:
            return 600
        if "400 to 600" in temp_str or "Very High" in temp_str:
            return 500
        if "250 to 400" in temp_str:
            return 350
        if "100 to 250" in temp_str:
            return 200
        if "0 to 100" in temp_str:
            return 80
        if "-40 to 0" in temp_str:
            return 0
        if "Below -40" in temp_str:
            return -40
        return None
    
    def _parse_pressure_range(self, pressure_str: str) -> Optional[int]:
        """Parse pressure from selection string."""
        if "Ultra High" in pressure_str or ">400" in pressure_str:
            return 450
        if "Very High" in pressure_str or "150-400" in pressure_str:
            return 300
        if "High" in pressure_str or "50-150" in pressure_str:
            return 100
        if "Medium" in pressure_str or "10-50" in pressure_str:
            return 40
        if "Low" in pressure_str:
            return 10
        return None
    
    def _get_match_reasons(
        self,
        product: Dict[str, Any],
        params: Dict[str, Any]
    ) -> List[str]:
        """Get reasons why product matches."""
        reasons = []
        
        if params.get("max_temperature"):
            reasons.append(f"Rated to {product['max_temp']}°C")
        if params.get("max_pressure"):
            reasons.append(f"Rated to {product['max_pressure']} bar")
        if product["certifications"]:
            reasons.append(f"Certified: {', '.join(product['certifications'][:2])}")
        
        return reasons
    
    def _format_recommendations_message(
        self,
        products: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> str:
        """Format recommendations as a message."""
        if not products:
            return """Based on your requirements, we could not find an exact match in our standard product range.

Please contact our technical team for a custom solution:
- Email: technical@jdjones.com
- Phone: 1-800-JD-JONES"""
        
        message = "## Recommended Products\n\n"
        message += f"Based on your requirements ({params.get('industry', 'N/A')}, "
        message += f"{params.get('max_temperature', 'N/A')}, {params.get('max_pressure', 'N/A')}), "
        message += "we recommend:\n\n"
        
        for i, prod in enumerate(products, 1):
            message += f"### {i}. {prod['product_name']}\n"
            message += f"- **Match Score**: {prod['score']}/10\n"
            message += f"- **Type**: {prod['type'].title()}\n"
            message += f"- **Max Temperature**: {prod['specifications']['max_temperature']}\n"
            message += f"- **Max Pressure**: {prod['specifications']['max_pressure']}\n"
            message += f"- **Certifications**: {', '.join(prod['specifications']['certifications'])}\n"
            
            if prod.get("warnings"):
                message += f"- **Note**: {'; '.join(prod['warnings'])}\n"
            
            message += "\n"
        
        message += "\n---\n"
        message += "**Next Steps:**\n"
        message += "1. Request a formal quotation\n"
        message += "2. Request technical datasheet\n"
        message += "3. Speak with a technical specialist\n"
        
        return message
    
    def get_session(self, session_id: str) -> Optional[SelectionState]:
        """Get a session state."""
        return self.sessions.get(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_recommendations_from_answers(self, answers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get product recommendations directly from structured answers.
        Used by the wizard frontend that collects all answers at once.
        
        Args:
            answers: Dict with keys like 'application_type', 'industry', 'temperature_range', etc.
            
        Returns:
            List of product recommendation dicts
        """
        # Map wizard answers to internal parameter names expected by get_recommendations_for_selection
        # Parse temperature from wizard options like "High Temperature (200°C to 400°C)"
        import re
        temp_range = answers.get('temperature_range', '') or answers.get('max_temperature', '')
        operating_temp = None
        if temp_range:
            # Extract max temperature from ranges like "200°C to 400°C" or "(200-400C)"
            temp_match = re.search(r'(\d+)\s*°?C?\s*(?:to|\-)\s*(\d+)', temp_range)
            if temp_match:
                operating_temp = float(temp_match.group(2))  # Use max temp
            elif 'cryogenic' in temp_range.lower() or 'below -40' in temp_range.lower():
                operating_temp = -40.0
            elif '-40 to 0' in temp_range.lower() or 'sub-zero' in temp_range.lower():
                operating_temp = 0.0
            elif 'above 600' in temp_range.lower() or 'extreme' in temp_range.lower():
                operating_temp = 650.0
            elif 'very high' in temp_range.lower() or '400 to 600' in temp_range.lower():
                operating_temp = 500.0
            elif '250 to 400' in temp_range.lower() or ('high' in temp_range.lower() and 'very' not in temp_range.lower()):
                operating_temp = 350.0
            elif '100 to 250' in temp_range.lower() or 'moderate' in temp_range.lower():
                operating_temp = 200.0
            elif '0 to 100' in temp_range.lower() or 'ambient' in temp_range.lower():
                operating_temp = 80.0
        
        # Parse pressure from wizard options like "Medium (10-50 bar)"
        pressure_range = answers.get('pressure_range', '') or answers.get('max_pressure', '')
        operating_pressure = None
        if pressure_range:
            pressure_match = re.search(r'(\d+)\s*(?:\-|to)\s*(\d+)', pressure_range)
            if pressure_match:
                operating_pressure = float(pressure_match.group(2))  # Use max pressure
            elif 'ultra' in pressure_range.lower():
                operating_pressure = 450.0
            elif 'very high' in pressure_range.lower():
                operating_pressure = 300.0
            elif 'high' in pressure_range.lower() and 'very' not in pressure_range.lower():
                operating_pressure = 100.0
            elif 'medium' in pressure_range.lower():
                operating_pressure = 40.0
            elif 'low' in pressure_range.lower():
                operating_pressure = 10.0
        
        params = {
            'industry': answers.get('industry'),
            'application_type': answers.get('application_type'),
            'equipment_type': answers.get('equipment_type'),
            'sealing_type': answers.get('sealing_type'),
            'operating_temperature': operating_temp,
            'operating_pressure': operating_pressure,
            'media_type': answers.get('media_type'),
            'media_description': answers.get('media_details', ''),
            'certifications_required': answers.get('required_certifications', answers.get('certifications_required', [])),
            'material_preference': answers.get('material_preference'),
            # Also include keys expected by _score_product fallback
            'temperature': operating_temp,
            'pressure': operating_pressure,
            'media': answers.get('media_type'),
            'max_temperature': answers.get('temperature_range', '') or answers.get('max_temperature', ''),
            'max_pressure': answers.get('pressure_range', '') or answers.get('max_pressure', ''),
        }
        
        scored_products = []
        
        # Try to use real product catalog if available
        if self.use_real_catalog and self.product_retriever:
            try:
                real_recommendations = self.product_retriever.get_recommendations_for_selection(params)
                
                for rec in real_recommendations:
                    scored_products.append({
                        "product_id": rec.get("product_code", ""),
                        "product_name": rec.get("product_name", rec.get("product_code", "")),
                        "type": rec.get("category", "packing"),
                        "score": rec.get("score", 50),
                        "specifications": rec.get("specifications", {}),
                        "warnings": [],
                        "match_reasons": rec.get("match_reasons", []),
                        "confidence": rec.get("confidence", "medium"),
                        "certifications": rec.get("certifications", []),
                        "applications": rec.get("applications", []),
                        "source_url": rec.get("source_url", ""),
                        "features": rec.get("features", []),
                        "material": rec.get("material", ""),
                        "industries": rec.get("industries", []),
                        "recommendation_text": rec.get("recommendation_text", ""),
                    })
                
            except Exception as e:
                import logging
                logging.warning(f"Error fetching from catalog, falling back to static: {e}")
        
        # Fallback to static products if nothing from catalog
        if not scored_products:
            for prod_id, product in self.PRODUCTS.items():
                score, warnings = self._score_product(product, params)
                if score > 0:
                    scored_products.append({
                        "product_id": prod_id,
                        "product_name": product["name"],
                        "type": product["type"],
                        "score": score,
                        "specifications": {
                            "max_temperature": f"{product.get('max_temp', 'N/A')}°C",
                            "max_pressure": f"{product.get('max_pressure', 'N/A')} bar",
                            "certifications": product.get("certifications", [])
                        },
                        "warnings": warnings,
                        "match_reasons": self._get_match_reasons(product, params),
                        "certifications": product.get("certifications", []),
                        "applications": product.get("applications", [])
                    })
        
        # Sort by score and return top 5
        scored_products.sort(key=lambda x: x["score"], reverse=True)
        return scored_products[:5]


"""
Response Generator for External Customer Portal.
Generates personalized responses using RAG for customer queries.

Enhanced with:
- FAQ Prompt Cache for instant FAQ responses
- Embedding Cache for reduced API calls
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings
from src.knowledge_base.main_context import MainContextDatabase
from src.external_system.classifier import CustomerIntent, ClassificationResult

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates responses for customer queries using RAG and templates.
    Combines knowledge base retrieval with LLM generation.
    
    Enhanced with:
    - FAQ Prompt Cache for instant responses to common questions
    - Embedding Cache to reduce embedding API costs
    """
    
    SYSTEM_PROMPT = """You are a helpful customer service assistant for JD Jones Manufacturing.

You help customers with:
- Product information and specifications
- Pricing inquiries
- Order tracking assistance
- Technical support
- Returns and warranty questions

STRICT GUIDELINES:
1. Be friendly, professional, and helpful
2. ONLY provide information from the given context — NEVER fabricate specifications
3. If a product specification (temperature, pressure, pH, speed) is NOT in the context below, DO NOT provide one
4. If you don't have specific information, say: "I don't have that information available. Let me connect you with our technical team."
5. Keep responses concise but complete
6. Always include relevant next steps or options
7. NEVER use "typically", "usually", or "generally" for specific product specs
8. If asked to "check again" or "try harder" for missing data, DO NOT invent numbers — repeat what you know and offer to connect them with experts

PRODUCT SPECIFICATION PROTOCOL:
- Only quote specs that appear VERBATIM in the context below
- Temperature, pressure, pH, speed values MUST come from the context — not from your training data
- If context says "Temperature: -240°C to 650°C", quote exactly "-240°C to 650°C"
- For missing specs, say: "I recommend checking our product data sheet for exact specifications. Would you like me to arrange a callback from our technical team?"

CONTEXT FROM KNOWLEDGE BASE:
{context}

CUSTOMER INTENT: {intent}
EXTRACTED INFORMATION: {entities}

Respond naturally to the customer's query. Remember: it is better to say "I don't have that data" than to provide incorrect specifications."""
    
    def __init__(self):
        """Initialize response generator with caching."""
        from src.config.settings import get_llm
        # Dual strategy: Creative temperature (0.7) for intelligent, natural
        # customer responses + GroundingValidator enforces factual accuracy
        # post-generation. The validator is the real guard, not temperature.
        self.llm = get_llm(temperature=0.7)
        
        # Use hybrid retrieval with RRF and Query Expansion
        from src.retrieval.enhanced_retrieval import HybridRetrieval
        self.hybrid_retrieval = HybridRetrieval(
            vector_weight=0.4,
            keyword_weight=0.6,
            exact_match_boost=2.0,
            use_rrf=True,
            use_query_expansion=True
        )
        self.knowledge_base = MainContextDatabase()
        
        # Programmatic grounding validator (anti-hallucination)
        from src.retrieval.grounding_validator import get_grounding_validator
        self.grounding_validator = get_grounding_validator()
        
        # Initialize FAQ Prompt Cache for instant FAQ responses
        self.faq_cache = None
        try:
            from src.retrieval.faq_prompt_cache import FAQPromptCache
            self.faq_cache = FAQPromptCache()
            logger.info("FAQ Prompt Cache initialized")
        except ImportError as e:
            logger.warning(f"FAQ Prompt Cache not available: {e}")
        
        # Initialize Embedding Cache for reduced API calls
        self.embedding_cache = None
        try:
            from src.retrieval.embedding_cache import get_embedding_cache
            self.embedding_cache = get_embedding_cache()
            logger.info("Embedding Cache initialized")
        except ImportError as e:
            logger.warning(f"Embedding Cache not available: {e}")
        
        # Response templates for common scenarios
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load response templates."""
        return {
            "greeting": "Hello! Welcome to JD Jones. How can I help you today?",
            
            "no_info": """I don't have specific information about that in my knowledge base. 
Let me connect you with the right team who can help.

Would you like to:
- Request a callback from our team
- Send us a message
- Browse our product catalog""",
            
            "order_not_found": """I couldn't find an order with that number. Please double-check:
- Order numbers start with "ORD-"
- Check the confirmation email for your order number

Need help? Call 1-800-JD-JONES for immediate assistance.""",
            
            "after_hours": """Thank you for reaching out! Our team is currently unavailable.

**Business Hours:**
Monday - Friday: 8:00 AM - 6:00 PM EST
Saturday: 9:00 AM - 1:00 PM EST

Leave a message and we'll respond on the next business day.""",
            
            "transfer_sales": """I'll connect you with our sales team who can provide detailed pricing and quotes.

In the meantime, you can:
- Browse our product catalog
- Download our price list
- Request a call back at your convenience""",
            
            "technical_emergency": """For urgent technical issues:

**Emergency Hotline:** 1-800-JD-JONES (Option 3)
Available 24/7 for critical production issues.

Please have your:
- Product serial number
- Order number
- Description of the issue""",
        }
    
    async def generate_response(
        self,
        query: str,
        classification: ClassificationResult,
        session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response for a customer query.
        
        Enhanced with:
        - FAQ cache for instant responses to common questions
        - Multi-agent product search for grounded product recommendations
        
        Args:
            query: Customer's query text
            classification: Classification result with intent
            session_context: Previous conversation context including history
            
        Returns:
            Response dict with message and metadata
        """
        import asyncio
        import re
        
        # Step 0: Check FAQ cache first for instant response
        if self.faq_cache:
            cache_result = self.faq_cache.get(query)
            if cache_result and cache_result.get("confidence", 0) > 0.85:
                logger.debug(f"FAQ cache hit: {cache_result.get('category', 'unknown')}")
                return {
                    "response": cache_result["answer"],
                    "intent": classification.primary_intent.value,
                    "confidence": cache_result["confidence"],
                    "sources": [{"document": "FAQ Cache", "relevance": cache_result["confidence"]}],
                    "suggested_actions": self._get_suggested_actions(classification.primary_intent),
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit": True
                }
        
        # Step 1: Detect if this is a product-related query
        product_keywords = [
            'product', 'packing', 'seal', 'gasket', 'valve', 'pump', 'graphite',
            'ptfe', 'temperature', 'pressure', 'recommend', 'suggestion', 'code',
            'NA ', 'high temp', 'mining', 'refinery', 'steam', 'chemical'
        ]
        is_product_query = any(kw.lower() in query.lower() for kw in product_keywords)
        
        # Step 2: Run parallel searches - KB search + Product Catalog search
        async def search_knowledge_base():
            """Search general knowledge base."""
            return self.hybrid_retrieval.hybrid_search(
                query=query,
                n_results=5,
                include_public_only=True
            )
        
        async def search_product_catalog():
            """Search real product catalog with grounded data."""
            try:
                from src.data_ingestion.product_catalog_retriever import get_product_retriever
                retriever = get_product_retriever()
                
                # Step 1: Check if user is asking about a specific product code
                # Extract product codes from query (e.g., "NA 701", "NA715", "na-701")
                code_pattern = re.compile(r'\bNA\s*[-]?\s*(\d+[A-Z]*)\b', re.IGNORECASE)
                code_matches = code_pattern.findall(query)
                
                direct_results = []
                if code_matches:
                    for code_num in code_matches:
                        normalized_code = f"NA {code_num.upper()}"
                        details = retriever.get_product_details(normalized_code)
                        if details:
                            # Create a ProductMatch-like object for consistent handling
                            from src.data_ingestion.product_catalog_retriever import ProductMatch, MatchConfidence
                            product = retriever.catalog.get_product_by_code(normalized_code)
                            if product:
                                direct_results.append(ProductMatch(
                                    product=product,
                                    score=100.0,
                                    confidence=MatchConfidence.HIGH,
                                    match_reasons=[f"Direct code match: {normalized_code}"]
                                ))
                    
                    if direct_results:
                        logger.debug(f"Direct product code lookup found {len(direct_results)} products")
                        return direct_results
                
                # Step 2: Fall back to parametric search
                # Extract parameters from query
                temp_match = re.search(r'(\d+)\s*°?C|(\d+)\s*degrees?|high\s+temp|extreme\s+temp', query.lower())
                operating_temp = None
                if temp_match:
                    if temp_match.group(1):
                        operating_temp = float(temp_match.group(1))
                    elif temp_match.group(2):
                        operating_temp = float(temp_match.group(2))
                    elif 'high' in query.lower() or 'extreme' in query.lower():
                        operating_temp = 350.0  # Default high temp
                
                pressure_match = re.search(r'(\d+)\s*bar|high\s+pressure', query.lower())
                operating_pressure = None
                if pressure_match:
                    if pressure_match.group(1):
                        operating_pressure = float(pressure_match.group(1))
                    elif 'high pressure' in query.lower():
                        operating_pressure = 150.0  # Default high pressure
                
                # Detect industry
                industry = None
                if 'mining' in query.lower() or 'coal' in query.lower():
                    industry = "mining"
                elif 'refinery' in query.lower() or 'oil' in query.lower():
                    industry = "refinery"
                elif 'chemical' in query.lower() or 'petrochemical' in query.lower():
                    industry = "petrochemical"
                elif 'power' in query.lower() or 'plant' in query.lower():
                    industry = "power_plant"
                
                # Detect application
                application = None
                if 'valve' in query.lower():
                    application = "valve"
                elif 'pump' in query.lower():
                    application = "pump"
                elif 'agitator' in query.lower() or 'mixer' in query.lower():
                    application = "agitator"
                
                # Search with extracted parameters
                matches = retriever.find_products(
                    industry=industry,
                    application=application,
                    operating_temp=operating_temp,
                    operating_pressure=operating_pressure,
                    limit=5
                )
                
                return matches
            except Exception as e:
                logger.warning(f"Product catalog search failed: {e}")
                return []
        
        # Run both searches in parallel
        if is_product_query:
            kb_task = asyncio.create_task(asyncio.to_thread(lambda: self.hybrid_retrieval.hybrid_search(query=query, n_results=5, include_public_only=True)))
            product_task = asyncio.create_task(search_product_catalog())
            
            kb_results, product_matches = await asyncio.gather(kb_task, product_task)
        else:
            kb_results = self.hybrid_retrieval.hybrid_search(
                query=query,
                n_results=5,
                include_public_only=True
            )
            product_matches = []
        
        # Step 3: Format context for LLM including real product data
        context = self._format_context(kb_results)
        
        # Add grounded product data to context
        product_context = ""
        if product_matches:
            product_context = "\n\n=== REAL PRODUCT DATA FROM CATALOG (USE ONLY THESE PRODUCT CODES) ===\n"
            for i, match in enumerate(product_matches[:5], 1):
                product = match.product
                specs = product.specs
                product_context += f"""
Product #{i}: {product.code} - {product.name}
- Category: {product.category}
- Material: {product.material}
- Description: {product.description[:200]}...
- Temperature Range: {specs.temperature_min}°C to {specs.temperature_max}°C
- Max Pressure: {max(specs.pressure_static or 0, specs.pressure_rotary or 0, specs.pressure_reciprocating or 0)} bar
- Certifications: {', '.join(product.certifications) if product.certifications else 'N/A'}
- Applications: {', '.join(product.applications[:3]) if product.applications else 'N/A'}
- Match Score: {match.score}
- Why Recommended: {', '.join(match.match_reasons)}
"""
            product_context += "\nCRITICAL: Only recommend products from the above list. Never make up product codes.\n"
        
        # Format entities
        entities_str = ", ".join([
            f"{k}: {v}" for k, v in classification.entities.items()
        ]) if classification.entities else "None"
        
        # Build conversation history context for the prompt
        conversation_context = ""
        if session_context and session_context.get("conversation_history"):
            history = session_context["conversation_history"]
            if history:
                conversation_context = "\n\nCONVERSATION HISTORY (use this to understand follow-up questions):\n"
                for msg in history[-6:]:  # Last 6 messages
                    role = "Customer" if msg["role"] == "user" else "Assistant"
                    conversation_context += f"{role}: {msg['content']}\n"
        
        # Build prompt with conversation context
        
        # PRE-GENERATION: Grounding sufficiency check
        is_sufficient, fallback = self.grounding_validator.check_context_sufficiency(
            context=context + product_context,
            product_matches=product_matches,
            query=query
        )
        
        if not is_sufficient and fallback:
            # Short-circuit: don't invoke the LLM if we have no data
            response_text = fallback
        else:
            system_content = self.SYSTEM_PROMPT.format(
                context=context + product_context,
                intent=classification.primary_intent.value,
                entities=entities_str
            )
            
            # Add conversation context instruction
            if conversation_context:
                system_content += conversation_context
                system_content += "\nIMPORTANT: The customer's current message may be a follow-up or reference to the conversation above. Interpret their intent based on the full conversation context."
            
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=query)
            ]
            
            # Generate response
            try:
                response = await self.llm.ainvoke(messages)
                raw_response = response.content
            except Exception as e:
                # Fallback to template response
                raw_response = self._get_fallback_response(classification.primary_intent)
            
            # POST-GENERATION: Grounding validation
            grounding_result = self.grounding_validator.validate_response(
                response=raw_response,
                context=context + product_context,
                product_matches=product_matches,
                query=query
            )
            response_text = grounding_result.validated_response
        
        # Extract sources
        sources = [
            {
                "document": r.metadata.get("file_name", "Knowledge Base"),
                "relevance": round(r.relevance_score, 2)
            }
            for r in kb_results[:3]
            if r.relevance_score > 0.7
        ]
        
        # Add product catalog as source if products were found
        if product_matches:
            sources.append({
                "document": "Product Catalog (Grounded)",
                "relevance": 1.0,
                "products_found": len(product_matches)
            })
        
        return {
            "response": response_text,
            "intent": classification.primary_intent.value,
            "confidence": classification.confidence,
            "sources": sources,
            "suggested_actions": self._get_suggested_actions(classification.primary_intent),
            "timestamp": datetime.now().isoformat(),
            "products_searched": len(product_matches) if product_matches else 0
        }
    
    def _format_context(self, results: List[Any]) -> str:
        """Format knowledge base results for context."""
        if not results:
            return "No specific information available in the knowledge base."
        
        context_parts = []
        for result in results[:8]:  # Include more results
            # Lowered threshold from 0.6 to 0.3 to avoid discarding
            # marginally-relevant chunks that may contain useful product info
            if result.relevance_score > 0.3:
                context_parts.append(
                    f"[{result.metadata.get('file_name', 'Source')}]: {result.content}"
                )
        
        if not context_parts:
            return "No highly relevant information found."
        
        return "\n\n".join(context_parts)
    
    def _get_fallback_response(self, intent: CustomerIntent) -> str:
        """Get fallback response based on intent."""
        fallbacks = {
            CustomerIntent.PRODUCT_INFO: """I'd be happy to help you find product information.

Could you please specify:
- The product name or category
- What specifications you need
- How you plan to use the product

Or browse our product catalog for detailed specifications.""",
            
            CustomerIntent.PRICING_QUOTE: self.templates["transfer_sales"],
            
            CustomerIntent.ORDER_STATUS: """To track your order, I'll need:
- Your order number (format: ORD-XXXXX)
- The email address used for the order

You can also track orders at jdjones.com/track""",
            
            CustomerIntent.TECHNICAL_SUPPORT: """I understand you need technical assistance.

For immediate help:
- Check our troubleshooting guides
- Call our support line: 1-800-JD-JONES

Or submit a support ticket and we'll respond within 24 hours.""",
            
            CustomerIntent.RETURNS_WARRANTY: """For returns and warranty claims:

**Returns:** Within 30 days, original packaging required
**Warranty:** 2-year standard warranty on all products

Would you like to:
- Start a return request
- File a warranty claim
- Read our full return policy""",
            
            CustomerIntent.COMPLAINT: """I'm sorry to hear you're having a problem. 
Your satisfaction is important to us.

Please share the details of your concern, and I'll make sure it's addressed by the right team.

For urgent matters, call 1-800-JD-JONES and ask for a supervisor.""",
        }
        
        return fallbacks.get(intent, self.templates["no_info"])
    
    def _get_suggested_actions(self, intent: CustomerIntent) -> List[Dict[str, str]]:
        """Get suggested follow-up actions based on intent."""
        actions = {
            CustomerIntent.PRODUCT_INFO: [
                {"label": "Browse Catalog", "action": "navigate", "target": "product_catalog"},
                {"label": "Request Specs", "action": "navigate", "target": "product_specs_form"},
                {"label": "Get Quote", "action": "navigate", "target": "quote_main"},
            ],
            CustomerIntent.PRICING_QUOTE: [
                {"label": "Standard Quote", "action": "navigate", "target": "quote_standard_form"},
                {"label": "Custom Quote", "action": "navigate", "target": "quote_custom_form"},
                {"label": "Contact Sales", "action": "navigate", "target": "contact_sales_form"},
            ],
            CustomerIntent.ORDER_STATUS: [
                {"label": "Track Order", "action": "navigate", "target": "order_tracking"},
                {"label": "Contact Support", "action": "navigate", "target": "support_main"},
            ],
            CustomerIntent.TECHNICAL_SUPPORT: [
                {"label": "Submit Ticket", "action": "navigate", "target": "support_ticket_form"},
                {"label": "View Guides", "action": "navigate", "target": "support_documentation"},
                {"label": "Request Callback", "action": "navigate", "target": "contact_callback_form"},
            ],
            CustomerIntent.RETURNS_WARRANTY: [
                {"label": "Start Return", "action": "navigate", "target": "returns_start"},
                {"label": "Warranty Claim", "action": "navigate", "target": "warranty_claim"},
                {"label": "View Policy", "action": "navigate", "target": "returns_policy"},
            ],
        }
        
        return actions.get(intent, [
            {"label": "Contact Us", "action": "navigate", "target": "contact_main"},
            {"label": "Browse Products", "action": "navigate", "target": "product_main"},
        ])
    
    def get_template_response(self, template_key: str) -> Optional[str]:
        """Get a template response by key."""
        return self.templates.get(template_key)
    
    async def generate_faq_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate answer for FAQ-type questions.
        
        Enhanced with FAQ cache for instant responses.
        
        Args:
            question: FAQ question
            
        Returns:
            Answer dict
        """
        # Step 1: Check FAQ cache first
        if self.faq_cache:
            cache_result = self.faq_cache.get(question)
            if cache_result and cache_result.get("confidence", 0) > 0.8:
                logger.debug(f"FAQ cache hit for: {question[:50]}...")
                return {
                    "answer": cache_result["answer"],
                    "confidence": cache_result["confidence"],
                    "source": f"FAQ Cache ({cache_result.get('category', 'predefined')})",
                    "cache_hit": True
                }
        
        # Step 2: Search knowledge base for FAQ answers
        results = self.knowledge_base.query(
            query_text=question,
            n_results=3,
            include_public_only=True,
            filter_metadata={"category": "faq"}
        )
        
        if results and results[0].relevance_score > 0.8:
            # Add to FAQ cache for future queries
            if self.faq_cache:
                self.faq_cache.set(
                    question=question,
                    answer=results[0].content,
                    category="knowledge_base"
                )
            
            return {
                "answer": results[0].content,
                "confidence": results[0].relevance_score,
                "source": results[0].metadata.get("file_name", "FAQ Database")
            }
        
        # Step 3: Fall back to general knowledge base
        results = self.knowledge_base.query(
            query_text=question,
            n_results=3,
            include_public_only=True
        )
        
        if results and results[0].relevance_score > 0.7:
            return {
                "answer": results[0].content,
                "confidence": results[0].relevance_score,
                "source": results[0].metadata.get("file_name", "Knowledge Base")
            }
        
        return {
            "answer": "I don't have a specific answer to that question. Please contact our support team for assistance.",
            "confidence": 0.0,
            "source": None
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics from FAQ and Embedding caches."""
        stats = {}
        
        if self.faq_cache:
            stats["faq_cache"] = self.faq_cache.get_stats()
        
        if self.embedding_cache:
            stats["embedding_cache"] = self.embedding_cache.get_stats()
        
        return stats

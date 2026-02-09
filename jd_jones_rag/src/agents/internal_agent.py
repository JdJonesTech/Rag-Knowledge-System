"""
Internal Agent - AI Chatbot for Employees.
Conversational agent with RAG integration and access control.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.config.settings import settings
from src.knowledge_base.retriever import HierarchicalRetriever, UserRole
from src.auth.authentication import User


class InternalAgent:
    """
    Internal conversational AI agent for employees.
    
    Features:
    - Access-controlled RAG retrieval
    - Conversation memory
    - Source citations
    - Personalized responses based on user role
    """
    
    def __init__(self):
        """Initialize the internal agent."""
        from src.config.settings import get_llm
        # Dual strategy: Creative temperature (0.7) for intelligent, natural responses
        # + GroundingValidator catches any fabricated specs post-generation.
        # This gives engaging customer/team responses that are factually accurate.
        self.llm = get_llm(temperature=0.7)
        
        self.retriever = HierarchicalRetriever()
        self.system_prompt = self._load_system_prompt()
        
        # Programmatic grounding validator (anti-hallucination)
        from src.retrieval.grounding_validator import get_grounding_validator
        self.grounding_validator = get_grounding_validator()
        
        # Session storage (in-memory, use Redis in production)
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file or use default."""
        prompt_path = Path(__file__).parent / "prompts" / "internal_system_prompt.txt"
        
        if prompt_path.exists():
            return prompt_path.read_text()
        
        # Default system prompt
        return """You are JD Jones AI Assistant, an intelligent assistant for company employees.

You have access to the company's knowledge base and can help with:
- Product information and specifications
- Company policies and procedures
- Department-specific documentation
- General inquiries

IMPORTANT GUIDELINES:
1. Only answer based on the provided context from the knowledge base
2. If information is not in the context, say you don't have that information
3. Always cite your sources when providing information
4. Tailor your response to the user's role and department
5. Be professional, helpful, and concise
6. For sensitive topics (HR, legal, financial), recommend consulting appropriate departments

USER CONTEXT:
- Name: {user_name}
- Role: {user_role}
- Department: {user_department}

KNOWLEDGE BASE CONTEXT:
{context}

Remember: You are here to help employees be more productive. Be accurate and helpful."""
    
    async def chat(
        self,
        user: User,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate a response.
        
        Enhanced with parallel product catalog search for grounded product recommendations.
        
        Args:
            user: Authenticated user
            query: User's message
            conversation_history: Previous messages in conversation
            session_id: Session identifier for conversation tracking
            
        Returns:
            Response dict with message, sources, and metadata
        """
        import asyncio
        import re
        import logging
        logger = logging.getLogger(__name__)
        
        # Get or create session
        if session_id and session_id in self.sessions:
            conversation_history = self.sessions[session_id]
        elif not conversation_history:
            conversation_history = []
        
        # Map user role to retriever role
        try:
            user_role = UserRole(user.role)
        except ValueError:
            user_role = UserRole.EMPLOYEE
        
        # Detect if this is a product-related query
        product_keywords = [
            'product', 'packing', 'seal', 'gasket', 'valve', 'pump', 'graphite',
            'ptfe', 'temperature', 'pressure', 'recommend', 'suggestion', 'code',
            'NA ', 'high temp', 'mining', 'refinery', 'steam', 'chemical',
            'offer', 'suggest', 'best', 'which product', 'what product'
        ]
        is_product_query = any(kw.lower() in query.lower() for kw in product_keywords)
        
        # Parallel search: KB retrieval + Product Catalog search
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
                        logger.info(f"Direct product code lookup found {len(direct_results)} products")
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
                        operating_temp = 350.0
                
                pressure_match = re.search(r'(\d+)\s*bar|high\s+pressure', query.lower())
                operating_pressure = None
                if pressure_match:
                    if pressure_match.group(1):
                        operating_pressure = float(pressure_match.group(1))
                    elif 'high pressure' in query.lower():
                        operating_pressure = 150.0
                
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
                elif 'food' in query.lower():
                    industry = "food"
                elif 'pharma' in query.lower():
                    industry = "pharmaceutical"
                
                # Detect application
                application = None
                if 'valve' in query.lower():
                    application = "valve"
                elif 'pump' in query.lower():
                    application = "pump"
                elif 'agitator' in query.lower() or 'mixer' in query.lower():
                    application = "agitator"
                elif 'flange' in query.lower():
                    application = "flange"
                elif 'boiler' in query.lower():
                    application = "boiler"
                
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
        
        # Run product search in parallel if needed
        product_matches = []
        if is_product_query:
            product_matches = await search_product_catalog()
        
        # Retrieve relevant documents from KB
        retrieval_response = self.retriever.retrieve(
            query=query,
            user_role=user_role,
            user_department=user.department,
            n_results=settings.max_retrieval_results
        )
        
        # Format context for LLM
        context = self.retriever.format_context_for_llm(
            retrieval_response,
            max_tokens=3000,
            include_sources=True
        )
        
        # Add grounded product data to context
        product_context = ""
        if product_matches:
            product_context = "\n\n=== REAL PRODUCT DATA FROM CATALOG (USE ONLY THESE PRODUCT CODES) ===\n"
            product_context += "IMPORTANT: For internal team - provide these exact product codes to customers.\n\n"
            for i, match in enumerate(product_matches[:5], 1):
                product = match.product
                specs = product.specs
                product_context += f"""
Product #{i}: **{product.code}** - {product.name}
- Category: {product.category}
- Material: {product.material}
- Description: {product.description[:300]}
- Temperature Range: {specs.temperature_min}°C to {specs.temperature_max}°C
- Max Pressure (Static): {specs.pressure_static or 'N/A'} bar
- Max Pressure (Rotary): {specs.pressure_rotary or 'N/A'} bar
- Certifications: {', '.join(product.certifications) if product.certifications else 'None'}
- Industries: {', '.join(product.industries[:3]) if product.industries else 'General'}
- Applications: {', '.join(product.applications[:3]) if product.applications else 'Various'}
- Match Score: {match.score} | Confidence: {match.confidence.value}
- Why Recommended: {', '.join(match.match_reasons)}
"""
            product_context += "\nCRITICAL: Only recommend products from this list. These are verified JD Jones product codes.\n"
        
        # Build system prompt with user context
        full_context = context + product_context
        
        # PRE-GENERATION: Grounding sufficiency check
        is_sufficient, fallback = self.grounding_validator.check_context_sufficiency(
            context=full_context,
            product_matches=product_matches,
            query=query
        )
        
        if not is_sufficient and fallback:
            # Short-circuit: don't even ask the LLM if we have no data
            response_text = fallback
        else:
            system_content = self.system_prompt.format(
                user_name=user.full_name,
                user_role=user.role,
                user_department=user.department or "General",
                context=full_context
            )
            
            # Build messages
            messages = [SystemMessage(content=system_content)]
            
            # Add conversation history (last 6 messages)
            for msg in conversation_history[-6:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current query
            messages.append(HumanMessage(content=query))
            
            # Generate response
            llm_response = await self.llm.ainvoke(messages)
            
            # POST-GENERATION: Grounding validation
            grounding_result = self.grounding_validator.validate_response(
                response=llm_response.content,
                context=full_context,
                product_matches=product_matches,
                query=query
            )
            
            response_text = grounding_result.validated_response
        
        # Extract sources from retrieval results
        sources = []
        for result in retrieval_response.all_results[:5]:
            source_info = {
                "file_name": result.metadata.get("file_name", "Unknown"),
                "source": result.source,
                "relevance": round(result.relevance_score, 3)
            }
            if source_info not in sources:
                sources.append(source_info)
        
        # Add product catalog as source if products were found
        if product_matches:
            sources.append({
                "file_name": "Product Catalog (Real-time)",
                "source": "product_catalog",
                "relevance": 1.0,
                "products_found": len(product_matches)
            })
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response_text})
        
        # Save to session
        if session_id:
            self.sessions[session_id] = conversation_history
        
        return {
            "response": response_text,
            "sources": sources,
            "session_id": session_id or self._generate_session_id(),
            "metadata": {
                "retrieval_count": retrieval_response.total_count,
                "user_role": user.role,
                "timestamp": datetime.now().isoformat(),
                "products_searched": len(product_matches) if product_matches else 0
            }
        }
    
    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of message dicts
        """
        return self.sessions.get(session_id, [])
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return f"session_{uuid.uuid4().hex[:12]}"
    
    async def get_suggested_questions(
        self,
        user: User,
        context: Optional[str] = None
    ) -> List[str]:
        """
        Generate suggested questions based on user role.
        
        Args:
            user: User object
            context: Optional context for suggestions
            
        Returns:
            List of suggested questions
        """
        role_suggestions = {
            "sales_rep": [
                "What are the current pricing guidelines for bulk orders?",
                "How do I generate a quote for a new customer?",
                "What's our competitive advantage over [competitor]?",
            ],
            "production_worker": [
                "What are the safety procedures for machine X?",
                "Where can I find the work instructions for product Y?",
                "What's the maintenance schedule for equipment Z?",
            ],
            "engineer": [
                "What are the material specifications for product X?",
                "Where can I find the latest test reports?",
                "What compliance requirements apply to our products?",
            ],
            "customer_service_rep": [
                "How do I handle a return request?",
                "What's the warranty policy for product X?",
                "How do I escalate a customer complaint?",
            ],
            "manager": [
                "Show me the latest performance metrics",
                "What are the current department priorities?",
                "How do I access budget information?",
            ],
        }
        
        # Get role-specific suggestions or default
        suggestions = role_suggestions.get(user.role, [
            "What products does JD Jones offer?",
            "How do I find company policies?",
            "What safety guidelines should I follow?",
        ])
        
        return suggestions

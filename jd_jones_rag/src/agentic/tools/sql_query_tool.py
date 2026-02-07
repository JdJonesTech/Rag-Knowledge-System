"""
SQL Query Tool
Executes queries against ERP/database systems.
Uses a specialized SLM for SQL generation for privacy and speed.
"""

from typing import Dict, Any, Optional, List
import json

from src.agentic.tools.base_tool import BaseTool, ToolResult, ToolStatus
from src.config.settings import settings


class SQLQueryTool(BaseTool):
    """
    Tool for querying ERP and database systems.
    
    In production:
    - Uses a specialized SLM (e.g., Mistral 7B fine-tuned for SQL)
    - Runs locally for data privacy
    - Provides near-instant responses
    """
    
    # Sample inventory/ERP data - loaded dynamically from JSON files via JDJonesDataLoader
    # No longer hardcoded - see _ensure_inventory_loaded() method
    SAMPLE_INVENTORY = {}  # Populated on first access via _ensure_inventory_loaded()
    SAMPLE_ORDERS = {}  # Sample order data
    _inventory_loaded = False
    
    @classmethod
    def _ensure_inventory_loaded(cls):
        """Load inventory data from JSON files if not already loaded."""
        if cls._inventory_loaded:
            return
            
        try:
            from src.data_ingestion.jd_jones_data_loader import get_data_loader
            import random
            from datetime import datetime, timedelta
            
            loader = get_data_loader()
            products = loader.get_all_products()
            
            # Generate simulated inventory data for each product
            for code, prod in products.items():
                # Simulate realistic inventory data
                base_price = random.randint(40, 150)
                stock = random.randint(100, 2000)
                lead_time = random.choice([0, 0, 0, 3, 5, 7, 14])  # Most in stock
                restock_date = datetime.now() - timedelta(days=random.randint(5, 30))
                
                cls.SAMPLE_INVENTORY[code] = {
                    "stock_quantity": stock,
                    "warehouse": "Kolkata-Main",
                    "unit_price": float(base_price),
                    "currency": "USD",
                    "lead_time_days": lead_time,
                    "reorder_level": stock // 4,
                    "last_restocked": restock_date.strftime("%Y-%m-%d"),
                    "product_name": prod.get("name", ""),
                }
            
            # Sample orders using actual product codes
            product_codes = list(products.keys())[:10] if len(products) >= 10 else list(products.keys())
            
            cls.SAMPLE_ORDERS = {
                "ORD-2024-1234": {
                    "customer": "Saudi Aramco",
                    "products": [product_codes[0]] if product_codes else ["NA 715"],
                    "quantity": 500,
                    "status": "shipped",
                    "ship_date": "2024-11-18",
                    "estimated_delivery": "2024-11-25",
                    "tracking": "DHL-9876543210"
                },
                "ORD-2024-1235": {
                    "customer": "Shell Chemicals",
                    "products": product_codes[:2] if len(product_codes) >= 2 else ["NA 701"],
                    "quantity": 1000,
                    "status": "processing",
                    "estimated_ship": "2024-11-28"
                }
            }
            
            cls._inventory_loaded = True
            
            import logging
            logging.info(f"Loaded inventory data for {len(cls.SAMPLE_INVENTORY)} products from JSON files")
            
        except ImportError:
            import logging
            logging.warning("JDJonesDataLoader not available, using fallback inventory data")
            # Minimal fallback data
            cls.SAMPLE_INVENTORY = {
                "NA 715": {"stock_quantity": 1000, "warehouse": "Kolkata-Main", "unit_price": 85.00, "currency": "USD", "lead_time_days": 0},
                "NA 701": {"stock_quantity": 500, "warehouse": "Kolkata-Main", "unit_price": 65.00, "currency": "USD", "lead_time_days": 3},
            }
            cls.SAMPLE_ORDERS = {}
            cls._inventory_loaded = True
        except Exception as e:
            import logging
            logging.error(f"Error loading inventory from JSON: {e}")
            cls._inventory_loaded = True
    

    def __init__(self):
        """Initialize SQL query tool."""
        super().__init__(
            name="erp_query",
            description="""
            Queries the ERP system for real-time data:
            - Stock/inventory levels
            - Pricing information
            - Order status
            - Lead times
            - Customer history
            
            Privacy: Runs locally using specialized SLM.
            """
        )
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """
        Execute ERP query.
        
        Args:
            query: Natural language query
            parameters: Query parameters
            intent: Query intent
            
        Returns:
            ToolResult with data
        """
        try:
            # Ensure inventory data is loaded
            self._ensure_inventory_loaded()
            
            query_type = parameters.get("query_type", "")
            
            if intent == "stock_availability" or "stock" in query.lower():
                return await self._check_stock(parameters)
            
            elif intent == "order_status" or "order" in query.lower():
                return await self._check_order(parameters)
            
            elif intent == "pricing_quote" or "price" in query.lower():
                return await self._get_pricing(parameters)
            
            else:
                # General query
                return await self._general_query(query, parameters)
                
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _check_stock(self, parameters: Dict[str, Any]) -> ToolResult:
        """Check stock availability."""
        product_ids = parameters.get("product_ids", [])
        
        if isinstance(product_ids, str):
            product_ids = [product_ids]
        
        results = []
        for pid in product_ids:
            pid_upper = pid.upper()
            if pid_upper in self.SAMPLE_INVENTORY:
                inv = self.SAMPLE_INVENTORY[pid_upper]
                results.append({
                    "product_id": pid_upper,
                    "in_stock": inv["stock_quantity"] > 0,
                    "quantity_available": inv["stock_quantity"],
                    "warehouse": inv["warehouse"],
                    "lead_time_days": inv["lead_time_days"],
                    "available_immediately": inv["stock_quantity"] > 0 and inv["lead_time_days"] == 0
                })
            else:
                # Search partial match
                for key, inv in self.SAMPLE_INVENTORY.items():
                    if pid.upper() in key:
                        results.append({
                            "product_id": key,
                            "in_stock": inv["stock_quantity"] > 0,
                            "quantity_available": inv["stock_quantity"],
                            "warehouse": inv["warehouse"],
                            "lead_time_days": inv["lead_time_days"]
                        })
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS if results else ToolStatus.PARTIAL,
            data={
                "inventory_status": results,
                "query_time": "real-time"
            },
            sources=[{"document": "ERP System", "relevance": 1.0}]
        )
    
    async def _check_order(self, parameters: Dict[str, Any]) -> ToolResult:
        """Check order status."""
        order_id = parameters.get("order_id", "").upper()
        
        if order_id in self.SAMPLE_ORDERS:
            order = self.SAMPLE_ORDERS[order_id]
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data={
                    "order_id": order_id,
                    "order_details": order
                },
                sources=[{"document": "Order System", "relevance": 1.0}]
            )
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.PARTIAL,
            data={"message": f"Order {order_id} not found"},
            error="Order not found"
        )
    
    async def _get_pricing(self, parameters: Dict[str, Any]) -> ToolResult:
        """Get pricing information."""
        product_ids = parameters.get("product_ids", [])
        quantity = parameters.get("quantity", 1)
        
        if isinstance(product_ids, str):
            product_ids = [product_ids]
        
        pricing = []
        total = 0
        
        for pid in product_ids:
            pid_upper = pid.upper()
            for key, inv in self.SAMPLE_INVENTORY.items():
                if pid_upper in key:
                    unit_price = inv["unit_price"]
                    
                    # Volume discount
                    discount = 0
                    if quantity >= 1000:
                        discount = 0.15
                    elif quantity >= 500:
                        discount = 0.10
                    elif quantity >= 100:
                        discount = 0.05
                    
                    final_price = unit_price * (1 - discount)
                    line_total = final_price * quantity
                    total += line_total
                    
                    pricing.append({
                        "product_id": key,
                        "unit_price": unit_price,
                        "discount_percent": discount * 100,
                        "final_unit_price": round(final_price, 2),
                        "quantity": quantity,
                        "line_total": round(line_total, 2),
                        "currency": inv["currency"]
                    })
                    break
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS if pricing else ToolStatus.PARTIAL,
            data={
                "pricing": pricing,
                "total_amount": round(total, 2),
                "currency": "USD",
                "valid_until": "30 days",
                "terms": "Net 30"
            },
            sources=[{"document": "Pricing System", "relevance": 1.0}]
        )
    
    async def _general_query(self, query: str, parameters: Dict[str, Any]) -> ToolResult:
        """Handle general queries."""
        # In production, this would use the SQL SLM to generate and execute queries
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.PARTIAL,
            data={"message": "General query executed", "query": query},
            metadata={"parameters": parameters}
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "query_type": {
                    "type": "string",
                    "enum": ["stock", "order", "pricing", "customer", "general"]
                },
                "product_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "order_id": {"type": "string"},
                "customer_id": {"type": "string"},
                "quantity": {"type": "integer"}
            }
        }

"""
SQL Generator SLM
Specialized small model for SQL query generation.
Runs locally for data privacy and low latency.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import re


class QueryType(str, Enum):
    """Types of SQL queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"


@dataclass
class SQLQuery:
    """Generated SQL query."""
    query: str
    query_type: QueryType
    tables_used: List[str]
    parameters: Dict[str, Any]
    confidence: float
    explanation: str
    is_safe: bool


class SQLGeneratorSLM:
    """
    Specialized SLM for SQL generation.
    
    In production, this would use a fine-tuned model like:
    - Mistral 7B fine-tuned for SQL
    - CodeLlama for SQL
    - SQLCoder
    
    Currently implements rule-based generation with templates.
    """
    
    # Database schema (simplified)
    SCHEMA = {
        "products": {
            "columns": ["product_id", "product_name", "category", "price", "stock_qty", "max_temp", "max_pressure"],
            "primary_key": "product_id"
        },
        "orders": {
            "columns": ["order_id", "customer_id", "order_date", "status", "total_amount"],
            "primary_key": "order_id"
        },
        "order_items": {
            "columns": ["item_id", "order_id", "product_id", "quantity", "unit_price"],
            "primary_key": "item_id"
        },
        "customers": {
            "columns": ["customer_id", "company_name", "contact_name", "email", "industry"],
            "primary_key": "customer_id"
        },
        "inventory": {
            "columns": ["inventory_id", "product_id", "warehouse", "quantity", "last_updated"],
            "primary_key": "inventory_id"
        }
    }
    
    # Query templates
    TEMPLATES = {
        "stock_check": """
            SELECT p.product_id, p.product_name, i.quantity as stock_qty, i.warehouse
            FROM products p
            LEFT JOIN inventory i ON p.product_id = i.product_id
            WHERE {conditions}
        """,
        "order_status": """
            SELECT o.order_id, o.order_date, o.status, c.company_name, o.total_amount
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE {conditions}
        """,
        "product_search": """
            SELECT product_id, product_name, category, price, max_temp, max_pressure
            FROM products
            WHERE {conditions}
        """,
        "sales_report": """
            SELECT 
                DATE_TRUNC('month', o.order_date) as month,
                COUNT(o.order_id) as order_count,
                SUM(o.total_amount) as total_sales
            FROM orders o
            WHERE o.order_date >= :start_date AND o.order_date <= :end_date
            GROUP BY DATE_TRUNC('month', o.order_date)
            ORDER BY month
        """
    }
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r";\s*DROP\s+",
        r";\s*DELETE\s+",
        r";\s*TRUNCATE\s+",
        r"--",
        r"/\*",
        r"UNION\s+SELECT",
        r"OR\s+1\s*=\s*1",
        r"EXEC\s+",
        r"EXECUTE\s+"
    ]
    
    def __init__(self, allow_writes: bool = False):
        """
        Initialize SQL generator.
        
        Args:
            allow_writes: Whether to allow INSERT/UPDATE/DELETE
        """
        self.allow_writes = allow_writes
    
    def generate(
        self,
        natural_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SQLQuery:
        """
        Generate SQL from natural language.
        
        Args:
            natural_query: Natural language query
            context: Additional context (user, filters, etc.)
            
        Returns:
            SQLQuery with generated SQL
        """
        query_lower = natural_query.lower()
        context = context or {}
        
        # Determine query type
        query_type = self._detect_query_type(query_lower)
        
        # Block writes if not allowed
        if not self.allow_writes and query_type in [QueryType.INSERT, QueryType.UPDATE, QueryType.DELETE]:
            return SQLQuery(
                query="",
                query_type=query_type,
                tables_used=[],
                parameters={},
                confidence=0,
                explanation="Write operations are not allowed",
                is_safe=False
            )
        
        # Generate SQL based on pattern matching
        if "stock" in query_lower or "inventory" in query_lower:
            return self._generate_stock_query(natural_query, context)
        
        elif "order" in query_lower and ("status" in query_lower or "track" in query_lower):
            return self._generate_order_query(natural_query, context)
        
        elif "product" in query_lower or "search" in query_lower:
            return self._generate_product_query(natural_query, context)
        
        elif "sales" in query_lower or "report" in query_lower:
            return self._generate_sales_query(natural_query, context)
        
        else:
            return self._generate_generic_query(natural_query, context)
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect type of query needed."""
        if any(kw in query for kw in ["add", "create", "insert", "new"]):
            return QueryType.INSERT
        elif any(kw in query for kw in ["update", "change", "modify"]):
            return QueryType.UPDATE
        elif any(kw in query for kw in ["delete", "remove"]):
            return QueryType.DELETE
        elif any(kw in query for kw in ["count", "total", "sum", "average", "report"]):
            return QueryType.AGGREGATE
        return QueryType.SELECT
    
    def _generate_stock_query(
        self,
        natural_query: str,
        context: Dict[str, Any]
    ) -> SQLQuery:
        """Generate stock/inventory query."""
        conditions = []
        params = {}
        
        # Extract product IDs
        product_ids = context.get("product_ids", [])
        if product_ids:
            placeholders = ", ".join([f":pid_{i}" for i in range(len(product_ids))])
            conditions.append(f"p.product_id IN ({placeholders})")
            for i, pid in enumerate(product_ids):
                params[f"pid_{i}"] = pid
        
        # Product name search
        product_name = context.get("product_name")
        if product_name:
            conditions.append("p.product_name ILIKE :product_name")
            params["product_name"] = f"%{product_name}%"
        
        if not conditions:
            conditions.append("1=1")
        
        query = self.TEMPLATES["stock_check"].format(
            conditions=" AND ".join(conditions)
        )
        
        return SQLQuery(
            query=self._clean_query(query),
            query_type=QueryType.SELECT,
            tables_used=["products", "inventory"],
            parameters=params,
            confidence=0.85,
            explanation="Stock check query for specified products",
            is_safe=self._is_safe(query)
        )
    
    def _generate_order_query(
        self,
        natural_query: str,
        context: Dict[str, Any]
    ) -> SQLQuery:
        """Generate order status query."""
        conditions = []
        params = {}
        
        order_id = context.get("order_id")
        if order_id:
            conditions.append("o.order_id = :order_id")
            params["order_id"] = order_id
        
        customer_id = context.get("customer_id")
        if customer_id:
            conditions.append("o.customer_id = :customer_id")
            params["customer_id"] = customer_id
        
        status = context.get("status")
        if status:
            conditions.append("o.status = :status")
            params["status"] = status
        
        if not conditions:
            conditions.append("1=1")
        
        query = self.TEMPLATES["order_status"].format(
            conditions=" AND ".join(conditions)
        )
        
        return SQLQuery(
            query=self._clean_query(query),
            query_type=QueryType.SELECT,
            tables_used=["orders", "customers"],
            parameters=params,
            confidence=0.9,
            explanation="Order status lookup",
            is_safe=self._is_safe(query)
        )
    
    def _generate_product_query(
        self,
        natural_query: str,
        context: Dict[str, Any]
    ) -> SQLQuery:
        """Generate product search query."""
        conditions = []
        params = {}
        
        # Temperature filter
        max_temp = context.get("temperature")
        if max_temp:
            conditions.append("max_temp >= :min_temp")
            params["min_temp"] = max_temp
        
        # Pressure filter
        max_pressure = context.get("pressure")
        if max_pressure:
            conditions.append("max_pressure >= :min_pressure")
            params["min_pressure"] = max_pressure
        
        # Category filter
        category = context.get("category")
        if category:
            conditions.append("category = :category")
            params["category"] = category
        
        # Search term
        search_term = context.get("search_term")
        if search_term:
            conditions.append("product_name ILIKE :search_term")
            params["search_term"] = f"%{search_term}%"
        
        if not conditions:
            conditions.append("1=1")
        
        query = self.TEMPLATES["product_search"].format(
            conditions=" AND ".join(conditions)
        )
        
        return SQLQuery(
            query=self._clean_query(query),
            query_type=QueryType.SELECT,
            tables_used=["products"],
            parameters=params,
            confidence=0.85,
            explanation="Product search with filters",
            is_safe=self._is_safe(query)
        )
    
    def _generate_sales_query(
        self,
        natural_query: str,
        context: Dict[str, Any]
    ) -> SQLQuery:
        """Generate sales report query."""
        params = {
            "start_date": context.get("start_date", "2024-01-01"),
            "end_date": context.get("end_date", "2024-12-31")
        }
        
        query = self.TEMPLATES["sales_report"]
        
        return SQLQuery(
            query=self._clean_query(query),
            query_type=QueryType.AGGREGATE,
            tables_used=["orders"],
            parameters=params,
            confidence=0.9,
            explanation="Monthly sales report",
            is_safe=self._is_safe(query)
        )
    
    def _generate_generic_query(
        self,
        natural_query: str,
        context: Dict[str, Any]
    ) -> SQLQuery:
        """Fallback for unrecognized queries."""
        return SQLQuery(
            query="SELECT 'Query not understood' as message",
            query_type=QueryType.SELECT,
            tables_used=[],
            parameters={},
            confidence=0.3,
            explanation="Unable to generate specific query - please provide more details",
            is_safe=True
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and format SQL query."""
        # Remove extra whitespace
        query = " ".join(query.split())
        return query.strip()
    
    def _is_safe(self, query: str) -> bool:
        """Check if query is safe from injection."""
        query_upper = query.upper()
        
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                return False
        
        return True
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate a SQL query for safety and correctness."""
        result = {
            "is_valid": True,
            "is_safe": self._is_safe(query),
            "issues": [],
            "tables_referenced": []
        }
        
        if not result["is_safe"]:
            result["is_valid"] = False
            result["issues"].append("Query contains potentially dangerous patterns")
        
        # Check for referenced tables
        for table in self.SCHEMA.keys():
            if table.lower() in query.lower():
                result["tables_referenced"].append(table)
        
        return result

"""
Compliance Checker Tool
Validates products and configurations against industry standards.
Implements the "Validator" pattern for agentic systems.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.agentic.tools.base_tool import BaseTool, ToolResult, ToolStatus


class ComplianceLevel(str, Enum):
    """Compliance check result levels."""
    FULLY_COMPLIANT = "fully_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_VERIFICATION = "requires_verification"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class StandardRequirement:
    """A single requirement from a standard."""
    requirement_id: str
    description: str
    is_mandatory: bool
    verification_method: str
    applies_to: List[str] = field(default_factory=list)


class ComplianceCheckerTool(BaseTool):
    """
    Tool for checking compliance with industry standards.
    
    Capabilities:
    - Verify API 622/624 compliance
    - Check Shell SPE requirements
    - Validate FDA/food-grade compliance
    - Cross-check certifications
    - Generate compliance reports
    """
    
    # Standards database
    STANDARDS = {
        "API_622": {
            "name": "API 622 - Type Testing of Process Valve Packing",
            "issuing_body": "American Petroleum Institute",
            "current_version": "4th Edition, 2018",
            "applies_to": ["valve_packing", "stem_seal", "gland_packing"],
            "requirements": [
                StandardRequirement(
                    "622-1", 
                    "Fugitive emission testing per API 641 or EPA Method 21", 
                    True, 
                    "Laboratory test",
                    ["all_packings"]
                ),
                StandardRequirement(
                    "622-2",
                    "Minimum 1510 mechanical cycles",
                    True,
                    "Endurance test",
                    ["all_packings"]
                ),
                StandardRequirement(
                    "622-3",
                    "Minimum 5 thermal cycles (-29°C to 315°C)",
                    True,
                    "Thermal cycling test",
                    ["high_temp_service"]
                ),
                StandardRequirement(
                    "622-4",
                    "Maximum leakage rate 100 ppm",
                    True,
                    "Emission measurement",
                    ["all_packings"]
                ),
                StandardRequirement(
                    "622-5",
                    "Live-loaded configuration capability",
                    False,
                    "Design review",
                    ["all_packings"]
                )
            ],
            "test_conditions": {
                "temperature_range": "-29°C to 315°C",
                "pressure": "According to valve class",
                "medium": "Methane or equivalent"
            }
        },
        "API_624": {
            "name": "API 624 - Type Testing of Rising Stem Valves",
            "issuing_body": "American Petroleum Institute",
            "current_version": "2nd Edition, 2020",
            "applies_to": ["gate_valve", "globe_valve", "rising_stem_valve"],
            "requirements": [
                StandardRequirement(
                    "624-1",
                    "Type testing per API 641",
                    True,
                    "Laboratory test",
                    ["rising_stem_valves"]
                ),
                StandardRequirement(
                    "624-2",
                    "310 mechanical cycles minimum",
                    True,
                    "Mechanical endurance",
                    ["rising_stem_valves"]
                ),
                StandardRequirement(
                    "624-3",
                    "3 thermal cycles minimum",
                    True,
                    "Thermal cycling",
                    ["rising_stem_valves"]
                ),
                StandardRequirement(
                    "624-4",
                    "ISO 15848-1 test method compliance",
                    True,
                    "Emission test",
                    ["rising_stem_valves"]
                )
            ]
        },
        "SHELL_SPE_77_312": {
            "name": "Shell SPE 77/312 - Valve Stem Packing",
            "issuing_body": "Shell Global Solutions",
            "current_version": "Rev 3.0",
            "applies_to": ["valve_packing", "stem_seal"],
            "requirements": [
                StandardRequirement(
                    "SPE-1",
                    "Fire-safe testing per API 607/ISO 10497",
                    True,
                    "Fire test",
                    ["all_packings"]
                ),
                StandardRequirement(
                    "SPE-2",
                    "Anti-extrusion capability at max pressure",
                    True,
                    "Extrusion test",
                    ["high_pressure"]
                ),
                StandardRequirement(
                    "SPE-3",
                    "Blowout resistance",
                    True,
                    "Pressure test",
                    ["all_packings"]
                ),
                StandardRequirement(
                    "SPE-4",
                    "Graphite content minimum 95%",
                    True,
                    "Material analysis",
                    ["graphite_packings"]
                )
            ]
        },
        "FDA_21CFR177": {
            "name": "FDA 21 CFR 177 - Indirect Food Additives",
            "issuing_body": "US Food and Drug Administration",
            "current_version": "Current",
            "applies_to": ["food_contact", "beverage", "pharmaceutical"],
            "requirements": [
                StandardRequirement(
                    "FDA-1",
                    "Materials listed in 21 CFR 177",
                    True,
                    "Material certification",
                    ["all_food_contact"]
                ),
                StandardRequirement(
                    "FDA-2",
                    "Extraction testing per FDA guidelines",
                    True,
                    "Laboratory test",
                    ["all_food_contact"]
                ),
                StandardRequirement(
                    "FDA-3",
                    "Non-toxic, non-tainting materials",
                    True,
                    "Material review",
                    ["all_food_contact"]
                )
            ]
        }
    }
    
    # Product certification database - loaded dynamically from JSON files via JDJonesDataLoader
    # No longer hardcoded - see _ensure_certifications_loaded() method
    PRODUCT_CERTIFICATIONS = {}  # Populated on first access via _ensure_certifications_loaded()
    _certifications_loaded = False
    
    @classmethod
    def _ensure_certifications_loaded(cls):
        """Load product certifications from JSON files if not already loaded."""
        if cls._certifications_loaded:
            return
            
        try:
            from src.data_ingestion.jd_jones_data_loader import get_data_loader
            loader = get_data_loader()
            products = loader.get_all_products()
            
            # Extract certifications for each product
            for code, prod in products.items():
                certs = prod.get("certifications", [])
                # Normalize certification names to match expected format
                normalized_certs = []
                for cert in certs:
                    if isinstance(cert, str):
                        # Convert certification names to standard format
                        cert_upper = cert.upper().replace(" ", "_").replace("-", "_")
                        if "API" in cert_upper:
                            normalized_certs.append(cert_upper)
                        elif "SHELL" in cert_upper or "SPE" in cert_upper:
                            normalized_certs.append(cert_upper)
                        elif "FDA" in cert_upper or "FOOD" in cert_upper:
                            normalized_certs.append(cert_upper)
                        else:
                            normalized_certs.append(cert)
                
                cls.PRODUCT_CERTIFICATIONS[code] = normalized_certs
            
            cls._certifications_loaded = True
            
            import logging
            logging.info(f"Loaded certifications for {len(cls.PRODUCT_CERTIFICATIONS)} products from JSON files")
            
        except ImportError:
            import logging
            logging.warning("JDJonesDataLoader not available, PRODUCT_CERTIFICATIONS will remain empty")
            cls._certifications_loaded = True
        except Exception as e:
            import logging
            logging.error(f"Error loading product certifications from JSON: {e}")
            cls._certifications_loaded = True
    

    def __init__(self):
        """Initialize compliance checker."""
        super().__init__(
            name="compliance_checker",
            description="""
            Validates products against industry standards:
            - API 622/624 for fugitive emissions
            - Shell SPE specifications
            - FDA food-grade requirements
            - ASME, ASTM, ISO standards
            
            Returns detailed compliance status and any gaps.
            """
        )
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """Execute compliance check."""
        try:
            # Ensure certifications are loaded
            self._ensure_certifications_loaded()
            
            action = parameters.get("action", "check_product")
            
            if action == "check_product":
                return await self._check_product_compliance(parameters)
            
            elif action == "get_standard_info":
                return await self._get_standard_info(parameters)
            
            elif action == "compare_standards":
                return await self._compare_with_standards(parameters)
            
            elif action == "generate_compliance_report":
                return await self._generate_compliance_report(parameters)
            
            else:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.PARTIAL,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _check_product_compliance(self, parameters: Dict[str, Any]) -> ToolResult:
        """Check if a product meets specified standards."""
        product_id = parameters.get("product_id", "").upper()
        required_standards = parameters.get("standards", [])
        
        # Normalize standard names
        required_standards = [s.upper().replace(" ", "_").replace("-", "_") for s in required_standards]
        
        # Get product certifications
        product_certs = self.PRODUCT_CERTIFICATIONS.get(product_id, [])
        product_certs_normalized = [c.upper().replace(" ", "_").replace("-", "_") for c in product_certs]
        
        results = {
            "product_id": product_id,
            "standards_requested": required_standards,
            "certifications_held": product_certs,
            "compliance_details": {},
            "overall_status": ComplianceLevel.FULLY_COMPLIANT.value
        }
        
        compliant = []
        non_compliant = []
        requires_verification = []
        
        for standard in required_standards:
            # Normalize for comparison
            std_key = standard.replace("API ", "API_").replace("SHELL ", "SHELL_").replace("FDA ", "FDA_")
            
            # Check if product has this certification
            is_certified = any(std_key in cert or cert in std_key for cert in product_certs_normalized)
            
            if is_certified:
                compliant.append(standard)
                results["compliance_details"][standard] = {
                    "status": ComplianceLevel.FULLY_COMPLIANT.value,
                    "certified": True,
                    "notes": "Product holds valid certification"
                }
            elif std_key in self.STANDARDS:
                # Standard exists but product not certified
                std_info = self.STANDARDS[std_key]
                if self._could_be_applicable(product_id, std_info):
                    non_compliant.append(standard)
                    results["compliance_details"][standard] = {
                        "status": ComplianceLevel.NON_COMPLIANT.value,
                        "certified": False,
                        "notes": f"Product not certified to {std_info['name']}",
                        "requirements_not_met": [r.description for r in std_info["requirements"][:3]]
                    }
                else:
                    results["compliance_details"][standard] = {
                        "status": ComplianceLevel.NOT_APPLICABLE.value,
                        "certified": False,
                        "notes": "Standard may not be applicable to this product type"
                    }
            else:
                requires_verification.append(standard)
                results["compliance_details"][standard] = {
                    "status": ComplianceLevel.REQUIRES_VERIFICATION.value,
                    "certified": False,
                    "notes": "Unable to verify - please contact technical team"
                }
        
        # Determine overall status
        if non_compliant:
            results["overall_status"] = ComplianceLevel.NON_COMPLIANT.value
        elif requires_verification:
            results["overall_status"] = ComplianceLevel.REQUIRES_VERIFICATION.value
        elif compliant:
            results["overall_status"] = ComplianceLevel.FULLY_COMPLIANT.value
        
        results["summary"] = {
            "compliant": compliant,
            "non_compliant": non_compliant,
            "requires_verification": requires_verification
        }
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data=results,
            sources=[{"document": "Compliance Database", "relevance": 1.0}]
        )
    
    async def _get_standard_info(self, parameters: Dict[str, Any]) -> ToolResult:
        """Get detailed information about a standard."""
        standard = parameters.get("standard", "").upper().replace(" ", "_").replace("-", "_")
        
        # Find matching standard
        std_key = None
        for key in self.STANDARDS:
            if standard in key or key in standard:
                std_key = key
                break
        
        if not std_key:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.PARTIAL,
                data={"message": f"Standard '{standard}' not found in database"}
            )
        
        std_info = self.STANDARDS[std_key]
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "standard_code": std_key,
                "name": std_info["name"],
                "issuing_body": std_info["issuing_body"],
                "current_version": std_info["current_version"],
                "applies_to": std_info["applies_to"],
                "requirements": [
                    {
                        "id": r.requirement_id,
                        "description": r.description,
                        "mandatory": r.is_mandatory,
                        "verification": r.verification_method
                    }
                    for r in std_info["requirements"]
                ],
                "test_conditions": std_info.get("test_conditions", {})
            }
        )
    
    async def _compare_with_standards(self, parameters: Dict[str, Any]) -> ToolResult:
        """Compare product specifications against standard requirements."""
        product_specs = parameters.get("specifications", {})
        standards = parameters.get("standards", [])
        
        comparison = {
            "specifications_provided": product_specs,
            "standards_compared": standards,
            "comparison_results": {}
        }
        
        for standard in standards:
            std_key = standard.upper().replace(" ", "_").replace("-", "_")
            
            if std_key in self.STANDARDS:
                std_info = self.STANDARDS[std_key]
                
                # Check each requirement
                requirement_checks = []
                for req in std_info["requirements"]:
                    check = {
                        "requirement": req.description,
                        "mandatory": req.is_mandatory,
                        "status": "not_evaluated"
                    }
                    
                    # Simplified checks based on specs
                    if "temperature" in req.description.lower() and "temperature_rating" in product_specs:
                        check["status"] = "potentially_met"
                        check["notes"] = f"Product temp rating: {product_specs['temperature_rating']}"
                    elif "emission" in req.description.lower():
                        check["status"] = "requires_test_verification"
                    
                    requirement_checks.append(check)
                
                comparison["comparison_results"][standard] = {
                    "standard_name": std_info["name"],
                    "requirement_checks": requirement_checks
                }
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data=comparison
        )
    
    async def _generate_compliance_report(self, parameters: Dict[str, Any]) -> ToolResult:
        """Generate a compliance report document."""
        product_id = parameters.get("product_id")
        standards = parameters.get("standards", [])
        
        # First check compliance
        check_result = await self._check_product_compliance({
            "product_id": product_id,
            "standards": standards
        })
        
        report = {
            "report_id": f"COMP-{datetime.now().strftime('%Y%m%d')}-{product_id[:8] if product_id else 'XXX'}",
            "generated_at": datetime.now().isoformat(),
            "product_id": product_id,
            "compliance_summary": check_result.data,
            "recommendations": [],
            "disclaimer": "This report is for informational purposes. Official certifications should be verified with issuing authorities."
        }
        
        # Add recommendations
        if check_result.data.get("summary", {}).get("non_compliant"):
            report["recommendations"].append(
                "Consider products with required certifications or contact engineering for custom solutions."
            )
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "report": report,
                "download_url": f"/api/reports/{report['report_id']}"
            }
        )
    
    def _could_be_applicable(self, product_id: str, standard_info: Dict) -> bool:
        """Check if a standard could be applicable to a product type."""
        product_lower = product_id.lower()
        applies_to = standard_info.get("applies_to", [])
        
        # Check for keyword matches
        for app in applies_to:
            if app.replace("_", "") in product_lower or product_lower in app:
                return True
        
        return True  # Default to applicable for safety
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["check_product", "get_standard_info", "compare_standards", "generate_compliance_report"]
                },
                "product_id": {"type": "string"},
                "standards": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of standards to check (e.g., ['API 622', 'Shell SPE'])"
                },
                "specifications": {
                    "type": "object",
                    "description": "Product specifications to compare"
                }
            }
        }

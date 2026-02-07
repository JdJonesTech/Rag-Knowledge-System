"""
Entity Extractor for GraphRAG
Extracts entities and relationships from JD Jones product documentation.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in JD Jones product domain."""
    PRODUCT = "product"
    PRODUCT_LINE = "product_line"
    MATERIAL = "material"
    APPLICATION = "application"
    INDUSTRY = "industry"
    STANDARD = "standard"
    SPECIFICATION = "specification"
    PROPERTY = "property"
    COMPONENT = "component"
    MANUFACTURER = "manufacturer"


class RelationType(Enum):
    """Types of relationships between entities."""
    MADE_OF = "made_of"
    USED_IN = "used_in"
    COMPLIES_WITH = "complies_with"
    HAS_PROPERTY = "has_property"
    VARIANT_OF = "variant_of"
    REPLACES = "replaces"
    COMPATIBLE_WITH = "compatible_with"
    PART_OF = "part_of"
    MANUFACTURED_BY = "manufactured_by"
    SUITABLE_FOR = "suitable_for"


@dataclass
class Entity:
    """Represents an extracted entity."""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    source_doc: Optional[str] = None
    confidence: float = 1.0
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class EntityExtractor:
    """
    Extracts entities and relationships from JD Jones product documentation.
    Uses pattern matching for product domain entities.
    """
    
    # JD Jones product patterns
    PRODUCT_PATTERNS = [
        r'\b(NA[\s\-]?\d{3,4}[A-Z]?)\b',  # NA 701, NA-702B
        r'\b(NJ[\s\-]?\d{3,4}[A-Z]?)\b',  # NJ 100
        r'\b(FLEXSEAL[\s\-]?\d*[A-Z]*)\b',
        r'\b(PACMAAN[\s\-]?\d*[A-Z]*)\b',
        r'\b(THERMEX[\s\-]?\d*[A-Z]*)\b',
    ]
    
    # Material patterns
    MATERIAL_PATTERNS = {
        r'\b(graphite|expanded graphite|flexible graphite)\b': "Graphite",
        r'\b(PTFE|teflon|polytetrafluoroethylene)\b': "PTFE",
        r'\b(aramid|kevlar|nomex)\b': "Aramid",
        r'\b(carbon fiber|carbon fibre)\b': "Carbon Fiber",
        r'\b(stainless steel|SS\s?\d{3})\b': "Stainless Steel",
        r'\b(inconel)\b': "Inconel",
        r'\b(monel)\b': "Monel",
        r'\b(mica)\b': "Mica",
        r'\b(ceramic fiber|ceramic fibre)\b': "Ceramic Fiber",
    }
    
    # Industry/Application patterns
    APPLICATION_PATTERNS = {
        r'\b(valve|valves)\b': "Valve Sealing",
        r'\b(pump|pumps)\b': "Pump Sealing",
        r'\b(flange|flanges)\b': "Flange Sealing",
        r'\b(heat exchanger)\b': "Heat Exchanger",
        r'\b(boiler)\b': "Boiler System",
        r'\b(turbine)\b': "Turbine System",
        r'\b(compressor)\b': "Compressor",
        r'\b(pipeline)\b': "Pipeline",
    }
    
    INDUSTRY_PATTERNS = {
        r'\b(oil\s*(&|and)?\s*gas|O&G)\b': "Oil & Gas",
        r'\b(petrochemical|petrochem)\b': "Petrochemical",
        r'\b(chemical|chemicals)\b': "Chemical",
        r'\b(pharmaceutical|pharma)\b': "Pharmaceutical",
        r'\b(power generation|power plant)\b': "Power Generation",
        r'\b(marine)\b': "Marine",
        r'\b(pulp\s*(&|and)?\s*paper)\b': "Pulp & Paper",
        r'\b(food\s*(&|and)?\s*beverage|F&B)\b': "Food & Beverage",
    }
    
    # Standard patterns
    STANDARD_PATTERNS = {
        r'\b(API[\s\-]?\d{3}[A-Z]?)\b': None,  # API 622, API 624
        r'\b(ASME[\s\-]?B\d+\.?\d*)\b': None,  # ASME B16.20
        r'\b(DIN[\s\-]?\d+)\b': None,
        r'\b(ISO[\s\-]?\d+)\b': None,
        r'\b(EN[\s\-]?\d+)\b': None,
        r'\b(Shell SPE[\s\-]?\d*/?\d*)\b': None,
    }
    
    # Property patterns (with value extraction)
    PROPERTY_PATTERNS = {
        r'temperature[:\s]+(?:up\s+to\s+)?(-?\d+)[\s°]*(C|F|°C|°F)': ("max_temperature", "temperature_unit"),
        r'pressure[:\s]+(?:up\s+to\s+)?(\d+[\d,]*)\s*(psi|bar|MPa)': ("max_pressure", "pressure_unit"),
        r'pH[:\s]+(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)': ("ph_min", "ph_max"),
    }
    
    def __init__(self):
        """Initialize the entity extractor."""
        self._compile_patterns()
        logger.info("EntityExtractor initialized with JD Jones domain patterns")
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self._compiled_products = [re.compile(p, re.IGNORECASE) for p in self.PRODUCT_PATTERNS]
        self._compiled_materials = {re.compile(k, re.IGNORECASE): v for k, v in self.MATERIAL_PATTERNS.items()}
        self._compiled_applications = {re.compile(k, re.IGNORECASE): v for k, v in self.APPLICATION_PATTERNS.items()}
        self._compiled_industries = {re.compile(k, re.IGNORECASE): v for k, v in self.INDUSTRY_PATTERNS.items()}
        self._compiled_standards = {re.compile(k, re.IGNORECASE): v for k, v in self.STANDARD_PATTERNS.items()}
    
    def extract_entities(self, text: str, source_doc: Optional[str] = None) -> List[Entity]:
        """
        Extract all entities from text.
        
        Args:
            text: Text to extract entities from
            source_doc: Optional source document identifier
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract products
        entities.extend(self._extract_products(text, source_doc))
        
        # Extract materials
        entities.extend(self._extract_materials(text, source_doc))
        
        # Extract applications
        entities.extend(self._extract_applications(text, source_doc))
        
        # Extract industries
        entities.extend(self._extract_industries(text, source_doc))
        
        # Extract standards
        entities.extend(self._extract_standards(text, source_doc))
        
        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.id not in seen:
                seen.add(entity.id)
                unique_entities.append(entity)
        
        logger.debug(f"Extracted {len(unique_entities)} entities from text")
        return unique_entities
    
    def _extract_products(self, text: str, source_doc: Optional[str]) -> List[Entity]:
        """Extract product entities."""
        entities = []
        for pattern in self._compiled_products:
            matches = pattern.findall(text)
            for match in matches:
                # Normalize product code
                product_code = re.sub(r'[\s\-]+', ' ', match.upper()).strip()
                entity_id = f"product:{product_code.replace(' ', '_')}"
                
                entities.append(Entity(
                    id=entity_id,
                    name=product_code,
                    entity_type=EntityType.PRODUCT,
                    source_doc=source_doc,
                    confidence=0.95
                ))
        return entities
    
    def _extract_materials(self, text: str, source_doc: Optional[str]) -> List[Entity]:
        """Extract material entities."""
        entities = []
        for pattern, normalized_name in self._compiled_materials.items():
            if pattern.search(text):
                entity_id = f"material:{normalized_name.lower().replace(' ', '_')}"
                entities.append(Entity(
                    id=entity_id,
                    name=normalized_name,
                    entity_type=EntityType.MATERIAL,
                    source_doc=source_doc,
                    confidence=0.9
                ))
        return entities
    
    def _extract_applications(self, text: str, source_doc: Optional[str]) -> List[Entity]:
        """Extract application entities."""
        entities = []
        for pattern, normalized_name in self._compiled_applications.items():
            if pattern.search(text):
                entity_id = f"application:{normalized_name.lower().replace(' ', '_')}"
                entities.append(Entity(
                    id=entity_id,
                    name=normalized_name,
                    entity_type=EntityType.APPLICATION,
                    source_doc=source_doc,
                    confidence=0.85
                ))
        return entities
    
    def _extract_industries(self, text: str, source_doc: Optional[str]) -> List[Entity]:
        """Extract industry entities."""
        entities = []
        for pattern, normalized_name in self._compiled_industries.items():
            if pattern.search(text):
                entity_id = f"industry:{normalized_name.lower().replace(' ', '_').replace('&', 'and')}"
                entities.append(Entity(
                    id=entity_id,
                    name=normalized_name,
                    entity_type=EntityType.INDUSTRY,
                    source_doc=source_doc,
                    confidence=0.85
                ))
        return entities
    
    def _extract_standards(self, text: str, source_doc: Optional[str]) -> List[Entity]:
        """Extract standard/specification entities."""
        entities = []
        for pattern, _ in self._compiled_standards.items():
            matches = pattern.findall(text)
            for match in matches:
                normalized = match.upper().replace(' ', '')
                entity_id = f"standard:{normalized.lower()}"
                entities.append(Entity(
                    id=entity_id,
                    name=normalized,
                    entity_type=EntityType.STANDARD,
                    source_doc=source_doc,
                    confidence=0.9
                ))
        return entities
    
    def extract_relationships(
        self, 
        entities: List[Entity], 
        text: str
    ) -> List[Relationship]:
        """
        Extract relationships between entities based on text context.
        
        Args:
            entities: List of extracted entities
            text: Original text for context
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Create entity lookup by type
        by_type: Dict[EntityType, List[Entity]] = {}
        for entity in entities:
            by_type.setdefault(entity.entity_type, []).append(entity)
        
        products = by_type.get(EntityType.PRODUCT, [])
        materials = by_type.get(EntityType.MATERIAL, [])
        applications = by_type.get(EntityType.APPLICATION, [])
        industries = by_type.get(EntityType.INDUSTRY, [])
        standards = by_type.get(EntityType.STANDARD, [])
        
        # Products -> Materials (MADE_OF)
        for product in products:
            for material in materials:
                relationships.append(Relationship(
                    source_id=product.id,
                    target_id=material.id,
                    relation_type=RelationType.MADE_OF,
                    confidence=0.8
                ))
        
        # Products -> Applications (USED_IN)
        for product in products:
            for application in applications:
                relationships.append(Relationship(
                    source_id=product.id,
                    target_id=application.id,
                    relation_type=RelationType.USED_IN,
                    confidence=0.75
                ))
        
        # Products -> Industries (SUITABLE_FOR)
        for product in products:
            for industry in industries:
                relationships.append(Relationship(
                    source_id=product.id,
                    target_id=industry.id,
                    relation_type=RelationType.SUITABLE_FOR,
                    confidence=0.7
                ))
        
        # Products -> Standards (COMPLIES_WITH)
        for product in products:
            for standard in standards:
                relationships.append(Relationship(
                    source_id=product.id,
                    target_id=standard.id,
                    relation_type=RelationType.COMPLIES_WITH,
                    confidence=0.85
                ))
        
        logger.debug(f"Extracted {len(relationships)} relationships")
        return relationships
    
    def extract_properties(self, text: str) -> Dict[str, Any]:
        """
        Extract properties/specifications from text.
        
        Args:
            text: Text to extract properties from
            
        Returns:
            Dictionary of extracted properties
        """
        properties = {}
        
        # Temperature
        temp_match = re.search(r'temperature[:\s]+(?:up\s+to\s+)?(-?\d+)\s*°?\s*(C|F)', text, re.IGNORECASE)
        if temp_match:
            properties["max_temperature"] = int(temp_match.group(1))
            properties["temperature_unit"] = temp_match.group(2).upper()
        
        # Pressure
        pressure_match = re.search(r'pressure[:\s]+(?:up\s+to\s+)?(\d+[\d,]*)\s*(psi|bar|MPa)', text, re.IGNORECASE)
        if pressure_match:
            properties["max_pressure"] = int(pressure_match.group(1).replace(',', ''))
            properties["pressure_unit"] = pressure_match.group(2)
        
        # pH range
        ph_match = re.search(r'pH[:\s]+(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if ph_match:
            properties["ph_min"] = float(ph_match.group(1))
            properties["ph_max"] = float(ph_match.group(2))
        
        return properties

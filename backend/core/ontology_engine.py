"""
Three-Layer Fashion Ontology System
Layer 1: Lexical - Language, raw words, phrases,aliases
Layer 2: Concept - Semantic meaninng, relationships( Core Ontology)
Layer 3: Instance - Real Product Objects

"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict
import re


# LAYER 1: LEXICAL LAYER
@dataclass
class LexicalTerm:
    """Represent a term in the lexical layer"""
    canonical_form: str  # Normalized form
    aliases: Set[str] = field(default_factory=set)
    languages: Set[str] = field(default_factory=set)
    variations: Set[str] = field(default_factory=set)  # spelling variations 

    def matches(self, text: str) -> bool:
        """Check if text matches this term or its aliases"""
        normalized = self._normalize(text)
        return normalized in {self.canonical_form, *self.aliases, *self.variations}
    
    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison"""
        return text.lower().strip().replace('-', '_').replace(' ', '_')
    

class LexicalLayer:
    """
    Lexical Layer: Handles linguistic variations and term normalization
    Responsibility: Convert raw text to canonical forms
    """
    
    def __init__(self):
        self.terms: Dict[str, LexicalTerm] = {}
        self.reverse_index: Dict[str, str] = {}  # alias -> canonical
        self._initialize_fashion_vocabulary()
    
    def _initialize_fashion_vocabulary(self):
        """Initialize with core fashion vocabulary"""
        vocabulary = [
            LexicalTerm("sleeve", {"arm_cover", "manga", "manche"}, {"en", "es", "fr"}),
            LexicalTerm("sleeveless", {"no_sleeve", "tank", "sans_manches"}, {"en", "fr"}),
            LexicalTerm("cold_shoulder", {"open_shoulder", "cut_out_shoulder", "shoulder_cutout"}, {"en"}),
            LexicalTerm("oversized", {"baggy", "loose_fit", "relaxed", "loose"}, {"en"}),
            LexicalTerm("fitted", {"tight", "slim_fit", "body_hugging", "skinny"}, {"en"}),
            LexicalTerm("floral", {"flower_print", "botanical", "flowers"}, {"en"}),
            LexicalTerm("denim", {"jean_material", "jeans", "denim_fabric"}, {"en"}),
            LexicalTerm("maxi", {"long_length", "floor_length", "ankle_length"}, {"en"}),
            LexicalTerm("mini", {"short", "short_length", "above_knee"}, {"en"}),
            LexicalTerm("midi", {"mid_length", "below_knee", "calf_length"}, {"en"}),
            LexicalTerm("v_neck", {"v_neckline", "vneck", "v-neck"}, {"en"}),
            LexicalTerm("crew_neck", {"crew", "round_neck", "crewneck"}, {"en"}),
            LexicalTerm("button_up", {"button_down", "button_front", "buttoned"}, {"en"}),
        ]
        
        for term in vocabulary:
            self.add_term(term)
    
    def add_term(self, term: LexicalTerm):
        """Add a term to the lexical layer"""
        self.terms[term.canonical_form] = term
        self.reverse_index[term.canonical_form] = term.canonical_form
        for alias in term.aliases:
            self.reverse_index[alias] = term.canonical_form
    
    def normalize(self, text: str) -> str:
        """Normalize text to canonical form"""
        normalized = LexicalTerm._normalize(text)
        return self.reverse_index.get(normalized, normalized)
    
    def extract_terms(self, text: str) -> List[str]:
        """Extract and normalize terms from text"""
        words = re.findall(r'\w+', text.lower())
        # Try multi-word combinations first
        terms = []
        i = 0
        while i < len(words):
            # Try 3-word, 2-word, then 1-word combinations
            for n in [3, 2, 1]:
                if i + n <= len(words):
                    phrase = '_'.join(words[i:i+n])
                    if phrase in self.reverse_index:
                        terms.append(self.normalize(phrase))
                        i += n
                        break
            else:
                i += 1
        return list(set(terms))


# LAYER 2: CONCEPT LAYER (ONTOLOGY)

class ConceptType(Enum):
    """Types of concepts in the ontology"""
    CATEGORY = "category"
    ATTRIBUTE = "attribute"
    VALUE = "value"


@dataclass
class Concept:
    """Represents a concept in the ontology"""
    id: str
    name: str
    type: ConceptType
    parent: Optional[str] = None
    children: Set[str] = field(default_factory=set)
    properties: Dict[str, any] = field(default_factory=dict)
    lexical_mappings: Set[str] = field(default_factory=set)  # Links to lexical terms
    
    def add_child(self, child_id: str):
        """Add a child concept"""
        self.children.add(child_id)


class ConceptLayer:
    """
    Concept Layer: Core Ontology defining semantic relationships
    Responsibility: Define what things mean and how they relate
    """
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.root_concepts: Set[str] = set()
        self._initialize_fashion_ontology()
    
    def _initialize_fashion_ontology(self):
        """Initialize the fashion ontology structure"""
        
        # Top-level categories
        self._add_concept("garment", "Garment", ConceptType.CATEGORY)
        self._add_concept("attribute", "Attribute", ConceptType.CATEGORY)
        
        # Garment subcategories
        garment_types = {
            "upper_body": ["shirt", "blouse", "t_shirt", "dress", "top"],
            "lower_body": ["pants", "jeans", "skirt", "shorts"],
            "outerwear": ["jacket", "coat", "cardigan", "blazer"],
            "footwear": ["sneakers", "boots", "sandals", "heels"]
        }
        
        for category, items in garment_types.items():
            self._add_concept(category, category.replace('_', ' ').title(), 
                            ConceptType.CATEGORY, parent="garment")
            for item in items:
                self._add_concept(item, item.replace('_', ' ').title(), 
                                ConceptType.CATEGORY, parent=category)
        
        # Attribute types
        attributes = {
            "sleeve_length": ["sleeveless", "short_sleeve", "three_quarter", "long_sleeve"],
            "neckline": ["crew_neck", "v_neck", "scoop_neck", "cold_shoulder", "off_shoulder"],
            "fit": ["slim", "regular", "oversized", "tailored", "loose"],
            "length": ["mini", "midi", "maxi", "knee_length", "ankle_length"],
            "pattern": ["solid", "striped", "floral", "geometric", "animal_print"],
            "material": ["cotton", "denim", "silk", "polyester", "leather", "wool"],
            "closure": ["button", "zipper", "pullover", "tie", "snap"]
        }
        
        for attr_type, values in attributes.items():
            self._add_concept(attr_type, attr_type.replace('_', ' ').title(), 
                            ConceptType.ATTRIBUTE, parent="attribute")
            for value in values:
                self._add_concept(f"{attr_type}_{value}", value.replace('_', ' ').title(), 
                                ConceptType.VALUE, parent=attr_type)
                # Add lexical mapping
                self.concepts[f"{attr_type}_{value}"].lexical_mappings.add(value)
    
    def _add_concept(self, id: str, name: str, type: ConceptType, parent: Optional[str] = None):
        """Add a concept to the ontology"""
        concept = Concept(id, name, type, parent)
        self.concepts[id] = concept
        
        if parent:
            if parent in self.concepts:
                self.concepts[parent].add_child(id)
        else:
            self.root_concepts.add(id)
    
    def get_concept_path(self, concept_id: str) -> List[str]:
        """Get the full path from root to concept"""
        path = []
        current = concept_id
        while current:
            if current in self.concepts:
                path.insert(0, self.concepts[current].name)
                current = self.concepts[current].parent
            else:
                break
        return path
    
    def find_concept_by_lexical(self, lexical_term: str) -> List[str]:
        """Find concepts that map to a lexical term"""
        matches = []
        for concept_id, concept in self.concepts.items():
            if lexical_term in concept.lexical_mappings or lexical_term == concept.id:
                matches.append(concept_id)
        return matches


# LAYER 3: INSTANCE LAYER

@dataclass
class ProductInstance:
    """Represents a real product instance"""
    id: str
    name: str
    category_id: str
    department_id: str
    brand: str
    description: str
    image_url: Optional[str] = None
    
    # Extracted features mapped to ontology concepts
    features: Dict[str, str] = field(default_factory=dict)  # concept_type -> concept_id
    
    # Raw extracted terms (for debugging/audit)
    raw_visual_features: List[str] = field(default_factory=list)
    raw_textual_features: List[str] = field(default_factory=list)
    
    # Metadata
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    extraction_timestamp: Optional[str] = None


class InstanceLayer:
    """
    Instance Layer: Stores real product instances
    Responsibility: Manage concrete product data with ontology mappings
    """
    
    def __init__(self):
        self.instances: Dict[str, ProductInstance] = {}
    
    def add_instance(self, instance: ProductInstance):
        """Add a product instance"""
        self.instances[instance.id] = instance
    
    def get_instance(self, product_id: str) -> Optional[ProductInstance]:
        """Retrieve a product instance"""
        return self.instances.get(product_id)
    
    def query_by_features(self, feature_query: Dict[str, str]) -> List[ProductInstance]:
        """Query products by ontology features"""
        results = []
        for instance in self.instances.values():
            if all(instance.features.get(k) == v for k, v in feature_query.items()):
                results.append(instance)
        return results


# MULTIMODAL FEATURE EXTRACTION ENGINE

class MultimodalFeatureExtractor:
    """
    Orchestrates feature extraction across all three layers
    """
    
    def __init__(self):
        self.lexical_layer = LexicalLayer()
        self.concept_layer = ConceptLayer()
        self.instance_layer = InstanceLayer()
    
    def extract_features(self, product_data: Dict) -> ProductInstance:
        """
        Extract features from product data (text + image)
        Steps:
        1. Extract raw terms from text (Lexical Layer)
        2. Map terms to concepts (Concept Layer)
        3. Create product instance (Instance Layer)
        """
        
        # Step 1: Lexical extraction
        text = f"{product_data.get('name', '')} {product_data.get('description', '')}"
        lexical_terms = self.lexical_layer.extract_terms(text)
        
        # Step 2: Concept mapping
        feature_map = {}
        for term in lexical_terms:
            concepts = self.concept_layer.find_concept_by_lexical(term)
            for concept_id in concepts:
                concept = self.concept_layer.concepts[concept_id]
                if concept.parent and concept.parent in self.concept_layer.concepts:
                    parent_concept = self.concept_layer.concepts[concept.parent]
                    feature_map[parent_concept.id] = concept_id
        
        # Step 3: Create instance
        instance = ProductInstance(
            id=product_data['product_id'],
            name=product_data.get('product_name', ''),
            category_id=product_data.get('category_id', ''),
            department_id=product_data.get('department_id', ''),
            brand=product_data.get('brand', ''),
            description=product_data.get('description', ''),
            image_url=product_data.get('feature_image', ''),
            features=feature_map,
            raw_textual_features=lexical_terms
        )
        
        self.instance_layer.add_instance(instance)
        return instance
    
    def get_ontology_summary(self) -> Dict:
        """Get summary of the ontology structure"""
        return {
            "lexical_terms": len(self.lexical_layer.terms),
            "concepts": len(self.concept_layer.concepts),
            "instances": len(self.instance_layer.instances),
            "concept_tree": self._build_concept_tree()
        }
    
    def _build_concept_tree(self) -> Dict:
        """Build a hierarchical tree of concepts"""
        tree = {}
        for root_id in self.concept_layer.root_concepts:
            tree[root_id] = self._build_subtree(root_id)
        return tree
    
    def _build_subtree(self, concept_id: str) -> Dict:
        """Recursively build concept subtree"""
        concept = self.concept_layer.concepts[concept_id]
        subtree = {
            "name": concept.name,
            "type": concept.type.value,
            "children": {}
        }
        for child_id in concept.children:
            subtree["children"][child_id] = self._build_subtree(child_id)
        return subtree


# DEMO/TESTING

def demo():
    """Demonstrate the three-layer system"""
    
    extractor = MultimodalFeatureExtractor()
    
    # Sample product data
    sample_products = [
        {
            "product_id": "P001",
            "product_name": "Floral Cold Shoulder Maxi Dress",
            "description": "Beautiful floral print dress with cold shoulder design, perfect for summer",
            "category_id": "dress_001",
            "department_id": "womens",
            "brand": "Fashion Brand",
            "feature_image": "https://example.com/dress1.jpg"
        },
        {
            "product_id": "P002",
            "product_name": "Oversized Denim Button-Up Shirt",
            "description": "Relaxed fit denim shirt with button front closure",
            "category_id": "shirt_001",
            "department_id": "womens",
            "brand": "Casual Co",
            "feature_image": "https://example.com/shirt1.jpg"
        }
    ]
    
    print("=" * 80)
    print("FASHION ONTOLOGY SYSTEM - THREE LAYER DEMO")
    print("=" * 80)
    
    for product_data in sample_products:
        print(f"\n\nProcessing: {product_data['product_name']}")
        print("-" * 80)
        
        instance = extractor.extract_features(product_data)
        
        print("\nLayer 1 - LEXICAL: Extracted Terms")
        print(f"  Terms: {', '.join(instance.raw_textual_features)}")
        
        print("\nLayer 2 - CONCEPT: Ontology Mappings")
        for feature_type, concept_id in instance.features.items():
            path = extractor.concept_layer.get_concept_path(concept_id)
            print(f"  {feature_type}: {' > '.join(path)}")
        
        print("\nLayer 3 - INSTANCE: Product Record Created")
        print(f"  ID: {instance.id}")
        print(f"  Brand: {instance.brand}")
        print(f"  Features Mapped: {len(instance.features)}")
    
    print("\n\n" + "=" * 80)
    print("ONTOLOGY SUMMARY")
    print("=" * 80)
    summary = extractor.get_ontology_summary()
    print(f"Lexical Terms: {summary['lexical_terms']}")
    print(f"Ontology Concepts: {summary['concepts']}")
    print(f"Product Instances: {summary['instances']}")


if __name__ == "__main__":
    demo()
# core/learning_system.py
"""
Feedback & Learning Loop System
Implements continuous learning and improvement mechanisms
Human-in-the-Loop (HITL) integration for ontology evolution
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from collections import defaultdict, Counter
import numpy as np


class FeedbackType(Enum):
    """Types of feedback in the system"""
    CORRECTION = "correction"  # User corrects a feature
    VALIDATION = "validation"  # User validates correct extraction
    ADDITION = "addition"      # User adds missing feature
    DELETION = "deletion"      # User removes incorrect feature
    NEW_TERM = "new_term"      # New terminology discovered


class FeedbackSource(Enum):
    """Source of feedback"""
    HUMAN_EXPERT = "human_expert"
    USER = "user"
    AUTOMATED = "automated"
    QUALITY_CHECK = "quality_check"


@dataclass
class Feedback:
    """Represents a single feedback instance"""
    id: str
    product_id: str
    feedback_type: FeedbackType
    source: FeedbackSource
    timestamp: datetime
    
    # Original extraction
    original_feature: Optional[str] = None
    original_value: Optional[str] = None
    
    # Corrected/suggested value
    corrected_feature: Optional[str] = None
    corrected_value: Optional[str] = None
    
    # Additional context
    confidence: float = 1.0
    notes: Optional[str] = None
    expert_id: Optional[str] = None
    
    # Learning metadata
    applied: bool = False
    impact_score: float = 0.0


@dataclass
class LearningPattern:
    """Represents a learned pattern from feedback"""
    pattern_id: str
    pattern_type: str  # e.g., "common_error", "new_term", "synonym"
    
    # Pattern details
    trigger: str  # What triggers this pattern
    action: str   # What action to take
    
    # Evidence
    occurrences: int = 0
    success_rate: float = 0.0
    confidence: float = 0.0
    
    # Examples that support this pattern
    examples: List[str] = field(default_factory=list)
    
    # When to apply
    min_confidence: float = 0.7
    active: bool = True


class FeedbackCollector:
    """
    Collects and manages feedback from various sources
    """
    
    def __init__(self):
        self.feedbacks: Dict[str, Feedback] = {}
        self.feedback_by_product: Dict[str, List[str]] = defaultdict(list)
        self.feedback_by_feature: Dict[str, List[str]] = defaultdict(list)
        self.pending_review: List[str] = []
        
        # Statistics
        self.stats = {
            'total_feedbacks': 0,
            'by_type': defaultdict(int),
            'by_source': defaultdict(int),
            'corrections_applied': 0,
            'new_terms_discovered': 0
        }
    
    def add_feedback(self, feedback: Feedback):
        """Add new feedback to the system"""
        self.feedbacks[feedback.id] = feedback
        self.feedback_by_product[feedback.product_id].append(feedback.id)
        
        if feedback.original_feature:
            self.feedback_by_feature[feedback.original_feature].append(feedback.id)
        
        # Update statistics
        self.stats['total_feedbacks'] += 1
        self.stats['by_type'][feedback.feedback_type.value] += 1
        self.stats['by_source'][feedback.source.value] += 1
        
        # Add to pending review if not automated
        if feedback.source != FeedbackSource.AUTOMATED:
            self.pending_review.append(feedback.id)
    
    def get_product_feedbacks(self, product_id: str) -> List[Feedback]:
        """Get all feedback for a specific product"""
        feedback_ids = self.feedback_by_product.get(product_id, [])
        return [self.feedbacks[fid] for fid in feedback_ids]
    
    def get_feature_feedbacks(self, feature: str) -> List[Feedback]:
        """Get all feedback for a specific feature"""
        feedback_ids = self.feedback_by_feature.get(feature, [])
        return [self.feedbacks[fid] for fid in feedback_ids]
    
    def get_pending_review(self, limit: int = 10) -> List[Feedback]:
        """Get feedbacks pending expert review"""
        return [self.feedbacks[fid] for fid in self.pending_review[:limit]]


class PatternLearner:
    """
    Learns patterns from feedback to improve future extractions
    """
    
    def __init__(self):
        self.patterns: Dict[str, LearningPattern] = {}
        self.pattern_history: List[Dict] = []
        
    def discover_patterns(self, feedbacks: List[Feedback]) -> List[LearningPattern]:
        """
        Discover patterns from feedback data
        """
        discovered = []
        
        # 1. Common correction patterns
        corrections = [f for f in feedbacks if f.feedback_type == FeedbackType.CORRECTION]
        common_corrections = self._find_common_corrections(corrections)
        discovered.extend(common_corrections)
        
        # 2. New terminology patterns
        new_terms = [f for f in feedbacks if f.feedback_type == FeedbackType.NEW_TERM]
        term_patterns = self._discover_new_term_patterns(new_terms)
        discovered.extend(term_patterns)
        
        # 3. Context-based patterns
        contextual = self._discover_contextual_patterns(feedbacks)
        discovered.extend(contextual)
        
        # Add discovered patterns
        for pattern in discovered:
            self.add_pattern(pattern)
        
        return discovered
    
    def _find_common_corrections(self, corrections: List[Feedback]) -> List[LearningPattern]:
        """Find frequently corrected mistakes"""
        patterns = []
        
        # Group by original -> corrected mapping
        correction_map = defaultdict(list)
        for fb in corrections:
            key = (fb.original_feature, fb.original_value)
            correction_map[key].append((fb.corrected_feature, fb.corrected_value))
        
        # Identify patterns with multiple occurrences
        for (orig_feat, orig_val), corrections_list in correction_map.items():
            if len(corrections_list) >= 3:  # Minimum 3 occurrences
                # Find most common correction
                counter = Counter(corrections_list)
                most_common, count = counter.most_common(1)[0]
                
                pattern = LearningPattern(
                    pattern_id=f"correction_{orig_feat}_{orig_val}",
                    pattern_type="common_error",
                    trigger=f"{orig_feat}:{orig_val}",
                    action=f"suggest_{most_common[0]}:{most_common[1]}",
                    occurrences=count,
                    success_rate=count / len(corrections_list),
                    confidence=min(count / 10.0, 1.0),
                    examples=[f"Corrected {count} times"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _discover_new_term_patterns(self, new_terms: List[Feedback]) -> List[LearningPattern]:
        """Discover patterns in new terminology"""
        patterns = []
        
        # Group new terms
        term_clusters = defaultdict(list)
        for fb in new_terms:
            if fb.corrected_value:
                term_clusters[fb.corrected_feature].append(fb.corrected_value)
        
        # Create patterns for frequently added terms
        for feature, terms in term_clusters.items():
            if len(terms) >= 2:
                term_counter = Counter(terms)
                for term, count in term_counter.items():
                    if count >= 2:
                        pattern = LearningPattern(
                            pattern_id=f"new_term_{feature}_{term}",
                            pattern_type="new_term",
                            trigger=f"missing_{feature}",
                            action=f"add_term:{term}",
                            occurrences=count,
                            confidence=min(count / 5.0, 1.0),
                            examples=[f"Added {count} times by experts"]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _discover_contextual_patterns(self, feedbacks: List[Feedback]) -> List[LearningPattern]:
        """Discover context-dependent patterns"""
        patterns = []
        
        # Example: If product has "dress" in name, certain features are more likely
        # This is simplified - in production, use more sophisticated ML
        
        return patterns  # Placeholder for advanced pattern discovery
    
    def add_pattern(self, pattern: LearningPattern):
        """Add a learned pattern"""
        self.patterns[pattern.pattern_id] = pattern
        self.pattern_history.append({
            'pattern_id': pattern.pattern_id,
            'discovered_at': datetime.now().isoformat(),
            'confidence': pattern.confidence
        })
    
    def apply_patterns(self, product_data: Dict, extracted_features: Dict) -> Dict:
        """Apply learned patterns to improve extraction"""
        improved_features = extracted_features.copy()
        suggestions = []
        
        for pattern_id, pattern in self.patterns.items():
            if not pattern.active or pattern.confidence < pattern.min_confidence:
                continue
            
            # Check if pattern applies
            if self._pattern_matches(pattern, product_data, extracted_features):
                suggestion = self._apply_pattern(pattern, improved_features)
                if suggestion:
                    suggestions.append(suggestion)
        
        improved_features['suggestions'] = suggestions
        improved_features['patterns_applied'] = len(suggestions)
        
        return improved_features
    
    def _pattern_matches(self, pattern: LearningPattern, 
                        product_data: Dict, features: Dict) -> bool:
        """Check if pattern should be applied"""
        # Simplified matching logic
        if pattern.pattern_type == "common_error":
            trigger_parts = pattern.trigger.split(':')
            if len(trigger_parts) == 2:
                feat, val = trigger_parts
                return features.get(feat) == val
        
        return False
    
    def _apply_pattern(self, pattern: LearningPattern, features: Dict) -> Dict:
        """Apply pattern and return suggestion"""
        if pattern.pattern_type == "common_error":
            action_parts = pattern.action.split('_', 1)[1].split(':')
            if len(action_parts) == 2:
                return {
                    'type': 'correction',
                    'feature': action_parts[0],
                    'suggested_value': action_parts[1],
                    'confidence': pattern.confidence,
                    'reason': f"Based on {pattern.occurrences} similar corrections"
                }
        
        return None


class OntologyEvolver:
    """
    Evolves the ontology based on learned patterns
    Manages ontology growth and updates
    """
    
    def __init__(self, concept_layer, lexical_layer):
        self.concept_layer = concept_layer
        self.lexical_layer = lexical_layer
        self.evolution_history: List[Dict] = []
        
    def evolve_from_patterns(self, patterns: List[LearningPattern]):
        """Evolve ontology based on learned patterns"""
        changes = []
        
        for pattern in patterns:
            if pattern.pattern_type == "new_term" and pattern.confidence > 0.7:
                change = self._add_new_term(pattern)
                if change:
                    changes.append(change)
            
            elif pattern.pattern_type == "common_error" and pattern.confidence > 0.8:
                change = self._add_synonym_mapping(pattern)
                if change:
                    changes.append(change)
        
        # Log evolution
        if changes:
            self.evolution_history.append({
                'timestamp': datetime.now().isoformat(),
                'changes': changes,
                'patterns_processed': len(patterns)
            })
        
        return changes
    
    def _add_new_term(self, pattern: LearningPattern) -> Optional[Dict]:
        """Add new term to lexical layer"""
        # Extract term from action
        if 'add_term:' in pattern.action:
            term = pattern.action.split('add_term:')[1]
            
            # Check if term already exists
            normalized = self.lexical_layer.normalize(term)
            if normalized == term:  # Not found, add it
                from core.ontology_engine import LexicalTerm
                new_term = LexicalTerm(term, set(), {"en"})
                self.lexical_layer.add_term(new_term)
                
                return {
                    'type': 'term_added',
                    'term': term,
                    'pattern_id': pattern.pattern_id,
                    'confidence': pattern.confidence
                }
        
        return None
    
    def _add_synonym_mapping(self, pattern: LearningPattern) -> Optional[Dict]:
        """Add synonym mapping based on common corrections"""
        # Simplified - in production, more sophisticated
        return None
    
    def get_evolution_stats(self) -> Dict:
        """Get statistics about ontology evolution"""
        return {
            'total_evolutions': len(self.evolution_history),
            'terms_added': sum(
                len([c for c in ev['changes'] if c['type'] == 'term_added'])
                for ev in self.evolution_history
            ),
            'latest_evolution': self.evolution_history[-1] if self.evolution_history else None
        }


class LearningEngine:
    """
    Main learning engine that orchestrates the feedback loop
    Integrates collection, pattern learning, and ontology evolution
    """
    
    def __init__(self, concept_layer, lexical_layer, instance_layer):
        self.concept_layer = concept_layer
        self.lexical_layer = lexical_layer
        self.instance_layer = instance_layer
        
        self.feedback_collector = FeedbackCollector()
        self.pattern_learner = PatternLearner()
        self.ontology_evolver = OntologyEvolver(concept_layer, lexical_layer)
        
        self.learning_cycles = 0
        self.improvements_made = 0
    
    def submit_feedback(self, feedback: Feedback):
        """Submit new feedback to the system"""
        self.feedback_collector.add_feedback(feedback)
        
        # Trigger learning if enough feedback accumulated
        if len(self.feedback_collector.pending_review) >= 10:
            self.run_learning_cycle()
    
    def run_learning_cycle(self):
        """
        Execute one learning cycle:
        1. Collect feedback
        2. Discover patterns
        3. Evolve ontology
        4. Apply improvements
        """
        print(f"\n{'='*80}")
        print(f"LEARNING CYCLE #{self.learning_cycles + 1}")
        print(f"{'='*80}")
        
        # 1. Get all feedbacks
        all_feedbacks = list(self.feedback_collector.feedbacks.values())
        print(f"Analyzing {len(all_feedbacks)} feedback instances...")
        
        # 2. Discover patterns
        patterns = self.pattern_learner.discover_patterns(all_feedbacks)
        print(f"Discovered {len(patterns)} new patterns")
        
        # 3. Evolve ontology
        changes = self.ontology_evolver.evolve_from_patterns(patterns)
        print(f"Applied {len(changes)} ontology changes")
        
        # 4. Update statistics
        self.learning_cycles += 1
        self.improvements_made += len(changes)
        
        # 5. Clear pending review
        self.feedback_collector.pending_review = []
        
        return {
            'cycle': self.learning_cycles,
            'patterns_discovered': len(patterns),
            'ontology_changes': len(changes),
            'improvements': changes
        }
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics"""
        return {
            'learning_cycles': self.learning_cycles,
            'total_improvements': self.improvements_made,
            'feedback_stats': self.feedback_collector.stats,
            'patterns_learned': len(self.pattern_learner.patterns),
            'ontology_evolution': self.ontology_evolver.get_evolution_stats(),
            'active_patterns': sum(
                1 for p in self.pattern_learner.patterns.values() 
                if p.active and p.confidence > 0.7
            )
        }
    
    def get_improvement_suggestions(self, product_id: str, 
                                   extracted_features: Dict) -> List[Dict]:
        """Get improvement suggestions for a product"""
        # Get product-specific feedback
        product_feedbacks = self.feedback_collector.get_product_feedbacks(product_id)
        
        suggestions = []
        
        # Check historical feedback
        for fb in product_feedbacks:
            if fb.feedback_type == FeedbackType.CORRECTION:
                suggestions.append({
                    'type': 'historical_correction',
                    'feature': fb.corrected_feature,
                    'value': fb.corrected_value,
                    'confidence': fb.confidence,
                    'source': fb.source.value
                })
        
        # Apply learned patterns
        pattern_suggestions = self.pattern_learner.apply_patterns(
            {'product_id': product_id},
            extracted_features
        ).get('suggestions', [])
        
        suggestions.extend(pattern_suggestions)
        
        return suggestions


# ============================================================================
# Demo/Testing
# ============================================================================

def demo_learning_system():
    """Demonstrate the learning system"""
    from core.ontology_engine import LexicalLayer, ConceptLayer, InstanceLayer
    
    print("=" * 80)
    print("FEEDBACK & LEARNING SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize layers
    lexical = LexicalLayer()
    concept = ConceptLayer()
    instance = InstanceLayer()
    
    # Initialize learning engine
    learning_engine = LearningEngine(concept, lexical, instance)
    
    # Simulate feedback
    print("\n1. Simulating user feedback...")
    
    feedbacks = [
        Feedback(
            id="FB001",
            product_id="P001",
            feedback_type=FeedbackType.CORRECTION,
            source=FeedbackSource.HUMAN_EXPERT,
            timestamp=datetime.now(),
            original_feature="fit",
            original_value="regular",
            corrected_feature="fit",
            corrected_value="oversized",
            expert_id="expert_001"
        ),
        Feedback(
            id="FB002",
            product_id="P002",
            feedback_type=FeedbackType.NEW_TERM,
            source=FeedbackSource.HUMAN_EXPERT,
            timestamp=datetime.now(),
            corrected_feature="style",
            corrected_value="cottagecore",
            notes="New trending aesthetic",
            expert_id="expert_001"
        ),
        Feedback(
            id="FB003",
            product_id="P003",
            feedback_type=FeedbackType.CORRECTION,
            source=FeedbackSource.USER,
            timestamp=datetime.now(),
            original_feature="fit",
            original_value="regular",
            corrected_feature="fit",
            corrected_value="oversized"
        )
    ]
    
    for fb in feedbacks:
        learning_engine.submit_feedback(fb)
        print(f"  - Received {fb.feedback_type.value} from {fb.source.value}")
    
    # Run learning cycle
    print("\n2. Running learning cycle...")
    result = learning_engine.run_learning_cycle()
    
    print(f"\n3. Learning Results:")
    print(f"   Patterns discovered: {result['patterns_discovered']}")
    print(f"   Ontology changes: {result['ontology_changes']}")
    
    # Get statistics
    print("\n4. Learning Statistics:")
    stats = learning_engine.get_learning_stats()
    print(f"   Total cycles: {stats['learning_cycles']}")
    print(f"   Total improvements: {stats['total_improvements']}")
    print(f"   Active patterns: {stats['active_patterns']}")
    print(f"   Feedback collected: {stats['feedback_stats']['total_feedbacks']}")
    

if __name__ == "__main__":
    demo_learning_system()
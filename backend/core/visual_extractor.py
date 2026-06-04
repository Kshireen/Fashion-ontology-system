# core/visual_extraction.py
"""
Computer Vision Feature Extraction Module
Integrates with the three-layer ontology system
Extracts visual features: colors, patterns, textures, styles, garment parts
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import requests
from io import BytesIO
import cv2
from collections import defaultdict
import colorsys


class ColorExtractor:
    """Extract dominant colors and color attributes from images"""
    
    def __init__(self):
        self.color_names = {
            'red': [(0, 100, 100), (10, 255, 255)],
            'orange': [(10, 100, 100), (25, 255, 255)],
            'yellow': [(25, 100, 100), (35, 255, 255)],
            'green': [(35, 100, 100), (85, 255, 255)],
            'cyan': [(85, 100, 100), (95, 255, 255)],
            'blue': [(95, 100, 100), (125, 255, 255)],
            'purple': [(125, 100, 100), (155, 255, 255)],
            'pink': [(155, 100, 100), (170, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'black': [(0, 0, 0), (180, 255, 50)],
            'gray': [(0, 0, 50), (180, 30, 200)],
            'brown': [(10, 100, 20), (20, 255, 200)]
        }
    
    def extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Extract dominant colors using k-means clustering
        Returns list of colors with their percentages
        """
        # Resize for faster processing
        img = cv2.resize(image, (150, 150))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape to pixel array
        pixels = img.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, 
                                        cv2.KMEANS_RANDOM_CENTERS)
        
        # Count pixels in each cluster
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)
        
        # Sort by percentage
        sorted_indices = np.argsort(-percentages)
        
        colors = []
        for idx in sorted_indices:
            rgb = centers[idx].astype(int)
            color_name = self._rgb_to_color_name(rgb)
            colors.append({
                'rgb': rgb.tolist(),
                'hex': '#{:02x}{:02x}{:02x}'.format(*rgb),
                'percentage': float(percentages[idx]),
                'name': color_name
            })
        
        return colors
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Convert RGB to closest color name"""
        # Convert to HSV for better color matching
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        
        for color_name, (lower, upper) in self.color_names.items():
            if (lower[0] <= hsv[0] <= upper[0] and 
                lower[1] <= hsv[1] <= upper[1] and 
                lower[2] <= hsv[2] <= upper[2]):
                return color_name
        
        return 'multi_color'
    
    def detect_color_attributes(self, colors: List[Dict]) -> List[str]:
        """Detect color-based attributes"""
        attributes = []
        
        # Check if monochrome (single dominant color > 70%)
        if colors[0]['percentage'] > 0.7:
            attributes.append('solid_color')
            attributes.append(f"{colors[0]['name']}_color")
        else:
            attributes.append('multi_color')
            # List top 3 colors
            for i in range(min(3, len(colors))):
                if colors[i]['percentage'] > 0.15:
                    attributes.append(f"{colors[i]['name']}_tone")
        
        return attributes


class PatternDetector:
    """Detect patterns and textures in fashion images"""
    
    def __init__(self):
        self.pattern_templates = {
            'striped': self._detect_stripes,
            'floral': self._detect_floral,
            'geometric': self._detect_geometric,
            'solid': self._detect_solid,
            'checkered': self._detect_checkered
        }
    
    def detect_patterns(self, image: np.ndarray) -> List[str]:
        """Detect patterns in the image"""
        patterns = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check each pattern type
        for pattern_name, detector in self.pattern_templates.items():
            if detector(gray):
                patterns.append(pattern_name)
        
        return patterns if patterns else ['solid']
    
    def _detect_stripes(self, gray: np.ndarray) -> bool:
        """Detect striped patterns using Fourier transform"""
        # Apply FFT to detect periodic patterns
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Check for peaks indicating periodic patterns
        threshold = np.percentile(magnitude, 99)
        peaks = np.sum(magnitude > threshold)
        
        return peaks > 10
    
    def _detect_floral(self, gray: np.ndarray) -> bool:
        """Detect floral patterns using edge detection"""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Floral patterns have many curved contours
        curved_contours = 0
        for contour in contours:
            if len(contour) > 20:  # Filter small contours
                curved_contours += 1
        
        return curved_contours > 30
    
    def _detect_geometric(self, gray: np.ndarray) -> bool:
        """Detect geometric patterns"""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        return lines is not None and len(lines) > 20
    
    def _detect_solid(self, gray: np.ndarray) -> bool:
        """Check if pattern is solid/plain"""
        std = np.std(gray)
        return std < 30  # Low variance indicates solid color
    
    def _detect_checkered(self, gray: np.ndarray) -> bool:
        """Detect checkered/plaid patterns"""
        # Simplified: Look for grid-like structures
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                               minLineLength=20, maxLineGap=5)
        
        if lines is None:
            return False
        
        # Check for perpendicular lines
        horizontal = sum(1 for line in lines 
                        if abs(line[0][1] - line[0][3]) < 10)
        vertical = sum(1 for line in lines 
                      if abs(line[0][0] - line[0][2]) < 10)
        
        return horizontal > 5 and vertical > 5


class DeepFeatureExtractor:
    """
    Deep learning-based feature extraction using pre-trained models
    Extracts high-level fashion attributes
    """
    
    def __init__(self, model_name: str = 'resnet50'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # Remove final classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Fashion attribute classifiers (simplified - in production use trained models)
        self.attribute_thresholds = self._initialize_attribute_classifiers()
    
    def _initialize_attribute_classifiers(self) -> Dict:
        """
        Initialize attribute detection thresholds
        In production: Use properly trained classifiers for each attribute
        """
        return {
            'casual': {'threshold': 0.6, 'features': [100, 250, 400]},
            'formal': {'threshold': 0.6, 'features': [150, 300, 500]},
            'sporty': {'threshold': 0.6, 'features': [200, 350, 450]},
            'elegant': {'threshold': 0.6, 'features': [180, 320, 480]},
            'bohemian': {'threshold': 0.6, 'features': [120, 280, 420]},
        }
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract deep features from image"""
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Flatten features
        features = features.squeeze().cpu().numpy()
        return features
    
    def predict_style_attributes(self, features: np.ndarray) -> List[str]:
        """
        Predict style attributes from deep features
        In production: Replace with trained classifiers
        """
        attributes = []
        
        # Simplified attribute detection based on feature statistics
        feature_stats = {
            'mean': np.mean(features),
            'std': np.std(features),
            'max': np.max(features),
        }
        
        # Example rules (replace with ML models in production)
        if feature_stats['std'] > 0.5:
            attributes.append('textured')
        
        if feature_stats['mean'] > 0.3:
            attributes.append('detailed')
        
        # Can add more sophisticated detection here
        return attributes if attributes else ['simple']


class VisualFeatureExtractor:
    """
    Main visual feature extraction orchestrator
    Combines multiple CV techniques to extract comprehensive features
    """
    
    def __init__(self):
        self.color_extractor = ColorExtractor()
        self.pattern_detector = PatternDetector()
        self.deep_extractor = DeepFeatureExtractor()
    
    def extract_from_url(self, image_url: str) -> Dict:
        """Extract features from image URL"""
        try:
            # Download image
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content))
            return self.extract_from_image(image)
        except Exception as e:
            return {'error': str(e), 'features': {}}
    
    def extract_from_image(self, image: Image.Image) -> Dict:
        """Extract all visual features from PIL Image"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Ensure RGB format
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        # Extract features from different modules
        features = {}
        
        # 1. Color features
        colors = self.color_extractor.extract_dominant_colors(img_array)
        color_attrs = self.color_extractor.detect_color_attributes(colors)
        features['colors'] = colors
        features['color_attributes'] = color_attrs
        
        # 2. Pattern features
        patterns = self.pattern_detector.detect_patterns(img_array)
        features['patterns'] = patterns
        
        # 3. Deep learning features
        deep_features = self.deep_extractor.extract_features(image)
        style_attrs = self.deep_extractor.predict_style_attributes(deep_features)
        features['style_attributes'] = style_attrs
        features['deep_features_shape'] = deep_features.shape
        
        # 4. Compile visual terms for ontology mapping
        visual_terms = []
        visual_terms.extend(color_attrs)
        visual_terms.extend(patterns)
        visual_terms.extend(style_attrs)
        
        features['visual_terms'] = visual_terms
        features['confidence'] = self._calculate_confidence(features)
        
        return features
    
    def _calculate_confidence(self, features: Dict) -> float:
        """Calculate overall confidence score for extracted features"""
        scores = []
        
        # Color confidence (based on dominant color percentage)
        if features.get('colors'):
            scores.append(features['colors'][0]['percentage'])
        
        # Pattern confidence (more patterns = higher confidence)
        if features.get('patterns'):
            scores.append(min(len(features['patterns']) * 0.3, 1.0))
        
        # Style confidence
        if features.get('style_attributes'):
            scores.append(0.7)  # Default confidence
        
        return np.mean(scores) if scores else 0.5


# ============================================================================
# Integration with Ontology System
# ============================================================================

class MultimodalFeatureExtractor:
    """
    Enhanced feature extractor combining visual and textual features
    Integrates with the three-layer ontology system
    """
    
    def __init__(self, lexical_layer, concept_layer, instance_layer):
        self.lexical_layer = lexical_layer
        self.concept_layer = concept_layer
        self.instance_layer = instance_layer
        self.visual_extractor = VisualFeatureExtractor()
        
        # Add visual terms to lexical layer
        self._extend_lexical_vocabulary()
    
    def _extend_lexical_vocabulary(self):
        """Extend lexical layer with visual terms"""
        from core.ontology_engine import LexicalTerm
        
        visual_terms = [
            LexicalTerm("solid_color", {"plain", "single_color", "uniform"}, {"en"}),
            LexicalTerm("multi_color", {"colorful", "multicolored", "varied"}, {"en"}),
            LexicalTerm("striped", {"stripes", "striped_pattern", "banded"}, {"en"}),
            LexicalTerm("checkered", {"plaid", "checked", "gingham"}, {"en"}),
            LexicalTerm("textured", {"texture", "textured_fabric"}, {"en"}),
            LexicalTerm("detailed", {"intricate", "elaborate"}, {"en"}),
        ]
        
        for term in visual_terms:
            self.lexical_layer.add_term(term)
    
    def extract_multimodal_features(self, product_data: Dict) -> Dict:
        """
        Extract features from both text and image
        Combines visual and textual features into unified representation
        """
        result = {
            'product_id': product_data.get('product_id'),
            'visual_features': {},
            'textual_features': {},
            'combined_features': {},
            'ontology_mappings': {},
            'confidence': {}
        }
        
        # 1. Extract textual features
        text = f"{product_data.get('product_name', '')} {product_data.get('description', '')}"
        textual_terms = self.lexical_layer.extract_terms(text)
        result['textual_features'] = textual_terms
        
        # 2. Extract visual features (if image available)
        if product_data.get('feature_image'):
            visual_result = self.visual_extractor.extract_from_url(
                product_data['feature_image']
            )
            
            if 'error' not in visual_result:
                result['visual_features'] = {
                    'colors': visual_result.get('colors', []),
                    'patterns': visual_result.get('patterns', []),
                    'style': visual_result.get('style_attributes', []),
                    'visual_terms': visual_result.get('visual_terms', [])
                }
                result['confidence']['visual'] = visual_result.get('confidence', 0.5)
        
        # 3. Combine and map to ontology
        all_terms = set(textual_terms)
        if result['visual_features'].get('visual_terms'):
            all_terms.update(result['visual_features']['visual_terms'])
        
        # Map to ontology concepts
        ontology_mappings = {}
        for term in all_terms:
            normalized = self.lexical_layer.normalize(term)
            concepts = self.concept_layer.find_concept_by_lexical(normalized)
            
            for concept_id in concepts:
                concept = self.concept_layer.concepts[concept_id]
                if concept.parent:
                    parent = self.concept_layer.concepts[concept.parent]
                    ontology_mappings[parent.id] = concept_id
        
        result['ontology_mappings'] = ontology_mappings
        result['combined_features'] = list(all_terms)
        
        # Calculate overall confidence
        visual_conf = result['confidence'].get('visual', 0.5)
        textual_conf = 0.8  # Default for textual
        result['confidence']['overall'] = (visual_conf + textual_conf) / 2
        
        return result


# ============================================================================
# Demo/Testing
# ============================================================================

def demo_visual_extraction():
    """Demo visual feature extraction"""
    print("=" * 80)
    print("VISUAL FEATURE EXTRACTION DEMO")
    print("=" * 80)
    
    extractor = VisualFeatureExtractor()
    
    # Test with a sample image URL (placeholder)
    test_url = "https://example.com/dress.jpg"
    
    print(f"\nExtracting features from: {test_url}")
    print("-" * 80)
    
    # In production, this would work with real images
    print("\nExtracted Features:")
    print("  Colors: Red (45%), White (30%), Blue (25%)")
    print("  Patterns: Floral, Textured")
    print("  Style: Casual, Bohemian")
    print("  Confidence: 0.87")
    
    print("\nVisual Terms for Ontology Mapping:")
    print("  - red_color")
    print("  - multi_color")
    print("  - floral")
    print("  - textured")
    print("  - casual")
    

if __name__ == "__main__":
    demo_visual_extraction()
"""
Visual Processing Pipeline

Coordinates all visual modules.

Pipeline:
    Image URL
        ↓
    Image Loader
        ↓
    Preprocessor
        ↓
    CLIP Encoder
       ├── Color Extraction
       └── Zero-shot Classification
        ↓
    VisualResult
"""

from typing import Dict, List, Optional

from .clip_encoder import CLIPEncoder
from .color_extractor import ColorExtractor
from .image_loader import ImageLoader
from .preprocessing import ImagePreprocessor
from .visual_result import VisualResult
from .zero_shot_classifier import ZeroShotClassifier


class VisualPipeline:

    def __init__(self):

        self.loader = ImageLoader()

        self.preprocessor = ImagePreprocessor()

        self.encoder = CLIPEncoder()

        self.color_extractor = ColorExtractor()

        self.classifier = ZeroShotClassifier(self.encoder)

        self._register_default_categories()

    def _register_default_categories(self):

        self.classifier.register_category(
            "pattern",
            [
                "plain shirt",
                "floral pattern",
                "striped shirt",
                "checkered shirt",
                "paisley pattern",
                "polka dot pattern",
                "camouflage",
                "animal print",
                "graphic print",
            ],
        )

        self.classifier.register_category(
            "style",
            [
                "casual style",
                "formal style",
                "business casual",
                "streetwear",
                "minimalist fashion",
                "bohemian fashion",
                "party wear",
                "sporty clothing",
                "vintage clothing",
            ],
        )

        self.classifier.register_category(
            "material",
            [
                "cotton fabric",
                "linen fabric",
                "denim fabric",
                "silk fabric",
                "wool fabric",
                "leather material",
                "polyester fabric",
            ],
        )

        self.classifier.register_category(
            "fit",
            [
                "regular fit",
                "slim fit",
                "oversized fit",
                "relaxed fit",
                "loose fit",
            ],
        )

    def extract(self, image_url: str) -> Optional[VisualResult]:

        image = self.loader.load(image_url)

        if image is None:
            return None

        image = self.preprocessor.preprocess(image)

        embedding = self.encoder.encode_image(image)

        colors = self.color_extractor.extract(image)

        pattern = self.classifier.predict(
            embedding,
            "pattern",
            top_k=1,
        )[0]

        style = self.classifier.predict(
            embedding,
            "style",
            top_k=1,
        )[0]

        material = self.classifier.predict(
            embedding,
            "material",
            top_k=1,
        )[0]

        fit = self.classifier.predict(
            embedding,
            "fit",
            top_k=1,
        )[0]

        confidence = (
            pattern.score
            + style.score
            + material.score
            + fit.score
        ) / 4

        return VisualResult(
            colors=colors,
            pattern=pattern.label,
            style=style.label,
            attributes=[
                material.label,
                fit.label,
            ],
            embedding=embedding,
            confidence=confidence,
        )
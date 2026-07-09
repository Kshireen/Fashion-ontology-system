"""
Zero-Shot Fashion Classifier

Uses CLIP embeddings for semantic image classification.

Instead of training separate classifiers,
we compare image embeddings with embeddings of
fashion concepts using cosine similarity.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .clip_encoder import CLIPEncoder


@dataclass
class Prediction:

    label: str
    score: float


class ZeroShotClassifier:

    def __init__(self, encoder: CLIPEncoder):

        self.encoder = encoder

        # Cache:
        # {
        #   "patterns": {
        #        "labels": [...],
        #        "embeddings": np.ndarray
        #   }
        # }

        self.embedding_cache: Dict = {}

    def register_category(
        self,
        category_name: str,
        labels: List[str],
    ) -> None:
        """
        Precompute embeddings for a category.
        """

        embeddings = self.encoder.encode_text(labels)

        self.embedding_cache[category_name] = {
            "labels": labels,
            "embeddings": embeddings,
        }

    def predict(
        self,
        image_embedding: np.ndarray,
        category_name: str,
        top_k: int = 3,
    ) -> List[Prediction]:

        if category_name not in self.embedding_cache:
            raise ValueError(
                f"Unknown category: {category_name}"
            )

        cache = self.embedding_cache[category_name]

        scores = self.encoder.cosine_similarity(
            image_embedding,
            cache["embeddings"],
        )

        order = np.argsort(scores)[::-1][:top_k]

        predictions = []

        for idx in order:

            predictions.append(
                Prediction(
                    label=cache["labels"][idx],
                    score=float(scores[idx]),
                )
            )

        return predictions
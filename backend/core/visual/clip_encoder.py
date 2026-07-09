"""
CLIP Encoder Module

Provides a lightweight wrapper around OpenCLIP for generating
image and text embeddings. These embeddings are later used for:

- Zero-shot attribute classification
- Pattern detection
- Style prediction
- Semantic retrieval
- Ontology mapping

CPU compatible.
"""

from typing import List, Union

import numpy as np
import open_clip
import torch
from PIL import Image


class CLIPEncoder:
    """
    Wrapper around OpenCLIP.

    Generates normalized embeddings for images and text.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = None,
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
            )
        )

        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a PIL image into a normalized embedding.

        Returns
        -------
        numpy.ndarray
            Shape: (embedding_dim,)
        """

        image_tensor = (
            self.preprocess(image)
            .unsqueeze(0)
            .to(self.device)
        )

        embedding = self.model.encode_image(image_tensor)

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_text(
        self,
        texts: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Encode one or more text prompts.

        Parameters
        ----------
        texts : str | List[str]

        Returns
        -------
        numpy.ndarray

        Shape:
            (embedding_dim,)
            OR
            (N, embedding_dim)
        """

        single = False

        if isinstance(texts, str):
            texts = [texts]
            single = True

        tokens = self.tokenizer(texts).to(self.device)

        embeddings = self.model.encode_text(tokens)

        embeddings = embeddings / embeddings.norm(
            dim=-1,
            keepdim=True,
        )

        embeddings = embeddings.cpu().numpy()

        if single:
            return embeddings[0]

        return embeddings

    @staticmethod
    def cosine_similarity(
        image_embedding: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between one image
        embedding and multiple text embeddings.

        Returns
        -------
        numpy.ndarray

        Shape:
            (N,)
        """

        return image_embedding @ text_embeddings.T
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class VisualResult:

    colors: List[str] = field(default_factory=list)

    pattern: str = ""

    style: str = ""

    attributes: List[str] = field(default_factory=list)

    embedding: np.ndarray | None = None

    confidence: float = 0.0

    def to_dict(self):

        return {
            "colors": self.colors,
            "pattern": self.pattern,
            "style": self.style,
            "attributes": self.attributes,
            "confidence": self.confidence,
        }
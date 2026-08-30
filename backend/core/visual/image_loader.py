from io import BytesIO
from typing import Optional

import requests
from PIL import Image


class ImageLoader:
    """Loads images from URLs."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def load(self, image_url: str) -> Optional[Image.Image]:

        if not image_url:
            return None

        try:
            response = requests.get(
                image_url,
                timeout=self.timeout,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            response.raise_for_status()

            image = Image.open(BytesIO(response.content))

            return image.convert("RGB")

        except Exception as e:

            print(f"[ImageLoader] {e}")

            return None
from PIL import Image

from core.visual.clip_encoder import CLIPEncoder
from core.visual.zero_shot_classifier import ZeroShotClassifier

encoder = CLIPEncoder()

classifier = ZeroShotClassifier(encoder)

classifier.register_category(
    "patterns",
    [
        "floral pattern",
        "striped pattern",
        "plain shirt",
        "checkered shirt",
        "paisley pattern",
        "polka dot pattern",
    ],
)

image = Image.open("shirt.jpg").convert("RGB")

embedding = encoder.encode_image(image)

predictions = classifier.predict(
    embedding,
    "patterns",
)

for prediction in predictions:
    print(prediction)
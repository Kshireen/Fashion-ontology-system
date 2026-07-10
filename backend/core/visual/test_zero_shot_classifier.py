from PIL import Image
import requests
from io import BytesIO

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

url = "https://img.ltwebstatic.com/images3_pi/2024/10/12/b8/1728726835f1344a1a3d996b904049c512fc61946d.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")

embedding = encoder.encode_image(image)

predictions = classifier.predict(
    embedding,
    "patterns",
)

for prediction in predictions:
    print(prediction)
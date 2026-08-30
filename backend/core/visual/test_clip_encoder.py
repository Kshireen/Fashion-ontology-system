from PIL import Image
import requests
from io import BytesIO
from PIL import Image
from core.visual.clip_encoder import CLIPEncoder

# 1. Fetch the image from the web URL
url = "https://img.ltwebstatic.com/images3_pi/2024/10/12/b8/1728726835f1344a1a3d996b904049c512fc61946d.jpg"
response = requests.get(url)

encoder = CLIPEncoder()

# image = Image.open("https://img.ltwebstatic.com/images3_pi/2024/10/12/b8/1728726835f1344a1a3d996b904049c512fc61946d.jpg").convert("RGB")

image = Image.open(BytesIO(response.content)).convert("RGB")

image_embedding = encoder.encode_image(image)

labels = [
    "floral shirt",
    "striped shirt",
    "plain shirt",
    "denim jacket",
]

text_embeddings = encoder.encode_text(labels)

scores = encoder.cosine_similarity(
    image_embedding,
    text_embeddings,
)

for label, score in zip(labels, scores):
    print(f"{label:20} {score:.3f}")

best = labels[scores.argmax()]

print("\nPrediction:", best)
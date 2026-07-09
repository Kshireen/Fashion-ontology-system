from PIL import Image


class ImagePreprocessor:

    def __init__(self, size=(224, 224)):
        self.size = size

    def preprocess(self, image: Image.Image):

        image = image.convert("RGB")

        image.thumbnail(self.size)

        return image
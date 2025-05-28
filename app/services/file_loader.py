from PIL import Image
import requests
from io import BytesIO

class FileLoader:
    def __init__(self, url):
        self.url = url

    def load_image(self):
        response = requests.get(self.url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img

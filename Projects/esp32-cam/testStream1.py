import requests
from PIL import Image
import io

response = requests.get("http://192.168.1.6/sustain?stream=0", auth=("ESPCAM", "1234"))
img = Image.open(io.BytesIO(response.content))
img.show()
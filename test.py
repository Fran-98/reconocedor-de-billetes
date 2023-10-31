from transformers import pipeline, set_seed
import joblib
from PIL import Image

img = Image.open('dataset/val/200.webp')
model = joblib.load('model.joblib')

print(model.predict(img))


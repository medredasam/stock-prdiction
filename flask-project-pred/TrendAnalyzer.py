import tensorflow as tf
import cv2
import numpy as np

class TrendAnalyzer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

def preprocess_image(self, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 1, 64, 64, 1)  # Corrigé ici
    return img

def predict(self, image_path):
    processed = self.preprocess_image(image_path)
    if processed is None:
        return None
    pred = self.model.predict(processed)[0][0]
    trend = "Uptrend" if pred > 0.5 else "Downtrend"
    confidence = max(pred, 1 - pred)
    return {
        'trend': trend,
        'confidence': round(confidence * 100, 2),
        'raw_score': float(pred)
    }

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import io
import os


class FaceInsightModel:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.target_size = (128, 128)  # MobileNetV2 input size

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert uploaded image bytes to model input format."""
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        image = image.resize(self.target_size)
        img_array = img_to_array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on an image."""
        img_array = self.preprocess_image(image_bytes)
        predictions = self.model.predict(img_array, verbose=0)

        age_pred = float(predictions[0][0][0])
        gender_prob = float(predictions[1][0][0])

        return {
            'age': round(age_pred),
            'gender': 'Male' if gender_prob > 0.5 else 'Female',
            'gender_confidence': round(gender_prob * 100 if gender_prob > 0.5 else (1 - gender_prob) * 100, 1)
        }


# Singleton instance
_model_instance = None


def get_model() -> FaceInsightModel:
    global _model_instance
    if _model_instance is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'face_insight_model.keras')
        _model_instance = FaceInsightModel(model_path)
    return _model_instance

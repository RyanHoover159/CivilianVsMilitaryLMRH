from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm
import numpy as np
import os

FEATURES_PATH = "data/features.npy"
LABELS_PATH = "data/labels.npy"

def save_features(features, labels):
    os.makedirs("data", exist_ok=True)
    np.save(FEATURES_PATH, features)
    np.save(LABELS_PATH, labels)
    print("Features and labels saved.")

def load_features():
    if os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH):
        print("Loading cached features and labels")
        features = np.load(FEATURES_PATH)
        labels = np.load(LABELS_PATH)
        return features, labels
    return None, None

def extract_features(images):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = []

    for img in tqdm(images, desc="Extracting features"):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            
        img = preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)
        feature = model.predict(img, verbose=0)[0]
        features.append(feature)

    return np.array(features)

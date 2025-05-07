from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

def load_and_preprocess_dataset(img_size=(224, 224), cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    images_path = os.path.join(cache_dir, "images.npy")
    labels_path = os.path.join(cache_dir, "labels.npy")

    if os.path.exists(images_path) and os.path.exists(labels_path):
        print("Loading cached dataset...")
        images = np.load(images_path)
        labels = np.load(labels_path)
        return images, labels


    dataset = load_dataset("Mr-Fox-h/Civil_or_Military")["train"]
    images = []
    labels = []

    for sample in tqdm(dataset, desc="Loading and preprocessing images"):
        image = sample["image"].convert("RGB")
        image = image.resize(img_size)
        image_array = np.array(image, dtype=np.uint8)
        images.append(image_array)
        labels.append(1 if sample["label"] == "military" else 0)

    images = np.array(images)
    labels = np.array(labels)

    np.save(images_path, images)
    np.save(labels_path, labels)
    print("Dataset cached for later uses.")

    return images, labels









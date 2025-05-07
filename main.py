from src.data_loader import load_and_preprocess_dataset
from src.feature_extractor import extract_features, save_features, load_features
from src.train import train_models
from src.evaluate import evaluate_models

def main():
    features, labels = load_features()

    if features is None or labels is None:
        print("Loading and preprocessing dataset")
        images, labels = load_and_preprocess_dataset()

        print("Extracting features")
        features = extract_features(images)

        save_features(features, labels)

    print("Training models")
    models = train_models(features, labels)

    print("Evaluating models")
    evaluate_models(models, features, labels)

if __name__ == "__main__":
    main()

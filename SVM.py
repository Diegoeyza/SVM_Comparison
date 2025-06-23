import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

## ------------------------------------------------------------------------- ##

def train_svm(feats_file, labels_file, save_path):
    print("Training SVM...")

    features = np.load(feats_file)
    labels = []
    with open(labels_file, "r") as f:
        for line in f:
            labels.append(line.strip())

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    clf = SVC(kernel='linear', probability=True)  # cambiemos el kernel si no da buenos resultados edu
    clf.fit(features, encoded_labels)

    joblib.dump({'model': clf, 'label_encoder': label_encoder}, save_path)
    print(f"SVM model saved to: {save_path}")

## ------------------------------------------------------------------------- ##

def val_svm(feats_file, labels_file, save_path):
    print("Evaluating SVM...")

    features = np.load(feats_file)
    labels = []
    with open(labels_file, "r") as f:
        for line in f:
            labels.append(line.strip())

    saved = joblib.load(save_path)
    clf = saved['model']
    label_encoder = saved['label_encoder']
    encoded_labels = label_encoder.transform(labels)

    predictions = clf.predict(features)

    acc = accuracy_score(encoded_labels, predictions)
    print(f"Accuracy: {acc:.4f}")

    print("\nPer-class performance:")
    print(classification_report(encoded_labels, predictions, target_names=label_encoder.classes_))

## ------------------------------------------------------------------------- ##

DATASET = 'VocVal'
MODEL = 'DINOv2'
DATA_DIR = 'VocPascal'

FEATS_FILE_TRAIN = f'data/feat_{MODEL}_{DATASET}_train.npy'
LABELS_FILE_TRAIN = f'data/labels_{MODEL}_{DATASET}_train.txt'

FEATS_FILE_VAL = f'data/feat_{MODEL}_{DATASET}.npy'
LABELS_FILE_VAL = f'data/labels_{MODEL}_{DATASET}.txt'

SAVE_PATH = f'data/SVM_{MODEL}.pkl'

option = int(input("- 1: Train SVM\n- 2: Evaluate SVM\n"))

if option == 1 and os.path.isfile(FEATS_FILE_TRAIN) and os.path.isfile(LABELS_FILE_TRAIN):
    print("Feature vectors and labels found")
    train_svm(FEATS_FILE_TRAIN, LABELS_FILE_TRAIN, SAVE_PATH)

elif option == 2 and os.path.isfile(FEATS_FILE_VAL) and os.path.isfile(LABELS_FILE_VAL):
    print("Feature vectors and labels found")
    val_svm(FEATS_FILE_VAL, LABELS_FILE_VAL, SAVE_PATH)

else:
    print("Feature vectors and labels not found")

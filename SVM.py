import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

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

    clf = SVC(kernel='rbf', probability=True)
    clf.fit(features, encoded_labels)

    joblib.dump({'model': clf, 'label_encoder': label_encoder}, save_path)
    print(f"SVM model saved to: {save_path}")

## ------------------------------------------------------------------------- ##

def acc_per_class(predictions, encoded_labels, label_encoder, acc_path):
    correct_per_class = {label: [0, 0] for label in label_encoder.classes_}

    for idx, pred in enumerate(predictions):
        prediction = label_encoder.classes_[pred]
        label = label_encoder.classes_[encoded_labels[idx]]

        correct_per_class[label][1] += 1
        if label == prediction:
          correct_per_class[label][0] += 1

    for key, val in correct_per_class.items():
        print(f"'{key}' Class accuracy: {(val[0]/val[1]):.4f}")

    np.save(acc_path, np.array([round((val[0]/val[1]), 4) for key, val in correct_per_class.items()]))

## ------------------------------------------------------------------------- ##

def val_svm(feats_file, labels_file, save_path, acc_path):
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
    acc_per_class(predictions, encoded_labels, label_encoder, acc_path)

    sns.heatmap(confusion_matrix(encoded_labels, predictions), annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

## ------------------------------------------------------------------------- ##

DATASET = 'VocVal'
MODEL = 'DINOv2'
DATA_DIR = 'VocPascal'

FEATS_FILE_TRAIN = f'data/feat_{MODEL}_{DATASET}_train.npy'
LABELS_FILE_TRAIN = f'data/labels_{MODEL}_{DATASET}_train.txt'

FEATS_FILE_VAL = f'data/feat_{MODEL}_{DATASET}_val.npy'
LABELS_FILE_VAL = f'data/labels_{MODEL}_{DATASET}_val.txt'

SAVE_PATH = f'data/SVM_{MODEL}.pkl'
ACC_PATH = f'data/ACC_SVM_{MODEL}.npy'

option = int(input("- 1: Train SVM\n- 2: Evaluate SVM\n"))

if option == 1 and os.path.isfile(FEATS_FILE_TRAIN) and os.path.isfile(LABELS_FILE_TRAIN):
    print("Feature vectors and labels found")
    train_svm(FEATS_FILE_TRAIN, LABELS_FILE_TRAIN, SAVE_PATH)

elif option == 2 and os.path.isfile(FEATS_FILE_VAL) and os.path.isfile(LABELS_FILE_VAL):
    print("Feature vectors and labels found")
    val_svm(FEATS_FILE_VAL, LABELS_FILE_VAL, SAVE_PATH, ACC_PATH)

else:
    print("Feature vectors and labels not found")

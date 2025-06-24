from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import os
import numpy as np
import joblib

## ------------------------------------------------------------------------- ##

def acc_per_class(predictions, encoded_labels, label_encoder):
    correct_per_class = {label: [0, 0] for label in label_encoder.classes_}

    for idx, pred in enumerate(predictions):
        prediction = label_encoder.classes_[pred]
        label = label_encoder.classes_[encoded_labels[idx]]

        correct_per_class[label][1] += 1
        if label == prediction:
          correct_per_class[label][0] += 1

    for key, val in correct_per_class.items():
      print(f"'{key}' Class accuracy: {(val[0]/val[1]):.4f}")

## ------------------------------------------------------------------------- ##

def train(feats_file, labels_file, save_path):
    print("Training RF...")

    features = np.load(feats_file)
    labels = []
    with open(labels_file, "r") as f:
        for line in f:
            labels.append(line.strip())

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Se supone que uno pone los params aca y grid search te cacha los que mejores funcionan
    param_grid = {
    'n_estimators': [1002],
    'max_depth': [20],
    'min_samples_split': [2]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(features, encoded_labels)

    best_model = grid_search.best_estimator_

    joblib.dump({'model': best_model, 'label_encoder': label_encoder}, save_path)
    print(f"RF model saved to: {save_path}")

## ------------------------------------------------------------------------- ##

def val(feats_file, labels_file, save_path):
    print("Evaluating RF...")

    features = np.load(feats_file)
    labels = []
    with open(labels_file, "r") as f:
        for line in f:
            labels.append(line.strip())

    saved = joblib.load(save_path)
    model = saved['model']
    label_encoder = saved['label_encoder']
    encoded_labels = label_encoder.transform(labels)

    predictions = model.predict(features)

    acc = accuracy_score(encoded_labels, predictions)
    print(f"Accuracy: {acc:.4f}")

    print("\nPer-class performance:")
    print(classification_report(encoded_labels, predictions, target_names=label_encoder.classes_))
    acc_per_class(predictions, encoded_labels, label_encoder)

## ------------------------------------------------------------------------- ##

DATASET = 'VocVal'
MODEL = 'DINOv2'
DATA_DIR = 'VocPascal'

FEATS_FILE_TRAIN = f'data/feat_{MODEL}_{DATASET}_train.npy'
LABELS_FILE_TRAIN = f'data/labels_{MODEL}_{DATASET}_train.txt'

FEATS_FILE_VAL = f'data/feat_{MODEL}_{DATASET}.npy'
LABELS_FILE_VAL = f'data/labels_{MODEL}_{DATASET}.txt'

SAVE_PATH = f'data/RF_{MODEL}.pth'

option = int(input("- 1: Train RF\n- 2: Evaluate\n"))

if option == 1 and os.path.isfile(FEATS_FILE_TRAIN) and os.path.isfile(LABELS_FILE_TRAIN):
  print("Feature vectors and labels found")
  train(FEATS_FILE_TRAIN, LABELS_FILE_TRAIN, SAVE_PATH)

elif option == 2 and os.path.isfile(FEATS_FILE_VAL) and os.path.isfile(LABELS_FILE_VAL):
  print("Feature vectors and labels found")
  val(FEATS_FILE_VAL, LABELS_FILE_VAL, SAVE_PATH)
  
else:
  print("Feature vectors and labels not found")
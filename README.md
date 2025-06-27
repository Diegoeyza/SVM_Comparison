# Encoder Performance Comparison: ResNet, DINOv2, CLIP with RF, SVM, and MLP

This repository evaluates image encoding models (ResNet34, DINOv2, CLIP) by extracting feature vectors from Pascal VOC images and classifying them using three different classifiers: **Random Forest (RF)**, **Support Vector Machine (SVM)**, and a **Multi-Layer Perceptron (MLP)**.

## Project Structure

- **Feature Extraction**: Extracts features from cropped images using one of three encoders.
- **Classifiers**:
  - MLP (PyTorch)
  - Random Forest (Scikit-learn)
  - SVM (Scikit-learn)

## Setup

### Requirements

```bash
pip install torch torchvision scikit-learn joblib numpy pillow git+https://github.com/openai/CLIP.git
```

### Dataset

- Uses a Pascal VOC-style folder: `VocPascal`
  - `JPEGImages/`: folder with all `.jpg` images.
  - `train_voc.txt` and `val_voc.txt`: text files with lines formatted as:
    ```
    filename label xmax xmin ymax ymin
    ```

## How to Use

### Step 1: Feature Extraction

Run the feature extraction script. You will be prompted to choose an encoder (ResNet34, DINOv2, or CLIP) and a dataset split (train or validation).

```python
python extract_features.py
```

This will save feature vectors to `.npy` and labels to `.txt` in the `data/` folder.

### Step 2: Train & Evaluate Models

#### MLP

```bash
python mlp.py
# Follow prompts to train or evaluate
```

#### Random Forest

```bash
python random_forest.py
# Follow prompts to train or evaluate
```

#### SVM

```bash
python svm.py
# Follow prompts to train or evaluate
```

## Output

Each classifier prints:
- Overall accuracy
- Per-class accuracy
- Classification report (precision, recall, F1)

## Notes

- The model uses bounding boxes from the `.txt` files to crop the image before feature extraction.
- All features are normalized and resized to 224x224 before encoding.
- MLP classifier is implemented in PyTorch, while SVM and RF are from scikit-learn.
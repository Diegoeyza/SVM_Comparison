import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
import clip


DATA_DIR = 'VocPascal'
DATASET = 'VocVal'

IMAGE_DIR = os.path.join(DATA_DIR, 'JPEGImages')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

opt1 = int(input("Select model:\n-1: ResNet34\n-2: DINOv2\n- 3: CLIP"))
if opt1 == 1:
    MODEL = 'ResnNet34'
    model = models.resnet34(pretrained=True).to(device)
    model.fc = torch.nn.Identity() 
    dim = 512

elif opt1 == 2:
    MODEL = 'DINOv2'
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dim = 384

elif opt1 == 3:
    MODEL = 'CLIP'
    model, preprocess_clip = clip.load("ViT-B/32", device=device)
    model = model.encode_image
    dim = 512

else:
    print("Invalid")

opt2 = int(input("1- Training data\n2- Validation data"))
if opt2 == 1:
    IM_FILE = os.path.join(DATA_DIR, 'train_voc.txt')
    FEATS_FILE = f'data/feat_{MODEL}_{DATASET}_train.npy'
    LABELS_FILE = f'data/labels_{MODEL}_{DATASET}_train.txt'

elif opt2 == 2:
    IM_FILE = os.path.join(DATA_DIR, 'val_voc.txt')
    FEATS_FILE = f'data/feat_{MODEL}_{DATASET}_val.npy'
    LABELS_FILE = f'data/labels_{MODEL}_{DATASET}_val.txt'

else:
    print("Invalid")

# ======================= Load image list ======================== #
files = []
labels = []
boxes = []

with open(IM_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 6:
            fname, label, xmax, xmin, ymax, ymin = parts
            if not fname.endswith('.jpg'):
                fname += '.jpg'
            files.append(fname)
            labels.append(label)
            boxes.append((int(xmax), int(xmin), int(ymax), int(ymin)))

# ======================= Preprocessing ======================= #
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

# ======================= Feature Extraction ======================= #
features = np.zeros((len(files), dim), dtype=np.float32)

for i, (fname, bbox) in enumerate(zip(files, boxes)):
    path = os.path.join(IMAGE_DIR, fname)
    image = Image.open(path).convert('RGB')
    xmax, xmin, ymax, ymin = bbox
    cropped = image.crop((xmin, ymin, xmax, ymax))
    input_tensor = preprocess(cropped).unsqueeze(0).to(device)

    with torch.no_grad():
        features[i, :] = model(input_tensor).cpu()[0, :]

    if i % 100 == 0:
        print(f"Processed {i}/{len(files)}")

# Save features and labels
os.makedirs("data", exist_ok=True)
np.save(FEATS_FILE, features)
with open(LABELS_FILE, "w") as f:
    f.writelines([l + "\n" for l in labels])
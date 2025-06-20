import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np

class Simple_model(nn.Module):
  def __init__(self, input_size, n_classes):
    super(Simple_model, self).__init__()

    self.flatten = nn.Flatten()
    self.input_layer = nn.Linear(input_size, 256)
    self.output_layer = nn.Linear(256, n_classes)
    self.relu = nn.ReLU()

  def forward(self, x):

    x = self.flatten(x)
    x = self.relu(self.input_layer(x))
    x = self.output_layer(x)
    return x
  
## ------------------------------------------------------------------------- ##
  
def train(feats_file, labels_file, save_path):

  epochs = 10
  batch_size = 16

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  features = np.load(feats_file)
  label_encoder = LabelEncoder()
  labels = []

  with open(labels_file, "r") as f:
      for line in f:
          labels.append(line.strip())
  
  encoded_labels = label_encoder.fit_transform(labels)

  features_tensor = torch.from_numpy(features).float()
  labels_tensor = torch.from_numpy(encoded_labels).long()

  dataset = TensorDataset(features_tensor, labels_tensor)

  train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

  model = Simple_model(features.shape[1], len(label_encoder.classes_))

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())

  model.to(device)
  model.train()

  for epoch in range(epochs):
    epoch_loss = 0.0
    TP = 0
    for i, data in enumerate(train_loader):
      inputs, labels = data

      inputs = inputs.to(device=device, dtype=torch.float32)
      labels = labels.to(device=device, dtype=torch.long)

      optimizer.zero_grad()

      outputs = model(inputs)
      #print(torch.argmax(outputs, dim=1))
      TP += (torch.argmax(outputs, dim=1) == labels).sum().item()

      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
    print(f"Trained epoch {epoch + 1} / {epochs}\nLoss = {(epoch_loss/len(train_loader)):.4f}\nAcc = {TP/features.shape[0]}")

  torch.save(model.state_dict(), save_path)

  print(f"Model saved to: {save_path}")

## ------------------------------------------------------------------------- ##

def val(feats_file, labels_file, save_path):

  batch_size = 16
  input_size = 384
  n_classes = 20

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  features = np.load(feats_file)
  label_encoder = LabelEncoder()
  labels = []

  with open(labels_file, "r") as f:
      for line in f:
          labels.append(line.strip())
  
  encoded_labels = label_encoder.fit_transform(labels)

  features_tensor = torch.from_numpy(features).float()
  labels_tensor = torch.from_numpy(encoded_labels).long()

  dataset = TensorDataset(features_tensor, labels_tensor)

  val_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

  model = Simple_model(input_size, n_classes)
  model.load_state_dict(torch.load(save_path, map_location=device))

  loss_fn = nn.CrossEntropyLoss()

  model.to(device)
  model.eval()

  with torch.no_grad():
    TP = 0
    loss = 0
    for i, data in enumerate(val_loader):
      inputs, labels = data

      inputs = inputs.to(device=device, dtype=torch.float32)
      labels = labels.to(device=device, dtype=torch.long)

      outputs = model(inputs)
      TP += (torch.argmax(outputs, dim=1) == labels).sum().item()
      loss += loss_fn(outputs, labels).item()
      
    print(f"Loss = {(loss/len(val_loader)):.4f}\nAcc = {TP/features.shape[0]}")

## ------------------------------------------------------------------------- ##

DATASET = 'VocVal'
MODEL = 'dinov2'
DATA_DIR = 'VocPascal'

FEATS_FILE_TRAIN = f'data/feat_{MODEL}_{DATASET}.npy'
LABELS_FILE_TRAIN = f'data/labels_{MODEL}_{DATASET}.txt'

FEATS_FILE_VAL = f'data/feat_{MODEL}_{DATASET}.npy'
LABELS_FILE_VAL = f'data/labels_{MODEL}_{DATASET}.txt'

SAVE_PATH = f'data/MLP_{MODEL}.pth'

option = int(input("- 1: Train MLP\n- 2:Evaluate\n"))

if option == 1 and os.path.isfile(FEATS_FILE_TRAIN) and os.path.isfile(LABELS_FILE_TRAIN):
  print("Feature vectors and labels found")
  train(FEATS_FILE_TRAIN, LABELS_FILE_TRAIN, SAVE_PATH)

elif option == 2 and os.path.isfile(FEATS_FILE_VAL) and os.path.isfile(LABELS_FILE_VAL):
  print("Feature vectors and labels found")
  val(FEATS_FILE_VAL, LABELS_FILE_VAL, SAVE_PATH)
  
else:
  print("Feature vectors and labels not found")
import matplotlib.pyplot as plt
import numpy as np

CLASSIFICATOR = "RF"

ResNet_acc = np.load(f'data/ACC_{CLASSIFICATOR}_ResNet34.npy')
DINO_acc = np.load(f'data/ACC_{CLASSIFICATOR}_DINOv2.npy')
CLIP_acc = np.load(f'data/ACC_{CLASSIFICATOR}_CLIP.npy')

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#print(ResNet_acc)
#print(DINO_acc)
#print(CLIP_acc)

x = np.arange(len(classes))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Each bar set offset by width
bars1 = ax.bar(x - width, CLIP_acc, width, label='CLIP', color='skyblue')
bars2 = ax.bar(x, DINO_acc, width, label='DINOv2', color='salmon')
bars3 = ax.bar(x + width, ResNet_acc, width, label='ResNet', color='seagreen')

# Labels and legend
ax.set_xlabel('Classes')
ax.set_ylabel('Accuracy')
ax.set_title(f'Accuracy per Class ({CLASSIFICATOR})')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"plots/acc_bar_chart_{CLASSIFICATOR}.png", dpi=300, bbox_inches='tight')
plt.show()
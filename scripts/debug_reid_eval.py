"""Debug Re-ID evaluation."""

import sys
from pathlib import Path

import numpy as np
import torch

from src.data.market_dataset import Market1501Dataset
from src.data.transforms import get_val_transforms
from src.reid.resnet_reid import ResNet50ReID

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load model
device = torch.device("cuda")
checkpoint = torch.load("models/reid/final/resnet50_market_best.pth", map_location=device)
# Get actual num_classes from checkpoint state dict
classifier_weight_shape = checkpoint["model_state_dict"]["classifier.weight"].shape
num_classes = classifier_weight_shape[0]
print(f"Model has {num_classes} classes")

model = ResNet50ReID(num_classes=num_classes, pretrained=False, feature_dim=512)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Load datasets
query_ds = Market1501Dataset(Path("data/processed/market1501"), "query", get_val_transforms())
gallery_ds = Market1501Dataset(Path("data/processed/market1501"), "gallery", get_val_transforms())

# Pick first query
query_img, _, query_cam, query_pid = query_ds[0]
print(f"Query: Person ID {query_pid}, Camera {query_cam}")

# Extract query feature
with torch.no_grad():
    query_feat = model.extract_features(query_img.unsqueeze(0).to(device))
    query_feat = query_feat.cpu().numpy().flatten()

# Extract first 100 gallery features
gallery_feats = []
gallery_pids = []
gallery_cams = []

for i in range(min(100, len(gallery_ds))):
    img, _, cam, pid = gallery_ds[i]
    with torch.no_grad():
        feat = model.extract_features(img.unsqueeze(0).to(device))
        feat = feat.cpu().numpy().flatten()

    gallery_feats.append(feat)
    gallery_pids.append(pid)
    gallery_cams.append(cam)

    if pid == query_pid:
        print(f"  Gallery {i}: Person ID {pid}, Camera {cam} - MATCH!")

gallery_feats = np.array(gallery_feats)

# Compute distances (cosine)
similarities = np.dot(gallery_feats, query_feat)
distances = 1 - similarities

# Sort
indices = np.argsort(distances)

print("\nTop 10 matches:")
for rank, idx in enumerate(indices[:10]):
    pid = gallery_pids[idx]
    cam = gallery_cams[idx]
    dist = distances[idx]
    match = "✓ MATCH" if pid == query_pid else ""
    print(f"  Rank {rank+1}: Person {pid}, Cam {cam}, Dist {dist:.4f} {match}")

# Check if any matches in top 10
matches_in_top10 = sum(1 for idx in indices[:10] if gallery_pids[idx] == query_pid)
print(f"\nMatches in top 10: {matches_in_top10}")

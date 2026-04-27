"""
STEP 2 : Feature Extraction with ResNet50
==========================================
Loads every image from the dataset, passes it through a pre-trained ResNet50
(weights = ImageNet, final classification head removed), and saves the resulting
2048-dimensional embedding vectors for every image.

We then apply PCA to reduce to 100 dimensions before clustering.
"""

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Image transform (ResNet50 standard preprocessing) 
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  
        std =[0.229, 0.224, 0.225]    
    ),
])


# Build feature extractor 

def build_extractor(model_name: str = "resnet50"):
    """Load a pretrained model and strip the classification head."""
    if model_name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final FC layer → output is (batch, 2048) after global avg pool
        extractor = torch.nn.Sequential(*list(base.children())[:-1])
    elif model_name == "vgg16":
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Use only features + adaptive pool, strip classifier
        extractor = torch.nn.Sequential(base.features, base.avgpool)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    extractor = extractor.to(DEVICE)
    extractor.eval()
    return extractor


# Collect file paths

def collect_paths(root: str, categories: list, max_per_class: int = None):
    filepaths = []
    for cat in categories:
        cat_dir = os.path.join(root, cat)
        if not os.path.isdir(cat_dir):
            print(f"  [WARNING] Missing folder: {cat_dir}")
            continue
        files = (glob.glob(os.path.join(cat_dir, "*.jpg"))  +
                 glob.glob(os.path.join(cat_dir, "*.jpeg")) +
                 glob.glob(os.path.join(cat_dir, "*.png")))
        if max_per_class:
            files = files[:max_per_class]
        filepaths.extend(files)
        print(f"  {cat:12s}: {len(files)} images")
    return filepaths


# Extract features

def extract_features(extractor, filepaths: list, batch_size: int = 32):
    all_features = []

    for i in tqdm(range(0, len(filepaths), batch_size), desc="Extracting features"):
        batch_paths = filepaths[i : i + batch_size]
        tensors     = []

        for fp in batch_paths:
            try:
                img = Image.open(fp).convert("RGB")
                tensors.append(transform(img))
            except Exception as e:
                print(f"  [SKIP] {fp}: {e}")
                tensors.append(torch.zeros(3, 224, 224)) 

        batch = torch.stack(tensors).to(DEVICE)

        with torch.no_grad():
            feats = extractor(batch)

        # ResNet50 output shape is (batch, 2048, 1, 1) → flatten to (batch, 2048)
        feats = feats.view(feats.size(0), -1).cpu().numpy()
        all_features.append(feats)

    return np.vstack(all_features)


# Main body

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n========== STEP 2: Feature Extraction ==========\n")
    print(f"Model       : {config.MODEL_NAME}")
    print(f"PCA dims    : {config.PCA_COMPONENTS}")
    print(f"Dataset root: {config.TRAIN_SPLIT}\n")

    # 1. Collect image paths 
    print("Collecting file paths …")
    filepaths = collect_paths(
        config.TRAIN_SPLIT,
        config.CATEGORIES,
        config.MAX_IMAGES_PER_CLASS
    )
    print(f"\nTotal images: {len(filepaths)}\n")

    # 2. Build extractor 
    print("Loading pretrained ResNet50 …")
    extractor = build_extractor(config.MODEL_NAME)

    # 3. Extract raw features 
    print("\nRunning inference …")
    features_raw = extract_features(extractor, filepaths, batch_size=32)
    print(f"\nRaw feature shape  : {features_raw.shape}")  # (N, 2048)

    # 4. Standardise 
    print("Standardising features …")
    scaler   = StandardScaler()
    features_scaled = scaler.fit_transform(features_raw)

    # 5. PCA 
    n_components = min(config.PCA_COMPONENTS, features_scaled.shape[0], features_scaled.shape[1])
    print(f"Applying PCA: {features_scaled.shape[1]} → {n_components} dims …")
    pca      = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
    features = pca.fit_transform(features_scaled)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA explained variance: {explained:.1f}%")
    print(f"Final feature shape   : {features.shape}")
    from sklearn.preprocessing import normalize
    features = normalize(features, norm='l2')

    # 6. Save 
    np.save(config.FEATURES_FILE,  features)
    np.save(config.FILEPATHS_FILE, np.array(filepaths))
    print(f"\n[Saved] {config.FEATURES_FILE}")
    print(f"[Saved] {config.FILEPATHS_FILE}")

    print("\n========== Step 2 Complete ==========\n")


if __name__ == "__main__":
    main()

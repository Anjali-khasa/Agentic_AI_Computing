import os

# ─── Paths ────────────────────────────────────────────────────────────────────
# Folder structure confirmed:  dataset\seg_train\buildings\  etc.
DATASET_PATH   = "dataset"
OUTPUT_DIR     = "outputs"
PLOTS_DIR      = os.path.join(OUTPUT_DIR, "plots")
VIDEO_DIR      = os.path.join(OUTPUT_DIR, "videos")
FEATURES_FILE  = os.path.join(OUTPUT_DIR, "features.npy")
FILEPATHS_FILE = os.path.join(OUTPUT_DIR, "filepaths.npy")
CLUSTERS_FILE  = os.path.join(OUTPUT_DIR, "cluster_assignments.npy")

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Intel Image Classification categories
CATEGORIES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Use training split only for clustering
TRAIN_SPLIT = os.path.join(DATASET_PATH, "seg_test")

# Max images per class to load (set None to use all)
MAX_IMAGES_PER_CLASS = 200  # 200 x 6 = 1200 total images

# ─── Image preprocessing ──────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)   # ResNet50 expected input size

# ─── Feature extraction ───────────────────────────────────────────────────────
# Options: 'resnet50'  (2048-dim), 'vgg16' (4096-dim), 'efficientnetb0' (1280-dim)
MODEL_NAME   = "resnet50"
PCA_COMPONENTS = 150     # Reduce before clustering (speeds up + reduces noise)

# ─── Clustering ───────────────────────────────────────────────────────────────
N_CLUSTERS   = 6          # Start with 6 (matches the 6 scene categories)
RANDOM_SEED  = 42
K_RANGE      = range(2, 12)  # Range to test for elbow method

# ─── Video ────────────────────────────────────────────────────────────────────
SECONDS_PER_IMAGE = 3     # How long each image is shown in the slideshow
VIDEO_FPS         = 24    # Output video FPS
VIDEO_SIZE        = (640, 480)

# ─── Music ────────────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 44100  # Hz

# SENG 691 – Assignment 5: Image Clustering & Video Generation

An end-to-end deep learning pipeline that clusters natural scene images using ResNet50 feature extraction and K-Means, then generates slideshow videos with algorithmically composed background music — all driven by the visual properties of each cluster.

**Dataset:** Intel Image Classification (6 categories: buildings, forest, glacier, mountain, sea, street)

---

## Project Structure

```
Assignment 5/
├── dataset/
│   └── seg_test/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
├── outputs/
│   ├── plots/              # All generated visualisation plots
│   ├── videos/             # Final slideshow .mp4 files
│   ├── features.npy        # Extracted ResNet50 + PCA features
│   ├── filepaths.npy       # Image file path index
│   └── cluster_assignments.npy
├── config.py               # Central configuration (paths, params)
├── step1_analysis.py       # Dataset exploration & visual statistics
├── step2_feature_extraction.py  # ResNet50 + PCA + L2 normalisation
├── step3_clustering.py     # K-Means, Silhouette, t-SNE, Dendrogram
├── step4_video_music.py    # Algorithmic music + slideshow video
├── main.py                 # Single entry point to run all steps
└── requirements.txt
```

---

## Installation

### 1. Clone / Download the project

Place the project folder on your machine and open a terminal inside it.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is included in `requirements.txt`. If you have a CUDA-capable GPU, visit [pytorch.org](https://pytorch.org) to install the GPU build for faster feature extraction. The pipeline works on CPU by default.

---

## Dataset Setup

1. Download the **Intel Image Classification** dataset from Kaggle:
   [https://www.kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

2. Extract and place the `seg_test` folder inside a `dataset/` directory:

```
Assignment 5/
└── dataset/
    └── seg_test/
        ├── buildings/   (~437 images)
        ├── forest/      (~474 images)
        ├── glacier/     (~553 images)
        ├── mountain/    (~525 images)
        ├── sea/         (~500 images)
        └── street/      (~501 images)
```

> The pipeline uses `seg_test` (not `seg_train`) as it contains all 6 categories with sufficient images.

---

## Running the Pipeline

### Run everything at once

```bash
python main.py
```

### Run individual steps

```bash
python main.py --steps 1          # Dataset analysis only
python main.py --steps 1 2        # Analysis + feature extraction
python main.py --steps 3          # Clustering (requires step 2 done)
python main.py --steps 4          # Video + music (requires steps 2 & 3 done)
```

### Run steps individually

```bash
python step1_analysis.py
python step2_feature_extraction.py
python step3_clustering.py
python step4_video_music.py
```

---

## What Each Step Does

| Step | Script | Output |
|------|--------|--------|
| 1 | `step1_analysis.py` | Distribution chart, sample grid, brightness/warmth stats |
| 2 | `step2_feature_extraction.py` | ResNet50 (2048-dim) → PCA (150-dim, 86.7% variance) → L2 normalised features |
| 3 | `step3_clustering.py` | Elbow curve, K-Means (K=6), Silhouette plot, t-SNE, Dendrogram, cluster sample grid |
| 4 | `step4_video_music.py` | Algorithmically generated `.wav` audio + `.mp4` slideshow per cluster |

---

## Configuration

All key parameters are set in `config.py`:

```python
MAX_IMAGES_PER_CLASS = 200   # Images loaded per category
N_CLUSTERS = 6               # Number of K-Means clusters
PCA_COMPONENTS = 150         # PCA dimensionality reduction target
SECONDS_PER_IMAGE = 3        # Slideshow display time per image
MODEL_NAME = "resnet50"      # Feature extractor backbone
```

---

## Expected Outputs

After running all steps, the `outputs/` folder will contain:

```
outputs/
├── plots/
│   ├── distribution.png       # Image count per category
│   ├── sample_grid.png        # 3x6 sample image grid
│   ├── visual_stats.png       # Brightness & warmth per category
│   ├── elbow.png              # Inertia vs K elbow curve
│   ├── silhouette.png         # Silhouette coefficient per cluster
│   ├── tsne.png               # t-SNE 2D cluster visualisation
│   ├── dendrogram.png         # Hierarchical clustering dendrogram
│   └── cluster_samples.png    # Sample images per cluster
├── videos/
│   ├── cluster_0_slideshow.mp4
│   ├── cluster_1_slideshow.mp4
│   └── cluster_2_slideshow.mp4
├── cluster_0_audio.wav
├── cluster_1_audio.wav
├── cluster_2_audio.wav
├── features.npy
├── filepaths.npy
└── cluster_assignments.npy
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `torch`, `torchvision` | ResNet50 feature extraction |
| `scikit-learn` | PCA, K-Means, Silhouette, t-SNE |
| `scipy` | Hierarchical clustering, WAV file writing |
| `moviepy` | Slideshow video + audio merging |
| `Pillow` | Image loading and resizing |
| `matplotlib` | All plots and visualisations |
| `numpy` | Feature processing and audio synthesis |
| `tqdm` | Progress bars |

See `requirements.txt` for exact versions.

---

## Troubleshooting

**Missing category folders (only 3 of 6 load)**
→ Make sure you are pointing to `seg_test`, not `seg_train`. Check `TRAIN_SPLIT` in `config.py`.

**TSNE TypeError: unexpected keyword argument 'n_iter'**
→ You have scikit-learn ≥ 1.4. The script already uses `max_iter` — ensure you have the latest version of `step3_clustering.py`.

**MoviePy import error (`No module named 'moviepy.editor'`)**
→ You have MoviePy 2.x. The script already uses the updated `from moviepy import ...` syntax.

**Out of memory during feature extraction**
→ Reduce `MAX_IMAGES_PER_CLASS` in `config.py` (e.g., set to 100).

**Slow feature extraction**
→ Normal on CPU — 1,200 images takes approximately 3–5 minutes. GPU will be used automatically if available (CUDA).

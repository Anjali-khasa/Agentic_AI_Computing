"""
STEP 3 : Clustering & Visualisation
=====================================
Loads the PCA-reduced features from Step 2 and applies:
  1. Elbow method  → find optimal K
  2. K-Means       → primary clustering method
  3. Silhouette    → evaluate cluster quality
  4. t-SNE         → 2-D visualisation of clusters
  5. Agglomerative (hierarchical) clustering → comparison method
  6. Cluster summary grid → sample images per cluster

"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage

import config

# Helper functions

CATEGORY_COLORS = {
    'buildings': '#4C72B0',
    'forest'   : '#55A868',
    'glacier'  : '#64B5CD',
    'mountain' : '#8172B2',
    'sea'      : '#CCB974',
    'street'   : '#C44E52',
}

CLUSTER_PALETTE = [
    '#4C72B0', '#55A868', '#C44E52', '#8172B2',
    '#CCB974', '#64B5CD', '#E07B39', '#76C5AF',
    '#D4A0C1', '#A0C16E'
]


def infer_label_from_path(filepath: str, categories: list) -> str:
    """Guess ground-truth label from folder name."""
    for cat in categories:
        if os.sep + cat + os.sep in filepath or filepath.endswith(os.sep + cat):
            return cat
    return "unknown"


# 1. Elbow Method 

def elbow_method(features: np.ndarray, k_range, seed: int):
    print("\n[1/5] Elbow Method …")
    inertias = []
    for k in tqdm(k_range, desc="  K-Means sweep"):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(features)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker='o', color='#4C72B0', linewidth=2)
    ax.set_title("Elbow Method – Inertia vs. Number of Clusters", fontsize=12)
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, "elbow.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  [Saved] {out}")
    return inertias


# 2. K-Means 

def run_kmeans(features: np.ndarray, n_clusters: int, seed: int):
    print(f"\n[2/5] K-Means with K={n_clusters} …")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=15, max_iter=500)
    labels = km.fit_predict(features)
    print(f"  Cluster sizes: { {i: (labels == i).sum() for i in range(n_clusters)} }")
    return km, labels


# 3. Silhouette Analysis 

def silhouette_analysis(features: np.ndarray, labels: np.ndarray, n_clusters: int):
    print("\n[3/5] Silhouette Analysis …")
    score  = silhouette_score(features, labels)
    values = silhouette_samples(features, labels)
    print(f"  Overall Silhouette Score: {score:.4f}  (range -1 to 1, higher is better)")

    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10
    for i in range(n_clusters):
        cluster_vals = np.sort(values[labels == i])
        size_i       = cluster_vals.shape[0]
        y_upper      = y_lower + size_i
        color        = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_i, f"C{i}", fontsize=8)
        y_lower = y_upper + 10

    ax.axvline(score, color="red", linestyle="--", linewidth=1.5, label=f"Avg = {score:.3f}")
    ax.set_title("Silhouette Plot per Cluster", fontsize=12)
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, "silhouette.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  [Saved] {out}")
    return score


# 4. t-SNE Visualisation 

def tsne_plot(features: np.ndarray, labels: np.ndarray,
              true_labels: list, n_clusters: int, seed: int):
    print("\n[4/5] t-SNE visualisation (this takes a moment) …")
    tsne   = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(features)

    # Plot A – coloured by K-Means cluster
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i in range(n_clusters):
        mask = labels == i
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        c=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)],
                        s=8, alpha=0.7, label=f"Cluster {i}")
    axes[0].set_title("t-SNE – Coloured by K-Means Cluster", fontsize=11)
    axes[0].legend(markerscale=2, fontsize=8)
    axes[0].axis("off")

    # Plot B – coloured by ground-truth category
    unique_cats = list(CATEGORY_COLORS.keys())
    for cat in unique_cats:
        mask = np.array(true_labels) == cat
        if mask.sum() == 0:
            continue
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        c=CATEGORY_COLORS[cat], s=8, alpha=0.7, label=cat)
    axes[1].set_title("t-SNE – Coloured by True Category", fontsize=11)
    axes[1].legend(markerscale=2, fontsize=8)
    axes[1].axis("off")

    plt.suptitle("t-SNE Embedding of ResNet50 Features", fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, "tsne.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {out}")
    return coords


# 5. Hierarchical Clustering Dendrogram

def hierarchical_dendrogram(features: np.ndarray, n_samples: int = 200, seed: int = 42):
    print("\n[5/5] Hierarchical Clustering (dendrogram on sample) …")
    rng     = np.random.default_rng(seed)
    idx     = rng.choice(len(features), size=min(n_samples, len(features)), replace=False)
    subset  = features[idx]

    Z  = linkage(subset, method="ward")
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=6,
               color_threshold=0.7 * max(Z[:, 2]),
               above_threshold_color="gray")
    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward linkage, {n_samples} sample)", fontsize=12)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, "dendrogram.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  [Saved] {out}")


# 6. Cluster Sample Grid 

def cluster_sample_grid(filepaths: list, labels: np.ndarray, n_clusters: int,
                        samples_per_cluster: int = 8):
    print("\nGenerating cluster sample grid …")
    rng = np.random.default_rng(config.RANDOM_SEED)
    fig = plt.figure(figsize=(samples_per_cluster * 2, n_clusters * 2.2))
    gs  = gridspec.GridSpec(n_clusters, samples_per_cluster, figure=fig,
                            hspace=0.5, wspace=0.05)

    for c in range(n_clusters):
        idxs  = np.where(labels == c)[0]
        picks = rng.choice(idxs, size=min(samples_per_cluster, len(idxs)), replace=False)
        for col, idx in enumerate(picks):
            ax  = fig.add_subplot(gs[c, col])
            img = Image.open(filepaths[idx]).resize((120, 90))
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"Cluster {c}", fontsize=9, rotation=90, labelpad=4)

    fig.suptitle("Sample Images per Cluster", fontsize=13, y=1.01)
    out = os.path.join(config.PLOTS_DIR, "cluster_samples.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {out}")


# 7. Cluster Purity Analysis 

def cluster_purity_report(filepaths: list, labels: np.ndarray, n_clusters: int):
    """Show which true categories fall in each cluster."""
    print("\n── Cluster Purity Report ──────────────────────────────────")
    print(f"{'Cluster':>8}  {'Size':>6}  {'Dominant Category':>20}  {'Composition'}")
    print("-" * 80)
    for c in range(n_clusters):
        idxs       = np.where(labels == c)[0]
        cat_counts = {}
        for i in idxs:
            cat = infer_label_from_path(filepaths[i], config.CATEGORIES)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        dominant   = max(cat_counts, key=cat_counts.get)
        composition = ", ".join(f"{k}:{v}" for k, v in sorted(
            cat_counts.items(), key=lambda x: -x[1]))
        print(f"  C{c:>4}    {len(idxs):>6}  {dominant:>20}  {composition}")
    print()


# Main body

def main():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print("\n========== STEP 3: Clustering ==========\n")

    # Load features & file paths
    if not os.path.exists(config.FEATURES_FILE):
        raise FileNotFoundError(
            f"Features not found at {config.FEATURES_FILE}. Run step2 first.")

    features  = np.load(config.FEATURES_FILE)
    filepaths = list(np.load(config.FILEPATHS_FILE, allow_pickle=True))
    print(f"Loaded features : {features.shape}")
    print(f"Loaded paths    : {len(filepaths)}")

    true_labels = [infer_label_from_path(fp, config.CATEGORIES) for fp in filepaths]

    # 1. Elbow 
    elbow_method(features, config.K_RANGE, config.RANDOM_SEED)

    # 2. K-Means 
    km, labels = run_kmeans(features, config.N_CLUSTERS, config.RANDOM_SEED)

    # 3. Silhouette 
    silhouette_analysis(features, labels, config.N_CLUSTERS)

    # 4. t-SNE 
    tsne_plot(features, labels, true_labels, config.N_CLUSTERS, config.RANDOM_SEED)

    # 5. Hierarchical dendrogram 
    hierarchical_dendrogram(features, n_samples=300, seed=config.RANDOM_SEED)

    # 6. Sample grid
    cluster_sample_grid(filepaths, labels, config.N_CLUSTERS)

    # 7. Purity report 
    cluster_purity_report(filepaths, labels, config.N_CLUSTERS)

    # 8. Save cluster assignment 
    np.save(config.CLUSTERS_FILE, labels)
    print(f"[Saved] {config.CLUSTERS_FILE}")

    print("\n========== Step 3 Complete ==========\n")
    print(f"  Cluster assignments saved → use in step4_video_music.py\n")


if __name__ == "__main__":
    main()

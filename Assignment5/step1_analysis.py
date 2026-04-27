"""
STEP 1 -  Analysis

Loading the Intel Image Classification dataset, counting images per category,
plots the distribution, shows a sample grid, and computes basic visual
statistics (brightness, colour temperature).

"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from collections import defaultdict

import config

#Helper functions

def load_image_paths(root: str, categories: list, max_per_class: int = None) -> dict:
    """Return {category: [file_path, ...]} from a root folder."""
    paths = defaultdict(list)
    for cat in categories:
        cat_dir = os.path.join(root, cat)
        if not os.path.isdir(cat_dir):
            print(f"  [WARNING] Category folder not found: {cat_dir}")
            continue
        files = glob.glob(os.path.join(cat_dir, "*.jpg")) + \
                glob.glob(os.path.join(cat_dir, "*.jpeg")) + \
                glob.glob(os.path.join(cat_dir, "*.png"))
        if max_per_class:
            files = files[:max_per_class]
        paths[cat] = files
        print(f"  {cat:12s}: {len(files):>4d} images")
    return paths


def average_brightness(img_path: str) -> float:
    """Return mean pixel brightness (0-255) of a grayscale image."""
    img = Image.open(img_path).convert("L").resize((64, 64))
    return np.array(img, dtype=np.float32).mean()


def colour_temperature_score(img_path: str) -> float:
    """
    Simple warm/cool heuristic.
    Positive  → warm (reds/yellows dominate) – e.g. desert, buildings at sunset.
    Negative  → cool (blues dominate)         – e.g. glaciers, sea.
    """
    img = np.array(Image.open(img_path).convert("RGB").resize((64, 64)), dtype=np.float32)
    r, g, b = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    return (r - b)          # warm − cool balance

# Main execution

def main():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n========== STEP 1: Dataset Analysis ==========\n")
    print(f"Dataset root : {config.TRAIN_SPLIT}")
    print(f"Max per class: {config.MAX_IMAGES_PER_CLASS}\n")

    # 1. Load paths 
    paths = load_image_paths(
        config.TRAIN_SPLIT,
        config.CATEGORIES,
        config.MAX_IMAGES_PER_CLASS
    )

    total = sum(len(v) for v in paths.values())
    print(f"\nTotal images loaded : {total}")

    # 2. Bar chart – image distribution 
    fig, ax = plt.subplots(figsize=(8, 4))
    counts  = [len(paths[c]) for c in config.CATEGORIES]
    colors  = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
    bars    = ax.bar(config.CATEGORIES, counts, color=colors, width=0.6, edgecolor='white')
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Image Count per Category – Intel Scene Dataset", fontsize=13, pad=12)
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(counts) * 1.18)
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, "distribution.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"\n[Saved] {out}")

    # 3. image grid 
    n_cols   = 6
    n_rows   = 3
    fig      = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5))
    gs       = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.1)

    rng = np.random.default_rng(config.RANDOM_SEED)
    for col, cat in enumerate(config.CATEGORIES):
        samples = rng.choice(paths[cat], size=min(n_rows, len(paths[cat])), replace=False)
        for row, fpath in enumerate(samples):
            ax = fig.add_subplot(gs[row, col])
            img = Image.open(fpath).resize((120, 90))
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(cat, fontsize=9, pad=4)

    fig.suptitle("Sample Images – 3 per Category", fontsize=13, y=1.02)
    out = os.path.join(config.PLOTS_DIR, "sample_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {out}")

    # 4. Visual statistics per category 
    print("\nComputing visual statistics (brightness & colour temperature) …")
    stats = {}
    for cat in config.CATEGORIES:
        brightness = []
        warmth     = []
        sample     = paths[cat][:50]        
        for fp in sample:
            try:
                brightness.append(average_brightness(fp))
                warmth.append(colour_temperature_score(fp))
            except Exception:
                pass
        stats[cat] = {
            "avg_brightness"   : np.mean(brightness),
            "avg_warmth"       : np.mean(warmth),
            "std_brightness"   : np.std(brightness),
        }

    print(f"\n{'Category':12s}  {'Avg Brightness':>16s}  {'Warmth Score':>14s}")
    print("-" * 48)
    for cat, s in stats.items():
        print(f"{cat:12s}  {s['avg_brightness']:>16.1f}  {s['avg_warmth']:>14.1f}")

    # 5. Brightness & warmth comparison chart
    cats  = list(stats.keys())
    bvals = [stats[c]["avg_brightness"] for c in cats]
    wvals = [stats[c]["avg_warmth"]     for c in cats]

    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, bvals, width=w, label="Avg Brightness", color="#4C72B0", edgecolor="white")
    ax.bar(x + w/2, wvals, width=w, label="Warmth Score",   color="#C44E52", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_title("Visual Statistics per Category", fontsize=12)
    ax.set_ylabel("Score")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, "visual_stats.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"[Saved] {out}")

    # 6. Saving stats
    np.save(os.path.join(config.OUTPUT_DIR, "visual_stats.npy"), stats)

    print("\n========== Step 1 Complete ==========\n")
    print(f"  Total images : {total}")
    print(f"  Categories   : {config.CATEGORIES}")
    print(f"  Plots saved to: {config.PLOTS_DIR}/\n")


if __name__ == "__main__":
    main()

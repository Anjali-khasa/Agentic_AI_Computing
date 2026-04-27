"""
main.py – Run all steps in sequence
=====================================
This is the single entry point that executes the full pipeline:
  Step 1  → Dataset analysis & visualisation
  Step 2  → Feature extraction (ResNet50 + PCA)
  Step 3  → Clustering (K-Means, t-SNE, Silhouette, Dendrogram)
  Step 4  → Slideshow video + algorithmic music generation

Usage:
    python main.py                  # run all steps
    python main.py --steps 1 2      # run only steps 1 and 2
    python main.py --steps 3 4      # run only steps 3 and 4 (after 1&2 done)
"""

import argparse
import time

import step1_analysis
import step2_feature_extraction
import step3_clustering
import step4_video_music


def banner(msg: str):
    line = "═" * 60
    print(f"\n{line}")
    print(f"  {msg}")
    print(f"{line}\n")


def main():
    parser = argparse.ArgumentParser(description="SENG 691 HW5 – Image Clustering Pipeline")
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[1, 2, 3, 4],
        help="Which steps to run (default: all – 1 2 3 4)"
    )
    args = parser.parse_args()

    t_start = time.time()

    if 1 in args.steps:
        banner("STEP 1 – Dataset Analysis")
        step1_analysis.main()

    if 2 in args.steps:
        banner("STEP 2 – Feature Extraction (ResNet50)")
        step2_feature_extraction.main()

    if 3 in args.steps:
        banner("STEP 3 – Clustering & Visualisation")
        step3_clustering.main()

    if 4 in args.steps:
        banner("STEP 4 – Video + Algorithmic Music")
        step4_video_music.main()

    elapsed = time.time() - t_start
    print(f"\n✓ Pipeline finished in {elapsed/60:.1f} minutes")
    print("  Check the  outputs/  folder for all results.\n")


if __name__ == "__main__":
    main()

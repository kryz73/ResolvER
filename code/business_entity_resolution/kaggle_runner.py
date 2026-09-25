"""
ML Challenge 2026 — Business Entity Resolution
Standalone Kaggle Execution Runner
Auto-configures Kaggle environment, installs dependencies, executes Approach 4 pipeline,
and packages submission files into /kaggle/working/submission.zip.
"""

import os
import sys
import subprocess
import shutil
import zipfile
from pathlib import Path
import pandas as pd
import json


def run_cmd(cmd: str):
    print(f"\n[Kaggle Runner] Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[Kaggle Runner Warning] Command exited with code: {res.returncode}")
    return res.returncode


def main():
    print("=" * 80)
    print("2026 AMAZON BUSINESS ENTITY RESOLUTION — KAGGLE PIPELINE RUNNER")
    print("=" * 80)

    # 1. Install necessary dependencies if missing
    try:
        import rapidfuzz
        import indic_transliteration
        import lightgbm
    except ImportError:
        print("\n[Step 1/5] Installing core dependencies (rapidfuzz, indic-transliteration)...")
        run_cmd(f"{sys.executable} -m pip install -q rapidfuzz indic-transliteration lightgbm")

    # 2. Setup paths
    src_dir = Path(__file__).parent / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir.resolve()))
        os.chdir(str(src_dir.resolve()))
    else:
        sys.path.insert(0, ".")

    from config import (
        DATASET_ROOT, TRAIN_S1_PATH, TEST_S1_PATH,
        OUTPUT_DIR, OUTPUT_MATCHING_PATH, OUTPUT_CANDIDATE_PATH,
        VALIDATE_SCRIPT, TEST_DIR
    )
    from pipeline import run_pipeline, run_loco_evaluation

    print(f"\n[Step 2/5] Verified Dataset Paths:")
    print(f"  Dataset Root: {DATASET_ROOT}")
    print(f"  Train S1:     {TRAIN_S1_PATH} (Exists: {TRAIN_S1_PATH.exists()})")
    print(f"  Test S1:      {TEST_S1_PATH} (Exists: {TEST_S1_PATH.exists()})")
    print(f"  Output Dir:   {OUTPUT_DIR}")

    import argparse
    parser = argparse.ArgumentParser(description="Kaggle Runner for Business Entity Resolution")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "loco_check"],
                        help="Execution mode: 'standard' or 'loco_check' (Leave-One-Country-Out France proxy)")
    parser.add_argument("--sample-size", type=int, default=None, help="Train sample size (default: None for full Kaggle dataset)")
    parser.add_argument("--use-dense", action="store_true", help="Enable dense multilingual embedding blocking")
    parser.add_argument("--skip-test", action="store_true", help="Skip test set inference (useful for fast cross-validation diagnostics)")
    args = parser.parse_args()

    # Check for GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_gpu else "None (CPU Mode)"
    except ImportError:
        has_gpu = False
        gpu_name = "None (PyTorch not found)"

    device = "cuda" if has_gpu else "cpu"
    print(f"  Compute Device: {device} ({gpu_name})")

    log_dir = "/kaggle/working/logs" if Path("/kaggle/working").exists() else "logs"

    if args.mode == "loco_check":
        print(f"\n[Step 3/5] Launching Leave-One-Country-Out (LOCO) Proxy Evaluation on {device}...")
        run_loco_evaluation(
            sample_size=args.sample_size,
            device=device,
            log_dir=log_dir,
            experiment_name="kaggle_run"
        )
    else:
        sample_desc = f"{args.sample_size:,}" if args.sample_size else "FULL DATASET"
        test_desc = "SKIPPED (Evaluation Mode Only)" if args.skip_test else "ENABLED (Generating Submission)"
        print(f"\n[Step 3/5] Launching Approach 4 Pipeline on {device} (Train Sample: {sample_desc}, Test Inference: {test_desc})...")

        run_pipeline(
            train_sample_size=args.sample_size,  # Full dataset or custom sample
            test_mode=not args.skip_test,        # Generate test candidate_pairs.tsv and matching_results.tsv
            use_dense=args.use_dense,            # Dense FAISS vector blocking flag
            device=device,
            log_dir=log_dir,
            experiment_name="kaggle_run"
        )


    # 4. Display Diagnostic Results in Kaggle
    print("\n[Step 4/5] Pipeline Execution Summary & Diagnostics:")
    log_base = Path("/kaggle/working/logs/kaggle_run") if Path("/kaggle/working").exists() else Path("logs/kaggle_run")

    summary_file = log_base / "run_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        print("\n--- RUN SUMMARY ---")
        print(json.dumps(summary, indent=2))

    feat_file = log_base / "feature_importance.csv"
    if feat_file.exists():
        f_df = pd.read_csv(feat_file)
        print("\n--- TOP 15 FEATURE IMPORTANCES ---")
        print(f_df.head(15).to_string(index=False))

    fp_file = log_base / "error_analysis_false_positives.csv"
    if fp_file.exists():
        fp_df = pd.read_csv(fp_file)
        print(f"\n--- TOP FALSE POSITIVES (Total: {len(fp_df)}) ---")
        print(fp_df[['s1_id', 'target_id', 'prob', 's1_name', 'target_name']].head(5).to_string(index=False))

    # 5. Packaging Submission Zip
    if not args.skip_test and (OUTPUT_MATCHING_PATH.exists() or OUTPUT_CANDIDATE_PATH.exists()):
        print("\n[Step 5/5] Packaging Submission Zip...")
        working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else OUTPUT_DIR
        working_dir.mkdir(parents=True, exist_ok=True)
        zip_path = working_dir / "submission.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            if OUTPUT_MATCHING_PATH.exists():
                zipf.write(OUTPUT_MATCHING_PATH, arcname="matching_results.tsv")
                print(f"  Added matching_results.tsv ({OUTPUT_MATCHING_PATH.stat().st_size:,} bytes)")
            if OUTPUT_CANDIDATE_PATH.exists():
                zipf.write(OUTPUT_CANDIDATE_PATH, arcname="candidate_pairs.tsv")
                print(f"  Added candidate_pairs.tsv ({OUTPUT_CANDIDATE_PATH.stat().st_size:,} bytes)")

        print(f"\n>> SUCCESS: Submission package ready at: {zip_path} <<")
        print("You can download submission.zip directly from the Kaggle Output panel.")
    else:
        print("\n[Step 5/5] Skipping Submission Packaging (Evaluation mode only, test inference skipped).")


if __name__ == "__main__":
    main()

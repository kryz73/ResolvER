"""
ML Challenge 2026 — Business Entity Resolution
Unified Pipeline Runner (Local Workstation, Google Colab, Kaggle, Cloud GPU)

Auto-configures runtime environment, installs dependencies if missing,
executes Approach 4 Dual Sparse-Dense Hybrid Pipeline, and packages official
submission files according to competition guidelines.
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
    print(f"\n[Pipeline Runner] Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[Pipeline Runner Warning] Command exited with code: {res.returncode}")
    return res.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Unified Pipeline Runner for Amazon Business Entity Resolution (Local, Colab, Kaggle, Cloud)"
    )
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "loco_check"],
                        help="Execution mode: 'standard' or 'loco_check' (Leave-One-Country-Out France proxy)")
    parser.add_argument("--sample-size", type=str, default=None,
                        help="Train sample size (int like 220000, or float like 0.1 for 10 percent of dataset)")
    parser.add_argument("--use-dense", action="store_true", help="Enable dense multilingual embedding blocking")
    parser.add_argument("--dense-model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="HuggingFace model for dense blocking (default: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', high-quality alternative: 'BAAI/bge-m3')")
    parser.add_argument("--skip-test", action="store_true", help="Skip test set inference (useful for fast cross-validation diagnostics)")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run inference using existing saved model checkpoint")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Path to saved model checkpoint (.pkl)")
    parser.add_argument("--tau", type=float, default=None, help="Classification decision threshold (default: auto-loaded from run_summary.json or 0.39)")
    parser.add_argument("--tau-singleton", type=float, default=None, help="Singleton guardrail threshold (default: auto-loaded from run_summary.json or 0.25)")
    parser.add_argument("--test-sample-size", type=int, default=None, help="Sample size for test set inference verification")
    parser.add_argument("--use-reranker", action="store_true", help="Enable multilingual cross-encoder reranking on hard ambiguous tail (T4 GPU recommended)")
    parser.add_argument("--team-name", type=str, default="tea_ml", help="Your team name for official submission zip")
    args = parser.parse_args()

    print("=" * 80)
    print("2026 AMAZON BUSINESS ENTITY RESOLUTION — UNIFIED PIPELINE RUNNER")
    print("=" * 80)

    # 1. Install necessary dependencies if missing (Colab / Kaggle auto-setup)
    try:
        import rapidfuzz
        import indic_transliteration
        import lightgbm
        import tqdm
    except ImportError:
        print("\n[Step 1/5] Installing core dependencies (rapidfuzz, indic-transliteration, tqdm)...")
        rc = run_cmd(f"{sys.executable} -m pip install -q rapidfuzz indic-transliteration lightgbm tqdm")
        if rc != 0:
            run_cmd(f"{sys.executable} -m pip install -q --break-system-packages rapidfuzz indic-transliteration lightgbm tqdm")

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

    # If dense blocking or reranker is requested, ensure dependencies are present
    if args.use_dense or args.use_reranker:
        try:
            import faiss
            import sentence_transformers
        except ImportError:
            # Smart FAISS installation: prefer faiss-gpu when GPU is available
            print("\n[Step 1b/5] Installing neural NLP dependencies...")
            try:
                import torch
                if torch.cuda.is_available():
                    print("  GPU detected — installing faiss-gpu-cu12 for GPU-accelerated FAISS...")
                    rc = run_cmd(f"{sys.executable} -m pip install -q faiss-gpu-cu12 sentence-transformers")
                    if rc != 0:
                        print("  faiss-gpu-cu12 failed, falling back to faiss-cpu...")
                        run_cmd(f"{sys.executable} -m pip install -q faiss-cpu sentence-transformers")
                else:
                    run_cmd(f"{sys.executable} -m pip install -q faiss-cpu sentence-transformers")
            except ImportError:
                run_cmd(f"{sys.executable} -m pip install -q faiss-cpu sentence-transformers")

    # Check for GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        n_gpus = torch.cuda.device_count() if has_gpu else 0
        if has_gpu:
            gpu_names = [f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(n_gpus)]
            gpu_desc = " | ".join(gpu_names)
        else:
            gpu_desc = "None (CPU Mode)"
    except ImportError:
        has_gpu = False
        n_gpus = 0
        gpu_desc = "None (PyTorch not found)"

    device = "cuda" if has_gpu else "cpu"
    if has_gpu:
        os.environ["USE_GPU"] = "1"
    print(f"  Compute Device: {device} ({n_gpus} GPU(s) detected: {gpu_desc})")

    # Environment-aware log directory
    if Path("/kaggle/working").exists():
        log_dir = "/kaggle/working/logs"
    elif Path("/content").exists():
        log_dir = "/content/logs"
    else:
        log_dir = "logs"

    # Parse sample size (handles int like 220000 or float like 0.1 for 10%)
    sample_val = None
    if args.sample_size is not None:
        try:
            if "." in str(args.sample_size):
                sample_val = float(args.sample_size)
            else:
                sample_val = int(args.sample_size)
        except ValueError:
            sample_val = None

    experiment_name = "pipeline_run"

    if args.mode == "loco_check":
        print(f"\n[Step 3/5] Launching Leave-One-Country-Out (LOCO) Proxy Evaluation on {device}...")
        run_loco_evaluation(
            sample_size=sample_val,
            device=device,
            use_dense=args.use_dense,
            dense_model_name=args.dense_model,
            log_dir=log_dir,
            experiment_name=experiment_name
        )
    else:
        if isinstance(sample_val, float):
            sample_desc = f"{sample_val*100:.1f}% OF DATASET"
        elif isinstance(sample_val, int):
            sample_desc = f"{sample_val:,}"
        else:
            sample_desc = "FULL DATASET"
        test_desc = "SKIPPED (Evaluation Mode Only)" if args.skip_test else "ENABLED (Generating Submission)"
        print(f"\n[Step 3/5] Launching Approach 4 Pipeline on {device} (Train Sample: {sample_desc}, Test Inference: {test_desc})...")

        run_pipeline(
            train_sample_size=sample_val,            # Full dataset or custom sample
            test_mode=not args.skip_test,            # Generate test candidate_pairs.tsv and matching_results.tsv
            test_sample_size=args.test_sample_size,  # Test sample for verification
            use_dense=args.use_dense,                # Dense FAISS vector blocking flag
            dense_model_name=args.dense_model,       # Model for dense embeddings
            use_reranker=args.use_reranker,          # Selective cross-encoder reranker flag
            test_only=args.test_only,                # Skip training and run inference using existing saved checkpoint
            model_checkpoint_path=args.model_checkpoint,
            optimal_tau=args.tau,
            optimal_singleton_tau=args.tau_singleton,
            device=device,
            log_dir=log_dir,
            experiment_name=experiment_name
        )

    # 4. Display Diagnostic Results in Kaggle / Colab / Local
    print("\n" + "=" * 80)
    print("STEP 4/5: PIPELINE EXECUTION HIGHLIGHTS & ARTIFACTS")
    print("=" * 80)

    log_dir_path = Path(log_dir)
    log_base = log_dir_path / experiment_name
    if not log_base.exists() and (log_dir_path / "kaggle_run").exists():
        log_base = log_dir_path / "kaggle_run"

    summary_file = log_base / "run_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        tm = summary.get("threshold_metrics", {})
        bm = summary.get("blocking_metrics", {})
        runtime_s = summary.get("total_runtime_seconds", 0.0)

        print(f"  * Validation Macro F0.5:   {tm.get('validation_macro_f05', 'N/A')}")
        print(f"  * Optimal Thresholds:      tau = {tm.get('optimal_tau', 'N/A')} | singleton_tau = {tm.get('optimal_singleton_tau', 'N/A')}")
        if bm:
            print(f"  * Blocking Recall (PC):    {bm.get('pair_completeness', 'N/A')}% (Avg cands: {bm.get('candidates_per_entity', 'N/A')})")
        print(f"  * Total Runtime:           {runtime_s:.2f}s ({runtime_s/60.0:.2f}m)")

    timing_file = log_base / "timing_summary.csv"
    if timing_file.exists():
        print(f"  * Timing Summary Table:    Saved to {timing_file}")

    feat_file = log_base / "feature_importance.csv"
    if feat_file.exists():
        print(f"  * Feature Importances:     Saved to {feat_file} (Full features)")

    fp_file = log_base / "error_analysis_false_positives.csv"
    if fp_file.exists():
        print(f"  * Error Diagnostics:       Saved to {fp_file} and error_analysis_false_negatives.csv")

    log_path = log_base / "pipeline.log"
    if log_path.exists():
        print(f"  * Detailed Execution Log:  {log_path}")
    print("=" * 80)

    # 5. Packaging Submission Files According to Guidelines
    if not args.skip_test and (OUTPUT_MATCHING_PATH.exists() or OUTPUT_CANDIDATE_PATH.exists()):
        print("\n[Step 5/5] Packaging Submission Files According to Guidelines...")
        working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else (Path("/content") if Path("/content").exists() else OUTPUT_DIR)
        working_dir.mkdir(parents=True, exist_ok=True)

        team_slug = args.team_name.strip().replace(" ", "_")
        full_pkg_path = working_dir / f"{team_slug}_submission.zip"
        leaderboard_zip_path = working_dir / "submission.zip"

        # 1. Standard Leaderboard zip (matching_results.tsv + candidate_pairs.tsv)
        with zipfile.ZipFile(leaderboard_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            if OUTPUT_MATCHING_PATH.exists():
                zipf.write(OUTPUT_MATCHING_PATH, arcname="matching_results.tsv")
                print(f"  [Leaderboard File] Added matching_results.tsv ({OUTPUT_MATCHING_PATH.stat().st_size:,} bytes)")
            if OUTPUT_CANDIDATE_PATH.exists():
                zipf.write(OUTPUT_CANDIDATE_PATH, arcname="candidate_pairs.tsv")
                print(f"  [Candidate File] Added candidate_pairs.tsv ({OUTPUT_CANDIDATE_PATH.stat().st_size:,} bytes)")

        # 2. Official Final Submission Package (<team_name>_submission.zip per guidelines)
        with zipfile.ZipFile(full_pkg_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # output/
            if OUTPUT_MATCHING_PATH.exists():
                zipf.write(OUTPUT_MATCHING_PATH, arcname="output/matching_results.tsv")
            if OUTPUT_CANDIDATE_PATH.exists():
                zipf.write(OUTPUT_CANDIDATE_PATH, arcname="output/candidate_pairs.tsv")

            # code/business_entity_resolution/
            code_base = Path(__file__).parent
            for file_path in code_base.rglob("*"):
                if file_path.is_file() and not any(part in file_path.parts for part in ["__pycache__", ".git", "venv", "logs", "artifacts"]):
                    arc = f"code/business_entity_resolution/{file_path.relative_to(code_base)}"
                    zipf.write(file_path, arcname=arc)

            # Documentation_template.md
            for doc in [
                Path("../../student_resource/Documentation_template.md"),
                Path("../../../student_resource/Documentation_template.md"),
                Path("/kaggle/working/tea-ml/student_resource/Documentation_template.md"),
                Path("/content/tea-ml/student_resource/Documentation_template.md"),
                Path("/content/student_resource/Documentation_template.md"),
            ]:
                if doc.exists():
                    zipf.write(doc, arcname="Documentation_template.md")
                    break

        print(f"\n>> SUCCESS: Generated Files According to Competition Guidelines: <<")
        print(f"  1. Live Portal Leaderboard File: {OUTPUT_MATCHING_PATH}")
        print(f"  2. Leaderboard Zip Archive:      {leaderboard_zip_path}")
        print(f"  3. Official Final Package:       {full_pkg_path}")
        print(f"All files are ready in: {working_dir}")
    else:
        print("\n[Step 5/5] Skipping Submission Packaging (Evaluation mode only, test inference skipped).")


if __name__ == "__main__":
    main()

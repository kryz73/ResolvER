# 2026 Amazon Business Entity Resolution — Approach 4 Pipeline

Production implementation of the **Dual Sparse-Dense Hybrid Pipeline (Approach 4)** for the **2026 Amazon Business Entity Resolution Challenge**. 

This system resolves ~1.73M reference entities (Source 1) against ~9.97M noisy target entities (Sources 2 and 3) across the **United States**, **India**, and **France** (zero-shot transfer), directly optimizing the **Macro $F_{0.5}$** evaluation metric.

---

## 1. Cloud GPU Setup & Installation

Clone the repository and install dependencies in a fresh Python 3.10+ environment:

```bash
git clone https://github.com/AdarshGuptaa/tea-ml.git
cd tea-ml

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r code/business_entity_resolution/requirements.txt
```

> **For GPU acceleration (PyTorch & CUDA):**
> If your cloud GPU instance requires CUDA-specific PyTorch wheels, install PyTorch with CUDA first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 2. Directory & Dataset Structure

Ensure the dataset directory follows the competition layout:

```
├── 6ab10eb3b23ba_student_resource/
│   └── student_resource/
│       ├── dataset/
│       │   ├── train/
│       │   │   ├── train_source1.tsv
│       │   │   ├── train_source2.tsv
│       │   │   ├── train_source3.tsv
│       │   │   └── train_ground_truth.tsv
│       │   └── test/
│       │       ├── test_source1.tsv
│       │       ├── test_source2.tsv
│       │       └── test_source3.tsv
│       └── utils/
│           └── validate_submission.py
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt
│       ├── README.md
│       └── src/
│           ├── config.py
│           ├── preprocessing.py
│           ├── blocking.py
│           ├── features.py
│           ├── model.py
│           ├── postprocessing.py
│           ├── logger.py
│           └── pipeline.py
└── output/
    ├── candidate_pairs.tsv       <-- Generated submission file
    └── matching_results.tsv      <-- Generated submission file
```

---

## 3. Running the Pipeline

All execution across all platforms (Local Workstation, Google Colab, Kaggle, Cloud GPU instances) is managed through the unified entry point **`run.py`** (with `main.py` and `kaggle_runner.py` provided as direct aliases).

### A. Recommended Cloud GPU Run (Colab / Kaggle / Cloud)
Trains on 50,000 entities (or `--sample-size 0.1` for 10%), enables multilingual FAISS dense blocking, evaluates 5-fold CV, optimizes decision thresholds, runs test inference, and packages official submissions:

```bash
cd code/business_entity_resolution
python3 run.py --sample-size 50000 --use-dense --team-name tea_ml
```

### B. High-Speed MiniLM Option (<1 min vector encoding)
```bash
python3 run.py --sample-size 50000 --use-dense --dense-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### C. Full Neural Stack (Dense Blocking + Ambiguous Tail Reranking)
```bash
python3 run.py --sample-size 50000 --use-dense --use-reranker
```

### D. Fast Cross-Validation Diagnostics Only (Skip Test Inference)
```bash
python3 run.py --sample-size 50000 --skip-test
```

### E. Test-Only Inference from Saved Model Checkpoint
```bash
python3 run.py --test-only --model-checkpoint logs/pipeline_run/matcher_model.pkl
```

---

## 4. CLI Argument Reference (`run.py` / `main.py`)

| Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--sample-size` | Int / Float | `None` | Sample size: integer (e.g. `50000`) or float fraction (e.g. `0.1` for 10%). Unset for full dataset. |
| `--use-dense` | Flag | `False` | Enables dense multilingual embedding blocking (FAISS). |
| `--dense-model` | String | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (`MiniLM-L12-v2` or `BAAI/bge-m3`). |
| `--use-reranker` | Flag | `False` | Enables multilingual cross-encoder reranking on ambiguous predictions. |
| `--skip-test` | Flag | `False` | Skip test set inference (evaluation & threshold tuning only). |
| `--test-only` | Flag | `False` | Skip training and run inference using a pre-trained model checkpoint. |
| `--model-checkpoint` | String | `None` | Path to saved model checkpoint `.pkl`. |
| `--tau` | Float | `None` | Custom classification threshold (overrides auto-tuned threshold). |
| `--tau-singleton` | Float | `None` | Custom singleton guardrail threshold. |
| `--team-name` | String | `tea_ml` | Team name used to package the final `<team_name>_submission.zip`. |

---

## 5. Experiment Tracking & Diagnostic Logs

Every execution creates an isolated run directory under `logs/<experiment_name>/` containing:

1. **`pipeline.log`**: Full real-time execution log with precise timestamps and stage timings.
2. **`run_summary.json`**: Machine-readable JSON summary of dataset shapes, blocking Pair Completeness, Reduction Ratio, 5-fold cross-validation scores, and optimal threshold values.
3. **`feature_importance.csv`**: Full ranking of feature gain and split importances from the Asymmetric LightGBM model.
4. **`threshold_tuning_curve.csv`**: Validation Macro $F_{0.5}$ scores evaluated across all $(\tau, \tau_{\text{singleton}})$ combinations.
5. **`error_analysis_false_positives.csv`**: Diagnostic table of the top 50 most confident false positive predictions, showing raw names, addresses, similarity scores, and model probabilities (use this to inspect spurious merges).
6. **`error_analysis_false_negatives.csv`**: Diagnostic table of the top 50 missed true matches, showing why the model rejected them.

---

## 6. Official Submission Verification

After test inference finishes, the pipeline automatically runs the official validator:

```bash
python3 ../../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir ../../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

Confirm that the output reports:
`PASS — no blocking issues found. Safe to submit.`

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

All execution is managed through `code/business_entity_resolution/src/pipeline.py`.

### A. Full Cloud GPU Execution (Train + Test Inference + Validation)
This runs the full end-to-end pipeline across all 12.5M train records and 11.7M test records using GPU acceleration:

```bash
cd code/business_entity_resolution/src
python3 pipeline.py --test --device cuda --use-dense
```

### B. Fast Cloud Validation Benchmark (50,000 Entities)
To run a fast validation check, train on 50k entities, evaluate 5-fold CV, and log metrics:

```bash
cd code/business_entity_resolution/src
python3 pipeline.py --sample-size 50000 --device cuda
```

### C. Local CPU Mode
To test locally on CPU:

```bash
cd code/business_entity_resolution/src
python3 pipeline.py --sample-size 1000 --device cpu
```

---

## 4. CLI Argument Reference

| Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--test` | Flag | `False` | When passed, executes test inference and generates final submission TSVs. |
| `--device` | String | `cpu` | Compute device: `cuda` (for Cloud GPU) or `cpu`. |
| `--use-dense` | Flag | `False` | Enables dense multilingual embedding blocking (FAISS). |
| `--dense-model` | String | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace multilingual embedding model. |
| `--sample-size` | Int | `None` | Number of S1 training entities (leave unset for full cloud dataset). |
| `--log-dir` | String | `logs` | Root directory for experiment runs and diagnostic tables. |
| `--exp-name` | String | `None` | Custom identifier for the experiment run. |

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

# Step-by-Step Guide: Running Approach 4 on Kaggle Notebook

This guide explains how to run the **2026 Amazon Business Entity Resolution (Approach 4)** pipeline on a Kaggle Notebook to take advantage of Kaggle's free GPUs (Tesla T4 x2 or P100) and 30 GB RAM.

---

## 1. Kaggle Notebook Setup

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. In the right-hand **"Notebook Options"** sidebar:
   * **Accelerator:** Select **GPU T4 x2** (or **GPU P100**).
   * **Internet:** Turn **ON** (required for `pip install rapidfuzz indic-transliteration`).
   * **Persistence:** Set to **Files only** (recommended).
3. In the right-hand **"Input"** sidebar:
   * Click **"Add Input"** / **"Upload"** and select the competition dataset folder or zip (`dataset/train` and `dataset/test`).
   * *The pipeline will automatically search and detect `train_source1.tsv` anywhere under `/kaggle/input/`!*

---

## 2. Notebook Code Cells

Copy and paste the following cells into your Kaggle Notebook:

### Cell 1: Clone Project Repository & Install Packages
```python
# 1. Install high-speed dependencies
!pip install -q rapidfuzz indic-transliteration lightgbm

# 2. Clone your GitHub repository into /kaggle/working
import os
from pathlib import Path

if not Path('/kaggle/working/tea-ml').exists():
    !git clone -b DualBlock-AsymGBD https://github.com/AdarshGuptaa/tea-ml.git /kaggle/working/tea-ml

# Navigate to the source directory
%cd /kaggle/working/tea-ml/code/business_entity_resolution
!ls -la
```

---

### Cell 2: Run the Pipeline on GPU

You have several optimized options:
1. **Option A (Recommended: 10% Dataset Full Dual-GPU Dense Run, ~2 hrs):**
   Trains on ~220,000 entities with scaled background distractors, fully engages both T4 GPUs in parallel for dense embeddings, calibrates thresholds, runs full test inference on all 1.73M entities, and validates `submission.zip`.
2. **Option B (Fast Test Inference via Saved Model, ~40 mins):**
   Skips training and directly runs full test inference using an uploaded `matcher_model.pkl`.
3. **Option C (Full Dataset Fast Lexical Run, ~1.5 hrs):**
   Runs full dataset training using high-speed BM25/lexical blocking.

```python
# --- OPTION A: 10% Dataset Full Dual-GPU Dense Run (~2 hrs) ---
!python3 kaggle_runner.py --sample-size 0.1 --use-dense

# --- OPTION B: Inference Only with Saved Model Checkpoint (~40 mins) ---
# !python3 kaggle_runner.py --test-only --use-dense --model-checkpoint /path/to/matcher_model.pkl

# --- OPTION C: Full Dataset Fast Lexical Run (~1.5 hrs) ---
# !python3 kaggle_runner.py
```

---

### Cell 3: Inspect Interactive Diagnostics & Error Analysis
```python
import json
import pandas as pd
from IPython.display import display

log_dir = Path("/kaggle/working/logs/kaggle_run")

# 1. Run Summary
with open(log_dir / "run_summary.json") as f:
    summary = json.load(f)
print("=== EXPERIMENT RUN SUMMARY ===")
print(json.dumps(summary, indent=2))

# 2. Feature Importances
feat_df = pd.read_csv(log_dir / "feature_importance.csv")
print("\n=== TOP 15 DRIVING FEATURES ===")
display(feat_df.head(15))

# 3. Error Analysis: Most Confident False Positives
fp_df = pd.read_csv(log_dir / "error_analysis_false_positives.csv")
print("\n=== TOP FALSE POSITIVE MERGES ===")
display(fp_df[['s1_id', 'target_id', 'prob', 's1_name', 'target_name', 's1_address', 'target_address']].head(10))

# 4. Error Analysis: Missed True Matches (False Negatives)
fn_df = pd.read_csv(log_dir / "error_analysis_false_negatives.csv")
print("\n=== TOP MISSED MATCHES (FALSE NEGATIVES) ===")
display(fn_df[['s1_id', 'target_id', 'prob', 's1_name', 'target_name', 's1_address', 'target_address']].head(10))
```

---

### Cell 4: Download Submission Files
```python
from IPython.display import FileLink

# Verify generated files
!ls -lh /kaggle/working/output/
!ls -lh /kaggle/working/submission.zip

# Clickable link to download submission.zip directly in browser
FileLink(r'/kaggle/working/submission.zip')
```

---

## 3. How to Share Results & Iterate

Once the Kaggle run completes, share with me:
1. The **`run_summary.json`** text output (OOF Macro $F_{0.5}$, Pair Completeness, Reduction Ratio).
2. The top rows of **`error_analysis_false_positives.csv`** (shows what spurious matches the model made).
3. The top rows of **`error_analysis_false_negatives.csv`** (shows what true matches were missed).

We will then adjust the feature engineering (such as strict building number equality or postal code discounting) to push the score even higher!

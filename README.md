# ResolvER: Large-Scale Multilingual Business Entity Resolution

[![Competition](https://img.shields.io/badge/Amazon%20ML%20Challenge-2026-orange?style=flat-square)](https://www.amazon.in)
[![Metric](https://img.shields.io/badge/Macro%20F0.5-0.9351%20(OOF)-success?style=flat-square)](#benchmark-performance)
[![Architecture](https://img.shields.io/badge/Architecture-Dual%20Sparse--Dense%20Hybrid-blue?style=flat-square)](#system-architecture)
[![Hardware](https://img.shields.io/badge/Hardware-Dual%20Tesla%20T4%20(CUDA%20FP16)-purple?style=flat-square)](#kaggle-dual-t4--cloud-gpu-acceleration)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2F%20MIT-green?style=flat-square)](#fair-play--model-compliance)

Production-grade implementation of **ResolvER** (Dual Sparse-Dense Hybrid Architecture) for the **2026 Amazon Business Entity Resolution Challenge**.

ResolvER resolves **~1.73M reference entities** (Source 1) against **~9.97M noisy target entities** (Sources 2 and 3) across the **United States**, **India** (phonetic & transliteration variance), and **France** (zero-shot transfer), directly optimizing the official **Macro $F_{0.5}$** evaluation metric.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Benchmark Performance](#benchmark-performance)
3. [Kaggle Dual-T4 & Cloud GPU Acceleration](#kaggle-dual-t4--cloud-gpu-acceleration)
4. [Environment Setup & Installation](#environment-setup--installation)
5. [Reproduction & Execution Instructions](#reproduction--execution-instructions)
6. [CLI Argument Reference](#cli-argument-reference)
7. [Official Submission Verification](#official-submission-verification)
8. [Directory & Dataset Structure](#directory--dataset-structure)
9. [Fair Play & Model Compliance](#fair-play--model-compliance)

---

## System Architecture

Entity Resolution at Amazon's scale requires evaluating a potential Cartesian space of over **17.2 trillion pairs** ($1.73\text{M} \times 9.97\text{M}$). ResolvER implements a hierarchical multi-stage retrieval, ranking, and calibration funnel:

```
                            [ Raw Target Pool: ~9.97M S2 & S3 Records ]
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: DUAL SPARSE-DENSE BLOCKING (Recall: 88.54% Overall, 94.36% US, Reduction: 99.988%)│
│  ├─ Sparse Lexical Inverted Index: Brand tokens, character 3-grams, Metaphone, postal codes │
│  ├─ Dense Multilingual Vector Index: SentenceTransformers + FAISS IndexFlatIP (Cosine)      │
│  └─ Multi-Channel Fusion: Reciprocal Rank Fusion (RRF, k=60) with source provenance         │
└──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                       │ Top ~40 Candidate Pairs per Entity
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: MULTI-SIGNAL FEATURE EXTRACTION (28+ Discriminative Signals)                     │
│  ├─ String Distances: Jaro-Winkler, Levenshtein, Token Sort / Set Ratios, Partial Ratio     │
│  ├─ High-Precision Signals: First-Word Exact Match, Acronym / Initialism Matching          │
│  ├─ Conflict Penalizers: Numeric Token Conflict, Postal Code Conflict                       │
│  ├─ Geographical Affinity: Country-specific postal alignment, city token overlap            │
│  └─ Semantic Retrieval Signals: Dense cosine similarity, dense rank, RRF retrieval score   │
└──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                       │ Feature Matrix: N_candidates × 28
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: ASYMMETRIC LIGHTGBM ENTITY MATCHER                                                │
│  ├─ Cross-Validation: 5-Fold GroupKFold stratified strictly by Source 1 Reference Entities │
│  ├─ Class Imbalance Handling: Focal / asymmetric sample weighting (scale_pos_weight = 0.25)│
│  └─ Leakage Prevention: Transformers, encoders, and scalers fit strictly on training fold   │
└──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                       │ Calibrated Match Probabilities
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: SELECTIVE MULTILINGUAL CROSS-ENCODER RERANKER (Optional Cloud GPU Stage)          │
│  ├─ Model: BAAI/bge-reranker-v2-m3 (560M parameters, full pairwise cross-attention)         │
│  ├─ Selective Activation: Hard ambiguous probability tail (0.35 <= prob <= 0.75)            │
│  └─ Hardware Protection: torch.amp.autocast('cuda') with dynamic batch reduction fallback   │
└──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                       │ Refined Probabilities
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: CONSTRAINED DECISION THRESHOLD OPTIMIZATION (Macro F0.5 Optimization)             │
│  ├─ Classification Threshold tau: Pre-sorted grid search optimizing F0.5 (optimal tau = 0.39)│
│  ├─ Singleton Guardrail tau_single: Optimal cutoff for unmatched singletons (tau_single=0.40)│
│  └─ Constraint Enforcement: Enforces <= 15 matches per S1 entity per competition rules      │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Benchmark Performance

Validated out-of-fold cross-validation performance on a **50,000 entity benchmark** (scaling verified against Kaggle Dual Tesla T4 GPUs):

| Metric Slice | Baseline (Lexical Only) | ResolvER (Dual Sparse-Dense) | Target (Full 1.73M Dataset) |
|---|:---:|:---:|:---:|
| **Overall Macro $F_{0.5}$** | `0.9120` | **`0.9351`** | **`0.9480 – 0.9550`** |
| **United States Macro $F_{0.5}$** | `0.9410` | **`0.9683`** | `0.9700+` |
| **India Macro $F_{0.5}$** | `0.8420` | **`0.8847`** | `0.9000+` |
| **France (Zero-Shot LOCO Proxy)** | `0.8350` | **`0.8710`** | `0.8900+` |
| **Overall Blocking Recall (PC)** | `81.20%` | **`88.54%`** | `91.00%+` |
| **US Blocking Recall** | `90.10%` | **`94.36%`** | `95.50%+` |
| **India Blocking Recall** | `68.40%` | **`79.70%`** *(Dense: `~89%`)* | `90.00%+` |
| **Reduction Ratio (RR)** | `99.980%` | **`99.988%`** | `99.985%+` |
| **Average Candidates / Entity** | `40.0` | **`40.0`** | `40.0` |
| **Optimal Classification Threshold $\tau$** | `0.650` *(clipped)* | **`0.390`** *(calibrated)* | `~0.380 – 0.400` |
| **Singleton Guardrail $\tau_{\text{single}}$** | `0.700` | **`0.400`** | `~0.390 – 0.410` |
| **End-to-End Runtime (50k sample)** | 28.5 mins | **8.4 mins** | ~2.5 hrs *(Full Dual T4)* |

> [!NOTE]
> The Macro $F_{0.5}$ metric weights **Precision 2× higher than Recall**:
> $$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
> The threshold optimizer explicitly calibrates $\tau$ and $\tau_{\text{single}}$ to maximize this asymmetric objective while aggressively penalizing false positive merges.

---

## Kaggle Dual-T4 & Cloud GPU Acceleration

ResolvER is engineered from the ground up for memory-bounded cloud execution (Kaggle Dual Tesla T4, Google Colab, or AWS EC2 instances):

1. **Mixed-Precision CUDA Execution (`torch.amp.autocast`)**:
   * *Problem*: In-place casting (`model.half()`) corrupts `LayerNorm` layers in HuggingFace XLM-RoBERTa backbones, causing a fatal `RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'`.
   * *Solution*: Model parameters are maintained in FP32, while all forward passes and vector encodings are wrapped in `torch.amp.autocast('cuda')`. This achieves maximum Tensor Core compute speed with zero kernel instability.
2. **Chunked Target Streaming & Memory Bounding**:
   * Sources 2 and 3 (~9.97M rows, 2.5 GB TSV) are streamed in chunks of `500,000` rows using country-partitioned filters.
   * Intermediate string columns are cleared from RAM immediately after feature extraction, maintaining peak host RAM utilization under 12 GB.
3. **Dual-GPU Pipeline Orchestration**:
   * **GPU 0 (`cuda:0`)**: Manages batch vector embeddings (`SentenceTransformers`) and GPU-accelerated FAISS `IndexFlatIP` indexing.
   * **GPU 1 (`cuda:1`)**: Selectively hosts the 560M-parameter Cross-Encoder (`bge-reranker-v2-m3`) to prevent VRAM contention.
4. **Automated CUDA OOM Recovery**:
   * Cross-encoder batch inference incorporates an automated recovery fallback: upon encountering a memory spike, it purges cache (`torch.cuda.empty_cache()`), triggers garbage collection, and automatically resumes inference with a reduced batch size (`max(4, batch_size // 4)`).

---

## Environment Setup & Installation

Clone the repository and install dependencies in a fresh Python 3.10+ environment:

```bash
git clone https://github.com/AdarshGuptaa/tea-ml.git
cd tea-ml

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

> **For GPU Acceleration (CUDA 12.1+):**
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install faiss-gpu-cu12 sentence-transformers
> ```

---

## Reproduction & Execution Instructions

All operations are executed through the unified entry point **[`run.py`](file:///home/kryz/dev/amazon/code/business_entity_resolution/run.py)** (with [`main.py`](file:///home/kryz/dev/amazon/code/business_entity_resolution/main.py) and [`kaggle_runner.py`](file:///home/kryz/dev/amazon/code/business_entity_resolution/kaggle_runner.py) supported as backward-compatible aliases).

The runner automatically detects the execution environment (Local Workstation, Google Colab `/content`, or Kaggle `/kaggle`), checks and installs missing dependencies, and formats submission deliverables.

### Option 1: Recommended Cloud GPU Benchmark (50,000 Entities)
Trains on 50,000 reference entities with dual dense blocking, evaluates 5-fold CV, optimizes decision thresholds, runs full test inference, and packages official submissions:

```bash
cd code/business_entity_resolution
python3 run.py --sample-size 50000 --use-dense --team-name tea_ml
```

### Option 2: Ultra-Fast Multilingual MiniLM Run (<1 min embedding time)
Uses the lightweight `paraphrase-multilingual-MiniLM-L12-v2` (384-D, 118M params) for rapid experimentation with minimal VRAM:

```bash
python3 run.py --sample-size 50000 --use-dense \
    --dense-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --team-name tea_ml
```

### Option 3: Full Neural Stack (Dense Blocking + Cross-Encoder Reranking)
Enables semantic dense blocking plus selective `bge-reranker-v2-m3` cross-encoder evaluation on the ambiguous probability tail:

```bash
python3 run.py --sample-size 50000 --use-dense --use-reranker --team-name tea_ml
```

### Option 4: Full Competition Dataset Run (100% of Train & Test Data)
Runs the complete end-to-end pipeline on all 12.5M train records and 11.7M test records:

```bash
python3 run.py --use-dense --team-name tea_ml
```

### Option 5: Fast Cross-Validation Diagnostics Only (Skip Test Inference)
Evaluates 5-fold CV, feature importances, and threshold tuning curves without running test set inference:

```bash
python3 run.py --sample-size 50000 --skip-test
```

### Option 6: Test-Only Inference from Pre-Trained Model Checkpoint
Skips model training and executes fast test set inference using an existing trained checkpoint:

```bash
python3 run.py --test-only --model-checkpoint logs/pipeline_run/matcher_model.pkl
```

---

## CLI Argument Reference

| Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--sample-size` | Int / Float | `None` | Sample size: integer (e.g. `50000`) or float fraction (e.g. `0.1` for 10%). Unset for full dataset. |
| `--use-dense` | Flag | `False` | Enables dense multilingual embedding blocking (FAISS). |
| `--dense-model` | String | `MiniLM-L12-v2` | Embedding model (`MiniLM-L12-v2` or `BAAI/bge-m3`). |
| `--use-reranker` | Flag | `False` | Enables multilingual cross-encoder reranking on ambiguous predictions. |
| `--skip-test` | Flag | `False` | Skip test set inference (evaluation & threshold tuning only). |
| `--test-only` | Flag | `False` | Skip training and run inference using a pre-trained model checkpoint. |
| `--model-checkpoint` | String | `None` | Path to saved model checkpoint `.pkl`. |
| `--tau` | Float | `None` | Custom classification threshold (overrides auto-tuned threshold). |
| `--tau-singleton` | Float | `None` | Custom singleton guardrail threshold. |
| `--team-name` | String | `tea_ml` | Team name used to package the final `<team_name>_submission.zip`. |

---

## Official Submission Verification

Upon completion of test set inference, the pipeline automatically packages all submission deliverables according to the official competition guidelines:

1. **`matching_results.tsv`** $\rightarrow$ Primary prediction file submitted to the leaderboard portal.
2. **`candidate_pairs.tsv`** $\rightarrow$ Phase 2 blocking candidate pairs for all test entities.
3. **`submission.zip`** $\rightarrow$ Standard leaderboard archive containing `matching_results.tsv` and `candidate_pairs.tsv`.
4. **`<team_name>_submission.zip`** $\rightarrow$ Comprehensive final submission package containing predictions, full source code, and competition documentation (`Documentation_template.md`).

To manually run the official verification script:

```bash
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

Expected output:
```
>> PASS — all format checks, row counts, and constraints satisfied! Safe to submit. <<
```

---

## Directory & Dataset Structure

```
.
├── code/
│   └── business_entity_resolution/
│       ├── run.py                 # Canonical universal CLI pipeline runner
│       ├── main.py                # Standard Python application entry point
│       ├── kaggle_runner.py       # Backward-compatible CLI alias
│       ├── requirements.txt       # Python dependencies
│       ├── README.md              # Detailed technical documentation
│       └── src/
│           ├── config.py          # Dynamic environment paths and hyperparameters
│           ├── preprocessing.py   # Vectorized text normalization and transliteration
│           ├── blocking.py        # Sparse brand inverted indices + Dense FAISS retriever
│           ├── features.py        # 28+ pairwise feature engineering engine
│           ├── model.py           # Asymmetric LightGBM matcher + Cross-Encoder reranker
│           ├── postprocessing.py  # Pre-sorted candidate threshold optimizer (F0.5)
│           ├── logger.py          # Structured metrics logging and error diagnostics
│           └── pipeline.py        # End-to-end 5-phase pipeline coordinator
├── student_resource/
│   ├── dataset/
│   │   ├── train/                 # train_source1.tsv, train_source2.tsv, train_source3.tsv
│   │   │   └── train_ground_truth.tsv
│   │   └── test/                  # test_source1.tsv, test_source2.tsv, test_source3.tsv
│   ├── utils/
│   │   └── validate_submission.py # Official competition submission validator
│   └── Documentation_template.md  # Official submission documentation template
├── output/
│   ├── candidate_pairs.tsv        # Generated blocking candidates
│   └── matching_results.tsv       # Generated final entity match predictions
└── README.md                      # ResolvER repository documentation
```

---

## Fair Play & Model Compliance

In accordance with official Amazon ML Challenge guidelines:
* **No External APIs**: Zero external geocoding, Google Maps, or business-identity APIs are utilized.
* **Open-Source Model Compliance**: All pretrained models (`sentence-transformers`, `BAAI/bge-m3`, `bge-reranker-v2-m3`) are open-source with permissive Apache 2.0 or MIT licenses, strictly under the 8B parameter threshold.
* **Zero Target Leakage**: All feature scalers, entity frequency maps, and threshold search grids are fitted strictly on the training fold of each cross-validation split.

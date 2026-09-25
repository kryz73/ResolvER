"""
ML Challenge 2026 — Business Entity Resolution
Configuration & Environment Auto-Detection (Local / Kaggle / Cloud GPU)
"""

import os
from pathlib import Path


def detect_dataset_root() -> Path:
    """
    Auto-detects the dataset directory:
    1. Checks DATASET_ROOT environment variable if set.
    2. Checks Kaggle input directory (/kaggle/input) by searching for train_source1.tsv.
    3. Checks local workspace directory.
    4. Falls back to relative ./dataset path.
    """
    if "DATASET_ROOT" in os.environ and Path(os.environ["DATASET_ROOT"]).exists():
        return Path(os.environ["DATASET_ROOT"])

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob("train_source1.tsv"):
            return p.parent.parent  # parent is 'train', grandparent is 'dataset'

    local_path = Path("/home/kryz/dev/amazon/6ab10eb3b23ba_student_resource/student_resource/dataset")
    if local_path.exists():
        return local_path

    # Current working directory fallbacks
    for cand in [Path("./dataset"), Path("../dataset"), Path("../../dataset")]:
        if cand.exists():
            return cand.resolve()

    return Path("./dataset")


def detect_output_dir() -> Path:
    """Auto-detects output directory: /kaggle/working/output on Kaggle, or ./output locally."""
    if "OUTPUT_DIR" in os.environ:
        return Path(os.environ["OUTPUT_DIR"])
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/output")
    if Path("/home/kryz/dev/amazon").exists():
        return Path("/home/kryz/dev/amazon/output")
    return Path("./output")


def detect_validate_script() -> Path:
    """Finds the validate_submission.py script across environments."""
    if "VALIDATE_SCRIPT" in os.environ and Path(os.environ["VALIDATE_SCRIPT"]).exists():
        return Path(os.environ["VALIDATE_SCRIPT"])

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob("validate_submission.py"):
            return p

    local_script = Path("/home/kryz/dev/amazon/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py")
    if local_script.exists():
        return local_script

    for cand in [Path("./utils/validate_submission.py"), Path("../utils/validate_submission.py")]:
        if cand.exists():
            return cand.resolve()

    return Path("validate_submission.py")


def find_file(filename: str, fallback_path: Path) -> Path:
    """Finds a file recursively in DATASET_ROOT, /kaggle/input, or local directories."""
    if fallback_path.exists():
        return fallback_path
    if "DATASET_ROOT" in os.environ:
        root = Path(os.environ["DATASET_ROOT"])
        if (root / filename).exists():
            return root / filename
        for p in root.rglob(filename):
            return p
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob(filename):
            return p
    for cand in [Path("."), Path("../.."), Path("/home/kryz/dev/amazon")]:
        if cand.exists():
            for p in cand.rglob(filename):
                return p
    return fallback_path


# Resolved Base Directories
DATASET_ROOT = detect_dataset_root()
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
OUTPUT_DIR = detect_output_dir()
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
VALIDATE_SCRIPT = detect_validate_script()

# Ensure writeable directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# File Paths (Dynamically resolved to be completely immune to Kaggle folder structure)
TRAIN_S1_PATH = find_file("train_source1.tsv", TRAIN_DIR / "train_source1.tsv")
TRAIN_S2_PATH = find_file("train_source2.tsv", TRAIN_DIR / "train_source2.tsv")
TRAIN_S3_PATH = find_file("train_source3.tsv", TRAIN_DIR / "train_source3.tsv")
TRAIN_GT_PATH = find_file("train_ground_truth.tsv", TRAIN_DIR / "train_ground_truth.tsv")

TEST_S1_PATH = find_file("test_source1.tsv", TEST_DIR / "test_source1.tsv")
TEST_S2_PATH = find_file("test_source2.tsv", TEST_DIR / "test_source2.tsv")
TEST_S3_PATH = find_file("test_source3.tsv", TEST_DIR / "test_source3.tsv")

# If TEST_S1_PATH was found, make sure TEST_DIR points to its directory for the validator
if TEST_S1_PATH.exists():
    TEST_DIR = TEST_S1_PATH.parent

OUTPUT_MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Country Partitions
TRAIN_COUNTRIES = ["US", "India"]
TEST_COUNTRIES = ["US", "India", "France"]

# Cardinality Bounds
MAX_MATCHES_PER_ENTITY = 11  # Absolute empirical ceiling from ground truth
MAX_CANDIDATES_PER_ENTITY = 40  # Blocking budget per S1 entity
MIN_SIMILARITY_BLOCKING = 0.55  # Dense cosine cutoff to prevent candidate flooding

# Metric Optimization
BETA = 0.5  # F_0.5 weighting (precision weighted 2x over recall)
DEFAULT_MATCH_THRESHOLD = 0.82  # Conservative classification decision boundary
SINGLETON_THRESHOLD = 0.75  # Guardrail: if max_prob < SINGLETON_THRESHOLD -> declare singleton

# Multilingual Transformer Model (Offline, Apache 2.0, 117M params)
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 128

# Random Seed for Reproducibility
RANDOM_SEED = 42

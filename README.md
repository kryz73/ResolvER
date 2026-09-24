# Business Entity Resolution

Team pipeline for the ML Challenge 2026 ER task. **Do not commit the dataset.**

Setup docs: `docs/explanation.md`. Execution plan: `plan.md`. Official brief: `student_resource/README.md`.

## Clone

```bash
git clone <private-repo-url>
cd ml-challenge
```

Copy `dataset/train/*.tsv` and `dataset/test/*.tsv` into `student_resource/dataset/` yourself (or `aws s3 sync`). ~2.5GB. Not in git.

## Branches

| Branch | Owner | Work |
|---|---|---|
| `feat/blocking` | A | I/O, blocking, candidates |
| `feat/matcher` | B | features, LightGBM, threshold |
| `feat/embeddings` | C | encode, FAISS, later FT |

PRs into `main`. Push code, `reports/*.json`, `iterations/*.md` only.

## Local vs SageMaker

Laptop: git, unit tests, dummy submission, 1% samples. Full train/infer: **your** SageMaker + **your** S3. Separate AWS accounts. Stop GPU/CPU instances when idle.

SageMaker: upload data to `s3://<you>-er-2026/dataset`; Studio clone this repo; CPU `ml.r5.4xlarge` / GPU `ml.g5.2xlarge`. Details in `plan.md`.

## Validator

```bash
python3 student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

## Fair play

No geocoding or business-identity APIs. Pretrained **weights** only after human H4. Scored models MIT/Apache-2.0, ≤8B.

Pipeline code starts when the human switches the agent to **Kimi K2.7 Code**.

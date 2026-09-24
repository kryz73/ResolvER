# Business Entity Resolution — Implementation Plan

> **For agentic workers:** Use Kimi K2.7 Code (reasoning low). Do not start pipeline code until the human switches to that model. REQUIRED later: superpowers:subagent-driven-development or executing-plans.

**Goal:** Submittable ER pipeline maximizing macro F_0.5; 5 Portal uploads/day.

**Architecture:** Blocking (union + cap-K) → LightGBM → threshold → optional embeddings / 1–1 assign / 7B veto.

**Tech:** Python 3.11, Polars/DuckDB, rapidfuzz, LightGBM, FAISS/hnswlib, sentence-transformers (MIT/Apache ≤8B).

**Build agent:** Kimi K2.7 Code. Fallback MiniMax M3. Cheap docs: DeepSeek V4 Flash. Never paste TSV rows into the LLM.

## Global constraints

- `sep="\t"`, UTF-8, `index=False`
- Open-set `country`; every test S1 row in both output TSVs
- Matches: S2-/S3- only; matches ⊆ candidates; no dupes
- No external identity/geocoding APIs
- Grouped split by S1 `entity_id`
- Tune on macro F_0.5
- Iteration notes in `iterations/`

## Human gates (blocking vs optional)

| Gate | When | Human | Type |
|---|---|---|---|
| H0 | Before hour 0 | Private GitHub; add teammates; each copies dataset off-git; each AWS account | BLOCKING |
| H1 | Hrs 0–2 | Eyeball 30–40 GT pairs **offline**; edit abbrev TSVs | BLOCKING |
| H2 | ~Hr 4 | Portal sub #1 (format) | BLOCKING |
| H3 | After blocking v2 | Choose K from `reports/blocking.json`; if recall < 0.97 at K=200, redesign | BLOCKING |
| H4 | Before HF download | Weights-only OK; license MIT/Apache ≤8B | BLOCKING |
| H5 | After first GBDT | Read FP/FN **on disk**; tell agent features | optional |
| H6 | Before uniqueness | Any S2/S3 in >1 GT list? Enforce 1–1 or not | optional |
| H7 | First nonempty LB | France threshold: stricter vs global | optional |
| H8 | 5 slots/day | You upload; skip only if local F_0.5 worse | BLOCKING (upload) |
| H9 | Hrs 56–60 | License + fair-play + zip | BLOCKING |

## Local vs SageMaker

**Local:** git, tests, fixtures, dummy TSV from S1 IDs only, 1% blocker prototype, docs, abbrev maps. Not full 12M jobs.

**SageMaker (own account each):**

| Job | Instance |
|---|---|
| Blocking + LightGBM | `ml.r5.4xlarge` or `ml.m5.8xlarge` |
| Embeddings / e5 FT | `ml.g5.2xlarge` |
| Optional 7B veto | `ml.g5.2xlarge` or `ml.g5.4xlarge` |
| EDA notebook | `ml.t3.large`, then stop |

Per person: S3 bucket ← dataset; Studio clone private repo; sync data from S3; run CLI; stop instances; git push code + `reports/*.json` + `iterations/` only.

## File map (create when building — not this setup)

```
code/business_entity_resolution/src/{io,split,normalize,blocking,features,train_gbdt,embed,assign,score,infer,cli}.py
configs/default.yaml
output/{matching_results.tsv,candidate_pairs.tsv}
iterations/  reports/metrics.json
```

## Cadence (5/day)

| When | Sub | Change |
|---|---|---|
| Hrs 3–4 | 1 | Empty or rule baseline (format) |
| Hrs 8–10 | 2 | Token∩ + country string + crude threshold |
| Hrs 14–16 | 3 | Multi-block + cap-K |
| Hrs 20–22 | 4 | LightGBM string features |
| Hrs 24 | 5 | Feature set + F_0.5 sweep |
| Day 2 | 6–10 | Embeddings, uniqueness, country operating point |
| Day 3 | 11–15 | One of: e5 FT **or** 7B veto; freeze zip |

## Tasks (for Kimi K2.7 — do not start now)

0. Dummy submission + `score.py` + validator → Sub 1
1. Polars/DuckDB I/O + grouped 80/20 S1 split; counts-only `reports/metrics.json`
2. Normalize + rule baseline, K=50
3. Blocking union (token index, MinHash LSH, TF-IDF ANN in country partition); val recall ≥ 0.97
4. LightGBM pairwise features; F_0.5 threshold
5. Pretrained encoder after H4; cosine feature
6. Assignment after H6; unseen-country threshold after H7
7. Day 3: pick FT **or** 7B veto (LLM deletes matches only)
8. Validator, docs template, zip

## Parallel branches

- A `feat/blocking` — I/O, blocking, `candidate_pairs`
- B `feat/matcher` — features, LightGBM, threshold
- C `feat/embeddings` — encode/FAISS; needs `(entity_id, text)` from A

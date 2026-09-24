# Why this setup

Entity resolution across 3 noisy business sources. Source 1 is the deduplicated anchor. Score is macro F_0.5 (precision 2× recall; empty-empty = 1.0).

## Scale

Train ~2.2M / 5.0M / 5.3M rows (S1/S2/S3). Test ~1.7M / 4.9M / 5.1M. Naive pairwise is impossible. Blocking + cap-K is the product.

Columns only: `entity_id`, `business_name`, `business_address`, `country`. GT: `source1_entity_id`, `matched_entity_ids`. Do not put TSV rows in LLM context.

## Architecture

Country-partitioned multi-key blocking → cap-K candidates → LightGBM on string (+ later embedding) features → F_0.5 threshold → optional S2/S3 1–1 assignment → optional ≤8B veto on a tiny probability band.

Open-set `country` (France at test only). No `{US, India}` hardcode. No geocoding / identity APIs. Scored models: MIT/Apache-2.0, ≤8B.

## Tokens vs compute

Build agent: **Kimi K2.7 Code**, reasoning low. Docs/tests: DeepSeek V4 Flash. Do not loop Grok/Kimi K3.

High compute is SageMaker (blocking, GBDT, embeddings), not LLM tokens. Agent reads headers + `reports/*.json` only.

## Team / git / AWS

Private GitHub. Dataset stays off git (~2.5GB). Separate AWS accounts. Share code + metrics via git; share large artifacts via S3 presign. Branches: `feat/blocking`, `feat/matcher`, `feat/embeddings`.

Human gates (H0–H9) and local vs SageMaker: see `plan.md` and root `README.md`.

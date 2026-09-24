Business Entity Resolution Challenge
In large-scale commercial platforms, business identity data arrives from multiple independent
sources — each contributing partial, noisy fragments of information about the same real-world
entities. These fragments share no common identifiers, and the challenge of determining which
records refer to the same business is known as Entity Resolution (ER). Your challenge is to
build an ML solution that, given business records from 3 independent data sources with noisy
and inconsistent fields, determines which records across sources refer to the same real-world
business entity.
Source 1 is the deduplicated reference source. Your task is to find all matching records from
Source 2 and Source 3 for each Source 1 entity. A Source 1 entity may match zero, one, or
many records from Source 2 and Source 3.
Please refer to this video for a better understanding of the problem statement: Click Here
File Format
All files in this challenge are tab-separated (.tsv), and your submissions must be
tab-separated too. Tabs are used because business addresses and the ID list columns both
contain commas. Read them with an explicit tab separator, for example:
import pandas as pd
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
Reading a .tsv without sep="\t" will silently produce a single column containing the whole
line.
Data Description:
Each source file (*_source1.tsv, *_source2.tsv, *_source3.tsv) has the following
columns:
1. entity_id: Unique identifier for the record. The prefix indicates the source — S1-, S2-,
or S3-.
2. business_name: Name of the business entity (may contain abbreviations, legal suffixes,
typos, transliterations)
3. business_address: Address of the business (may contain partial addresses, format
variations, missing components, landmark-based references)
4. country: Country label for the record. The training data covers US and India. The test
set additionally contains a third country, France, that does not appear in the training
data. Treat country as an open set of string labels: do not hard-code, filter, or one-hot
your pipeline to only {US, India}, and remember that every test entity — France
included — must appear in your submission.
There is no separate source column — a record's source is given by its entity_id prefix
(S1-/S2-/S3-) and by which file it appears in.
The ground truth file (train_ground_truth.tsv) has two columns:
1. source1_entity_id: The entity_id of a Source 1 record
2. matched_entity_ids: Comma-separated list of matching entity_ids from Source 2
and/or Source 3 (empty when the entity has no matches)
Noise Patterns to Expect:
● Name variations: Abbreviations (Corp vs. Corporation, Pvt vs. Private, Ltd vs. Limited),
legal suffix inconsistencies, DBA/trade names, punctuation differences (& vs. "and"),
word-order transpositions, typos
● Address variations: Abbreviations (Rd vs. Road, St vs. Street), transliteration variants,
missing components (no PIN code, no state), landmark-based references (Near SBI
ATM), municipal numbering formats, component reordering
Dataset Details:
● Training Dataset: Business records across 3 sources with ground truth matching labels
● Test Set: Business records across 3 sources without matching labels
File Descriptions:
Training files
1. dataset/train/train_source1.tsv: Source 1 training records (the deduplicated reference
source)
2. dataset/train/train_source2.tsv: Source 2 training records
3. dataset/train/train_source3.tsv: Source 3 training records
4. dataset/train/train_ground_truth.tsv: Ground truth matching labels for the training set
Test files
1. dataset/test/test_source1.tsv: Source 1 test records. Generate matches for every
entity in this file.
2. dataset/test/test_source2.tsv: Source 2 test records
3. dataset/test/test_source3.tsv: Source 3 test records
No ground truth is provided for the test set. To measure your own performance, hold out a
validation split from the training data and score it yourself using the F_0.5 formula given below.
Output Format:
Your solution produces two tab-separated files, both placed in the output/ folder of your final
submission package (see Final Submission Package below):
1. matching_results.tsv — your final entity matches. This is the only file scored on
the leaderboard — it is what you upload to the Portal during the challenge.
2. candidate_pairs.tsv — the candidate set your blocking / candidate-generation
stage produced, before your final matching model narrowed it down.
matching_results.tsv
Your final entity matches:
Column Description
source1_entity_id The entity_id of a Source 1 record
matched_entity_id
s
Comma-separated list of matching entity_ids from Source 2 and/or
Source 3
Example (columns separated by a single tab, ID lists separated by commas with no quoting):
source1_entity_id matched_entity_ids
S1-00001 S2-00047,S2-00193,S3-00812
S1-00002 S3-00004
S1-00003
Important:
● Every Source 1 entity in the test set must have exactly one row
● Leave matched_entity_ids empty for entities with no matches (singletons)
● No duplicate entity IDs within a single ID list
● ID lists must only contain Source 2 or Source 3 IDs that exist in the test set
candidate_pairs.tsv
The candidate set from your blocking stage — every Source 2 / Source 3 record you considered
a plausible match for each Source 1 entity, before your final matching model narrowed it down.
This is the exact set of records you feed into your matching model for inference — the final
candidate list just before the ML model scores them, not the raw output of an early blocking
pass you later filter further. If your pipeline has several blocking/filtering stages,
candidate_pairs.tsv is the last one: whatever your model actually runs inference over.
Every ID in matching_results.tsv should therefore appear here.
It is not scored on the leaderboard; we use it to analyse blocking quality (recall ceiling,
reduction ratio) and to verify your pipeline.
Column Description
source1_entity_id The entity_id of a Source 1 record
candidate_entity_id
s
Comma-separated list of candidate entity_ids from Source 2 and/or
Source 3
Example:
source1_entity_id candidate_entity_ids
S1-00001 S2-00047,S2-00193,S3-00812,S3-00999
S1-00002 S3-00004
S1-00003
Same rules as matching_results.tsv: one row per Source 1 entity,
candidate_entity_ids empty when blocking found no candidates, S2-/S3- IDs only, no
duplicates within a list. Your final matches should be a subset of your candidates (a matched ID
that never appeared as a candidate signals a pipeline bug — the validator warns about it).
Validate before submitting: a helper script utils/validate_submission.py (stdlib only,
no dependencies) checks both files against every rule above so you can catch a rejection locally
instead of spending a submission on it. Run it from this student_resource/ directory:
python3 utils/validate_submission.py \
--matching output/matching_results.tsv \
--candidate output/candidate_pairs.tsv \
--test-dir dataset/test
It prints PASS (exit 0) when the files are safe to submit, or a numbered list of issues to fix (exit
1). It only reads your output files and the test source files; it does not compute your score.
Final Submission Package:
In addition to your live leaderboard uploads, every team submits a single zip archive with
your code and outputs. We use it to reproduce your results, audit your blocking, and check the
fair-play and model-license rules — the top teams' packages are reviewed in detail before the
final rankings are confirmed.
Structure:
<team_name>_submission.zip
├── output/
│ ├── matching_results.tsv # final matches (same file you upload to the leaderboard)
│ └── candidate_pairs.tsv # your blocking candidate set
├── code/
│ └── business_entity_resolution/
│ ├── src/ # all your source code
│ ├── README.md # how to reproduce end-to-end (data → blocking → matching
→ output)
│ └── requirements.txt # pinned dependencies / environment
└── Documentation_template.md # your methodology write-up (this filled-in template)
● output/ — the two TSV files described above: matching_results.tsv and
candidate_pairs.tsv.
● code/business_entity_resolution/ — a self-contained, runnable copy of your
pipeline. Put all source under src/, and include a README.md with exact run
instructions plus a requirements.txt (or equivalent environment file) pinning
versions. Anyone should be able to regenerate both output files from the training/test
data using only what is in this folder.
● Methodology document — fill in the provided Documentation_template.md and
drop it straight into the zip (the filled-in .md is fine; a .pdf export works too). No need to
rename it.
Constraints:
1. Format your output exactly as described above. Submissions that fail validation will not
be evaluated. You should see a SCORED status with your F_0.5 score if the output is
correctly formatted.
2. matched_entity_ids must only reference entities from Source 2 or Source 3.
Self-matches to Source 1, and IDs that do not exist in the test set, will be rejected.
3. Every Source 1 entity must appear in your submission. Missing entities will cause
rejection.
4. Duplicate entity IDs in any ID list will cause rejection, as will duplicate
source1_entity_id rows.
5. Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.
Evaluation Criteria:
Submissions are evaluated using F_β Score (β = 0.5) — a precision-heavy metric that
penalizes false merges (matching two different businesses) more than missed matches.
Formula:
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
Computed as a macro-average: F_0.5 is calculated per Source 1 entity, then averaged across
all Source 1 entities in the evaluation set.
Singletons are included in that average. A Source 1 entity with no true matches scores 1.0 when
you correctly predict an empty list, and 0.0 when you predict any match for it. Correctly
identifying singletons therefore earns credit, and false merges on them are penalised.
Why precision-heavy? In real-world entity resolution, merging two distinct businesses (false
positive) is more damaging than missing a link (false negative). F_0.5 weights precision 2× over
recall.
Example:
● Your model predicts S1-00001 matches [S2-00047, S2-00193, S3-00812]
● Ground truth says S1-00001 matches [S2-00047, S3-00812]
● Precision = 2/3, Recall = 2/2 = 1.0
● F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = 0.714
Leaderboard Information:
● Public Leaderboard: During the challenge, rankings will be based on a subset of the
test set to provide real-time feedback on your model's performance.
● Private Leaderboard: After the challenge ends, the private leaderboard will be
revealed, which uses the remaining portion of the test set for evaluation.
● Final Rankings: The final decision will be based on the private leaderboard.
You submit predictions for the full test set in both cases; the split is applied during scoring.
Submission Requirements:
1. Leaderboard (during the challenge): upload matching_results.tsv in the Portal
— tab-separated, with the exact column names described above. This is what drives the
public and private leaderboards.
2. Final submission package: submit the single zip described in Final Submission
Package above — output/ with both matching_results.tsv (final matches) and
candidate_pairs.tsv (your candidate-generation / blocking set fed to the model),
code/business_entity_resolution/ (runnable pipeline), and your methodology
document. All teams must submit it; the top teams' packages are reviewed before the
final rankings are confirmed.
3. Your methodology document must describe:
○ Methodology used
○ Candidate generation/blocking strategy
○ Model architecture and feature engineering
○ Any other relevant information about the approach
○ A template for this documentation is provided in
Documentation_template.md. There is no page limit — prioritise clarity and
technical depth over brevity.
Academic Integrity and Fair Play:
⚠ STRICTLY PROHIBITED: External Data Lookup
Participants are STRICTLY NOT ALLOWED to use external databases, APIs, or services to
look up business identities or resolve entities. This includes but is not limited to:
● Using commercial entity resolution APIs or services
● Looking up business registrations from government databases
● Using geocoding APIs to normalize addresses
● Any external data augmentation from internet sources
Enforcement:
● All submitted approaches, methodologies, and code pipelines will be thoroughly
reviewed and verified
● Any evidence of external data lookup will result in immediate disqualification
Fair Play: This challenge is designed to test your machine learning and data science skills
using only the provided training data.
Tips for Success:
● Invest in a strong blocking/candidate generation strategy — it determines the upper
bound of your recall
● Explore string similarity features (Jaccard, Levenshtein, TF-IDF cosine) for name and
address matching
● Pay attention to country-specific address patterns
● Consider the precision-recall trade-off carefully — F_0.5 rewards precision more than
recall
● Do not neglect singletons — correctly predicting "no match" is worth a full 1.0 on that
entity
● Validate your own output format against the rules above before submitting

We have 60 hours lets say to finish this. I am working in a team of 3. We want the best possible score on the metrics as possible. With hyperparameter fine tuning and multimodal approaches. If required we can plan multiple approaches and handle them together to see how it works out in the end and aim to ensemble them as well or split them. Also create an MD file of each iteration with explainations of all the decisions.

CERTAIN SECTIONS LIKE DATA CORRELATIONS WHICH REEQUIRE A HUMAN's INTUITION APPEAR DO NOT FEAR TO ASK ME TO MAKE ASSUMPTIONS AND CHANGES.

This is a record-linkage / entity-resolution problem, so let me break it down and give you a build plan you can actually execute in 60 hours.

The problem in plain terms

You have three messy lists of the same kind of thing (businesses), written by three different systems that never talk to each other. Source 1 is a clean list with no duplicates. For every business in that clean list, you need to find its "same business, different spelling" twins in Source 2 and Source 3 — where the twin might be misspelled, abbreviated, missing fields, transliterated, or referenced by a landmark instead of an address. No shared ID exists, so you can't just JOIN — you have to infer sameness from noisy text. And the scoring metric (F0.5) punishes you twice as hard for wrongly merging two different businesses as it does for missing a real match — so "confidently say no match" is worth more here than "find every possible match."

Solution alternatives, compared
Approach	Effort	Expected accuracy	Notes
Rule-based fuzzy matching (normalize + Levenshtein/Jaro-Winkler thresholds)	Low (~2-4 hrs)	Low-moderate	Good as a sanity baseline and to shake out pipeline/format bugs early. Brittle on the France test set since you can't hand-tune rules for a country you've never seen.
Blocking + engineered-feature pairwise classifier (GBDT)	Medium (~15-20 hrs)	High	The industry-standard recipe for this exact problem (this is literally how tools like Dedupe/Zingg/Magellan work). Learns feature weights instead of you guessing thresholds. Handles the F0.5 precision skew well via threshold tuning.
Fine-tuned bi-encoder / embedding matching	Medium-high (~8-12 hrs on top of the above)	Adds a few points	Sentence-transformer embeddings catch semantic/transliteration similarity that string metrics miss (e.g. "Sharma & Sons" ≈ "Sharma and Sons Enterprises"). Best used as features into the GBDT, not a standalone system.
LLM re-ranking (≤8B, Apache/MIT)	Medium (~6-10 hrs)	Adds precision on hard cases	Good at judging ambiguous borderline pairs using world knowledge of naming conventions. Too slow/costly to run on every candidate pair — use only on cases the GBDT is unsure about.
Pure end-to-end deep model (no blocking, no GBDT)	High	Not recommended	With no fixed candidate universe you'd be comparing every S1 to every S2/S3 (quadratic), and a single model has to do double duty as both a fast blocker and a precise scorer — worse on both counts than the layered approach, in this time budget.

Recommendation: a layered ensemble — multi-strategy blocking → GBDT classifier on engineered + embedding-similarity features → threshold tuned specifically for F0.5 → LLM re-ranking only on borderline pairs → consistency-aware final assignment. This is the standard high-accuracy ER architecture, it's fully explainable for your methodology write-up, and it degrades gracefully — if you run out of time, you still have a strong, submittable system after step 3.

The 60-hour pipeline

Phase 0 — Setup & EDA (hrs 0–2)

Spin up SageMaker Studio: a cheap CPU instance (ml.m5.xlarge) for EDA/feature work, and separately a GPU instance (ml.g5.xlarge or similar) you'll switch to later for embeddings/LLM.
Load with sep="\t", look at: rows per source, country breakdown, match-count distribution per S1 entity, % singletons, and eyeball 30–40 raw examples to see the actual noise patterns (abbreviations, landmark addresses, transliteration).
Do a grouped train/validation split by S1 entity, not by row, so you can score F0.5 locally before ever touching the leaderboard.
Run validate_submission.py against a dummy (all-empty) submission immediately, just to confirm you understand the exact format before building anything.

Phase 1 — Rule-based baseline, end-to-end (hrs 2–8)

Normalize: lowercase, strip punctuation, expand common abbreviations (Rd→Road, St→Street/Saint, Pvt→Private, Ltd→Limited, Corp→Corporation, & ↔ and).
Crude blocking (shared token + same country) + rapidfuzz similarity score + a threshold swept against your validation F0.5.
Get a full submission (matching_results.tsv + candidate_pairs.tsv) through the validator. This de-risks the whole 60 hours — you now have a working, submittable floor to improve on no matter what breaks later.

Phase 2 — Blocking / candidate generation (hrs 8–20)
This sets your recall ceiling, so invest here. Combine several strategies and take the union:

Inverted index on normalized name tokens (excluding legal-suffix stopwords).
Character n-gram MinHash/LSH for near-duplicates that share no exact tokens (typos).
TF-IDF (char n-gram) + nearest-neighbor search (sklearn.NearestNeighbors or FAISS/hnswlib if candidate volume is large) for top-K similar records per S1 entity.
Partition by the literal country string value present in the row — never hardcode {US, India}, since France appears only at test time and a hardcoded blocking key would silently drop it.
Measure blocking recall on your validation split (% of true matches that survive into candidates) — target ~97%+ before moving on, since anything missed here can never be recovered downstream. This candidate set is your candidate_pairs.tsv.

Phase 3 — Feature engineering + GBDT classifier (hrs 20–32) — main workhorse
Per candidate pair, compute:

Name: Levenshtein ratio, Jaro-Winkler, token-sort/set ratio, char n-gram Jaccard, TF-IDF cosine, exact-match-after-normalization flag.
Address: same metrics on address text, plus regex-extracted postal code / street-number match flags where parseable.
Country exact-match flag, length deltas, token-count deltas.
Embedding cosine similarity (pretrained multilingual sentence-transformer to start) — often your single strongest feature for transliteration/semantic noise.
Rank features: how this pair's score ranks among all candidates for that S1 entity (a locally-best match is meaningful even at a modest absolute score).

Train LightGBM/XGBoost/CatBoost on these, with the non-matching candidates as natural hard negatives, scale_pos_weight for imbalance, and threshold selection tuned directly against F0.5 (not accuracy or plain F1) on your held-out split.

Phase 4 — Fine-tuned embeddings (hrs 32–42, optional but worth it given GPU access)
Fine-tune a permissively-licensed multilingual sentence-transformer (e.g. intfloat/multilingual-e5-large — MIT, or a paraphrase-multilingual-* model — Apache 2.0) with contrastive loss (MultipleNegativesRankingLoss) on your true/false pairs. Recompute the embedding-similarity feature with this fine-tuned model and retrain the GBDT — this usually beats off-the-shelf embeddings noticeably on domain-specific noise.

Phase 5 — LLM re-ranking (hrs 42–50)
Deploy a small permissively-licensed model — I checked current licenses: Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 are Apache 2.0, Phi-3(.5)-mini is MIT — via SageMaker JumpStart or a self-hosted vLLM endpoint on your GPU instance. Apply it only to pairs where the GBDT probability sits in an ambiguous band near the threshold; give it the name/address/country pair (plus a few labeled examples from your training data — that's not external lookup, it's your provided data) and have it output match/no-match. Given F0.5's precision skew, treat the LLM as a conservative veto on borderline positives rather than a source of new positives.

Phase 6 — Assignment & threshold tuning (hrs 50–56)

Check empirically in your training ground truth whether any single S2/S3 record ever matches more than one S1 entity. If not (likely, since S1 is the deduplicated anchor), enforce it: when a S2/S3 record scores high against multiple S1 entities, keep only its best-scoring assignment — this removes a subtle false-positive source.
Do a final F0.5-maximizing threshold sweep on your validation set (consider per-country thresholds, since France has zero training examples and may need a different operating point than what US/India optimize for).
Default to "no match" when uncertain — a correctly predicted singleton is worth a full 1.0.

Phase 7 — Validate, document, package (hrs 56–60)

Run utils/validate_submission.py until it's clean.
Confirm and document your final model's exact name/version/license (must be MIT/Apache-2.0, ≤8B params) in the README — this gets audited.
Fill in Documentation_template.md: blocking recall numbers, features used, model choices, your validation F0.5, and an explicit statement that no external lookups were used.
Zip per the required structure, leave real buffer time for the leaderboard upload itself.
On DynamoDB

Given the likely record volume here (thousands, not billions of rows), I'd deprioritize it — a Parquet file on S3/local disk with pandas/DuckDB will be faster to iterate with under time pressure than standing up a DynamoDB schema for candidate pairs. It's worth using only if you want to showcase a more "productionized" architecture in the write-up, and only after the core pipeline works — don't let it eat into Phase 2/3 time.

One thing worth flagging: your 8B-parameter LLM candidates above are strong for name/address judgment calls, but double-check the exact license text on the model card for whatever checkpoint you actually download before the audit — license terms occasionally have quantization/derivative-specific caveats.


That's 15 total, not 5 — much more room. Revised cadence, roughly one submission every ~4-5 working hours, still tied to a real pipeline change:

| Day | Sub # | What changed |
|---|---|---|
| **1** | 1 | Rule-based baseline — confirms format/plumbing, gives you an anchor score, and checks your local scorer matches the leaderboard's |
| | 2 | Blocking v1 (single strategy) + threshold |
| | 3 | Blocking v2 (multi-strategy union) — recall check |
| | 4 | GBDT v1 on first feature set |
| | 5 | GBDT + full engineered feature set (day 1 best) |
| **2** | 6 | + off-the-shelf embedding similarity feature |
| | 7 | + fine-tuned embeddings |
| | 8 | Threshold retuned for F0.5 post-embedding update |
| | 9 | + S2/S3 uniqueness assignment logic |
| | 10 | Per-country threshold pass (day 2 best) |
| **3** | 11 | + LLM re-ranking on borderline pairs |
| | 12 | Re-ranking veto threshold tuned |
| | 13 | Final feature/threshold polish |
| | 14 | Near-final, doc-ready candidate |
| | 15 | Best overall config, submitted with buffer before the deadline |

Same rule as before, just less scarce now: skip a submission only when the change is fully verifiable locally (a pure format fix, an obviously-worse config).

We can update the order of this iterative appraoch.

I instead want to produce the best model possible at the preliminary mark and check how well we do and then go from there towards improvements.

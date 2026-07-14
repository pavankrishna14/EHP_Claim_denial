# Claim Denial Prediction — EHP Take-Home

Predicts whether a hospital claim will be **denied** using only information available **before** submission, so a front-end review team can catch risky claims early. It also generates short, plain-English explanations for the highest-risk claims.

## High-level approach

1. **Feature engineering** – encode the *gaps* that drive denials
   (`auth_gap`, `referral_gap`, `out_of_network`, `eligibility_unverified`, `billed_to_expected_ratio`).
   Identifier / leakage columns
   (`claim_id`, `is_denied`, `denial_reason`, `split`, `service_month`)
   are dropped.

2. **Preprocessing** – `StandardScaler` for numerics, `OneHotEncoder`
   for categoricals, passthrough for binary flags, all inside a
   `ColumnTransformer`.

3. **Splits** – the provided `split` column
   (train / validation / test) is used as-is; no custom splitting.

4. **Models** – Logistic Regression, Random Forest, and XGBoost,
   each with class imbalance handling
   (`class_weight="balanced"` / `scale_pos_weight`).
   Optional `GridSearchCV` tuning.

5. **Metric choice (driven by the business constraint)** – the review
   team can only inspect the **top 25% of claims by risk score**,
   so we optimize and report **PR-AUC** plus
   **precision/recall at the top 25%**, not raw accuracy
   (the data is ~19% denials, so accuracy is misleading).

6. **Selection** – pick the best model by
   **validation PR-AUC**, freeze its top-25% threshold on
   validation (no leakage), and confirm on the test set.
   On this data, **Logistic Regression** wins — the signal is largely
   linear and the dataset is small (~2.1K train rows), so tree
   ensembles overfit.

7. **Scoring & explanations** – score `current_claims.csv`,
   assign risk tiers, and generate LLM explanations for the top 10.

## Project structure

```text
src/
├── config.py           # column groups, constants, thresholds
├── data.py             # CSV loading + split helper
├── features.py         # domain feature engineering
├── preprocessing.py    # ColumnTransformer builder
├── models.py           # candidate models + hyperparameter grids
├── evaluate.py         # metrics (PR-AUC, precision/recall@25%) + plots
├── explain.py          # per-claim risk factors + LLM explanations
├── train.py            # CLI: train / tune / select / save model
├── evaluate_cli.py     # CLI: evaluate a saved model
└── predict.py          # CLI: score current claims + write predictions CSV

claims_history.csv
current_claims.csv
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce (exact commands)

Run all commands from the repo root so the `src` package resolves.

```bash
# 1) Train + tune all models, select the best, save it and the plots
python -m src.train \
    --data_path claims_history.csv \
    --model all \
    --tune \
    --seed 42 \
    --model_path outputs/model.pkl \
    --plots

# 2) Evaluate the saved model on the labeled history (train/val/test metrics)
python -m src.evaluate_cli \
    --model_path outputs/model.pkl \
    --data_path claims_history.csv

# 3) Score current claims + generate top-10 explanations -> predictions CSV
python -m src.predict \
    --model_path outputs/model.pkl \
    --data_path current_claims.csv \
    --output predictions_current_claims.csv
```

To train a single model without tuning:

```bash
python -m src.train \
    --data_path claims_history.csv \
    --model logreg \
    --seed 42 \
    --model_path outputs/model.pkl
```

## Output: `predictions_current_claims.csv`

Sorted by `denial_probability` descending, with columns:

| Column | Meaning |
|--------|---------|
| `claim_id` | Claim identifier |
| `denial_probability` | Predicted probability of denial (0–1) |
| `predicted_denial` | 0/1 at the chosen top-25% threshold |
| `risk_tier` | High / Medium / Low (see below) |
| `top_risk_factors` | 2–3 features most responsible for the score |
| `explanation` | Short plain-English note (top 10 claims) |

### Risk tiers

- **High** – `denial_probability >= threshold` (would be flagged in the top-25% review)
- **Medium** – `0.5 * threshold <= denial_probability < threshold`
- **Low** – `denial_probability < 0.5 * threshold`

The `threshold` is the probability cutoff that corresponds to the top-25% of validation claims by risk, frozen at training time.

## LLM explanations

`src/explain.py` builds a structured prompt per claim (features + model-derived top risk factors) and calls the OpenAI API when `OPENAI_API_KEY` is set:

```bash
export OPENAI_API_KEY=sk-...
```

If no key is present, a deterministic template generator produces equivalent plain-English output so the pipeline always runs — the prompt design is still visible in `explain.py`.

Every explanation is designed to:

- **Grounded only** in the claim's own field values and model risk drivers (no invented facts).
- Use **plain language** (no jargon).
- Include **one specific recommended action**.
- **Acknowledge it is a risk estimate**, not a guarantee of denial.


## LLM explanations

Every explanation is designed to:

- be **grounded only** in the claim's own field values and model risk drivers (no invented facts);
- use **plain language** (no jargon);
- include **one specific recommended action**;
- **acknowledge it is a risk estimate**, not a guarantee of denial;
- be **2–3 sentences** long.

### Prompt template (system)

```text
You are a claims-denial prevention assistant for a hospital revenue-cycle team.
Given a claim's risk factors, write a 2–3 sentence explanation an analyst can act
on in seconds. Rules: (1) ground every statement only in the provided field values
and risk factors – do not invent facts; (2) use plain language, no jargon;
(3) include exactly one specific recommended action; (4) make clear this is a risk
estimate, not a guarantee of denial. If the risk is low, say so plainly and note no
urgent action is needed.
```

### Example outputs

**High-risk claim** (auth gap):

> This claim has a high estimated denial risk (95%), driven mainly by prior authorization required but not on file. Recommended action: obtain and attach the prior authorization before submitting. This is a risk estimate, not a guarantee the claim will be denied.

**Low-risk claim** – behaves sensibly, no false alarms:

> This claim shows a low estimated denial risk (4%) with no dominant red flags. No urgent action is needed before submission. This is a risk estimate, not a guarantee of the final outcome.

## Interactive notebook

`claim_denial_model.ipynb` contains the full exploratory analysis (EDA, class balance, correlations, tuning comparison, and plots) behind these scripts.
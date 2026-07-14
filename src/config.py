"""Central configuration: column groups, constants, and risk thresholds."""

SEED = 42

# The review team can only inspect the top 25% of claims by risk score.
REVIEW_FRACTION = 0.25

# Target and identifier / leakage columns (never used as model inputs).
TARGET = "is_denied"
DROP_COLS = [
    "claim_id",
    "is_denied",
    "denial_reason",
    "split",
    "service_month",
]

# Feature groups used by the preprocessing pipeline.
CATEGORICAL_COLS = [
    "payer_id",
    "payer_type",
    "visit_type",
]

NUMERIC_COLS = [
    "total_billed",
    "expected_payment",
    "num_procedures",
    "num_diagnoses",
    "days_to_submit",
    "billed_to_expected_ratio",
]

BINARY_COLS = [
    "prior_auth_required",
    "has_prior_auth",
    "is_in_network",
    "missing_documentation_flag",
    "eligibility_verified",
    "referral_required",
    "referral_present",
    "auth_gap",
    "referral_gap",
    "out_of_network",
    "eligibility_unverified",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS

# Risk-tier boundaries (as a multiple of the chosen operating threshold).
# High: p >= threshold
# Medium: 0.5 * threshold <= p < threshold
# Low: p < 0.5 * threshold
MEDIUM_TIER_MULTIPLIER = 0.5
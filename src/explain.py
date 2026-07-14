"""Per-claim risk factors and LLM-generated analyst explanations."""

from __future__ import annotations

import os

# (name, condition(row) -> bool, human-readable message)
RISK_RULES = [
    (
        "auth_gap",
        lambda r: r["prior_auth_required"] == 1 and r["has_prior_auth"] == 0,
        "Prior authorization required but not on file",
    ),
    (
        "referral_gap",
        lambda r: r["referral_required"] == 1 and r["referral_present"] == 0,
        "Referral required but not present",
    ),
    (
        "missing_documentation",
        lambda r: r["missing_documentation_flag"] == 1,
        "Required documentation appears to be missing",
    ),
    (
        "eligibility_unverified",
        lambda r: r["eligibility_verified"] == 0,
        "Patient eligibility not verified",
    ),
    (
        "out_of_network",
        lambda r: r["is_in_network"] == 0,
        "Provider is out of network for this payer",
    ),
    (
        "late_submission",
        lambda r: r["days_to_submit"] > 30,
        "Claim submitted late (timely-filing risk)",
    ),
    (
        "high_billed_ratio",
        lambda r: r["billed_to_expected_ratio"] > 3,
        "Billed amount much higher than expected payment",
    ),
]


_FIXES = {
    "Prior authorization required but not on file":
        "obtain and attach the prior authorization before submitting",
    "Referral required but not present":
        "secure the referral document before submitting",
    "Required documentation appears to be missing":
        "gather the missing supporting documents",
    "Patient eligibility not verified":
        "verify the patient's coverage with the payer",
    "Provider is out of network for this payer":
        "confirm network status or route to an in-network provider",
    "Claim submitted late (timely-filing risk)":
        "expedite submission to avoid timely-filing denial",
    "Billed amount much higher than expected payment":
        "review charges against the contracted rate",
}


SYSTEM_PROMPT = (
    "You are a claims-denial prevention assistant for a hospital revenue-cycle team. "
    "Given a claim's risk factors, write a 2–3 sentence explanation an analyst can act on in "
    "seconds. Rules: (1) ground every statement ONLY in the provided field values and risk "
    "factors—do not invent facts; (2) use plain language, no jargon; (3) include exactly one "
    "specific recommended action; (4) make clear this is a risk estimate, not a guarantee of "
    "denial. If the risk is low, say so plainly and note no urgent action is needed."
)


def top_risk_factors(row, k: int = 3) -> list[str]:
    """Return up to k plain-English risk factors present for a claim row."""
    reasons = [
        msg
        for _, cond, msg in RISK_RULES
        if cond(row)
    ]
    return reasons[:k] if reasons else [
        "No dominant risk flags; elevated by feature combination"
    ]


def build_user_prompt(row) -> str:
    factors = ", ".join(row["top_risk_factors"])
    return (
        f"Claim {row['claim_id']} has a {row['denial_probability']:.0%} predicted denial risk "
        f"({row['risk_tier']} tier).\n"
        f"Payer type: {row['payer_type']}, Visit type: {row['visit_type']}.\n"
        f"Billed: ${row['total_billed']:.0f}, Days to submit: {row['days_to_submit']}.\n"
        f"Top risk factors: {factors}.\n"
        f"Write a 2–3 sentence plain-English explanation with one recommended action, and note "
        f"this is a risk estimate rather than a guaranteed denial."
    )


def template_explanation(row) -> str:
    """Deterministic fallback used when no LLM API key is available.

    Behaves sensibly across risk levels: low-risk claims get a reassuring,
    no-action message; higher-risk claims get a specific recommended fix.
    All statements are grounded in the claim's own risk factors.
    """
    prob = row["denial_probability"]
    tier = row["risk_tier"]
    primary = row["top_risk_factors"][0]

    if tier == "Low":
        return (
            f"This claim shows a low estimated denial risk ({prob:.0%}) with no dominant "
            f"red flags. No urgent action is needed before submission. "
            f"This is a risk estimate, not a guarantee of the final outcome."
        )

    action = _FIXES.get(primary, "review the claim details with the payer")

    return (
        f"This claim has a {tier.lower()} estimated denial risk ({prob:.0%}), driven mainly by: "
        f"{primary.lower()}. Recommended action: {action}. "
        f"This is a risk estimate, not a guarantee the claim will be denied."
    )


def call_llm(system: str, user: str) -> str | None:
    """Call OpenAI if OPENAI_API_KEY is set; return None to trigger the fallback."""
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from openai import OpenAI

        client = OpenAI()

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=90,
            temperature=0.3,
        )

        return resp.choices[0].message.content.strip()

    except Exception as exc:  # noqa: BLE001
        print("LLM call failed, using fallback:", exc)
        return None


def explain_row(row) -> str:
    """Return an LLM explanation if available, else a templated one."""
    out = call_llm(SYSTEM_PROMPT, build_user_prompt(row))
    return out if out else template_explanation(row)
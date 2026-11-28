"""
src/data/synthetic_meta_dataset.py
----------------------------------
Deterministic generator for the 5,000-row SYNTHETIC Meta-Dataset used as a
proof-of-concept for Agent 3 (Logistic Regression Meta-Classifier).

IMPORTANT DISCLAIMER:
- Every sample produced here is 100% SYNTHETIC (simulation only).
- Labels follow an explicit, documented synthetic moderation policy and are NOT
  human-annotated moderation ground truth.
- Evaluation results on this dataset must always be labeled as SYNTHETIC and
  never presented as real-world validation.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd

from src.data.meta_dataset import validate_meta_features_dataset
from src.models.logistic_regression import extract_meta_features_from_outputs


SYNTHETIC_DATASET_DEFAULT_PATH = "Data/processed/meta_dataset_synthetic.csv"
SYNTHETIC_POLICY_VERSION = "synthetic_policy_v1"

TARGET_LABEL_COUNTS: dict[str, int] = {
    "APPROVE": 2250,   # 45.0%
    "REVIEW": 1750,    # 35.0%
    "RESTRICT": 1000,  # 20.0%
}

# Thresholds used in the documented synthetic moderation policy
SYNTHETIC_CERTAINTY_THRESHOLD = 0.70
SYNTHETIC_RESTRICT_CONF_THRESHOLD = 0.78
SYNTHETIC_RESTRICT_INTERACTION_THRESHOLD = 0.62
SYNTHETIC_RESTRICT_RISK_THRESHOLD = 0.80
SYNTHETIC_APPROVE_RISK_MAX = 0.25


def assign_synthetic_moderation_label(
    fake_probability: float,
    bot_probability: float,
) -> str:
    """
    Documented Synthetic Moderation Labeling Policy (`synthetic_policy_v1`).

    Policy Principles:
      - Bot activity alone NEVER triggers RESTRICT or auto-APPROVE.
      - Fake news from a human-classified account escalates to REVIEW.
      - Any uncertain prediction (confidence < 0.70) escalates to REVIEW.
      - APPROVE requires both benign signals with >= 0.70 confidence and
        low weighted risk (< 0.25).
      - RESTRICT requires dual threat signals (`fake >= 0.78` AND `bot >= 0.78`),
        high threat interaction (`p_fake * p_bot >= 0.62`), and high weighted
        risk (`0.60 * p_fake + 0.40 * p_bot >= 0.80`).
    """
    p_fake = float(np.clip(fake_probability, 0.0, 1.0))
    p_bot = float(np.clip(bot_probability, 0.0, 1.0))

    fn_conf = max(p_fake, 1.0 - p_fake)
    bot_conf = max(p_bot, 1.0 - p_bot)
    min_conf = min(fn_conf, bot_conf)
    interaction = p_fake * p_bot
    weighted_risk = 0.60 * p_fake + 0.40 * p_bot

    # Rule 1: Uncertainty / borderline confidence escalates to REVIEW.
    if min_conf < SYNTHETIC_CERTAINTY_THRESHOLD:
        return "REVIEW"

    is_fake = p_fake >= 0.50
    is_bot = p_bot >= 0.50

    # Rule 2: Conflicting single-signal cases always route to REVIEW.
    if (is_bot and not is_fake) or (is_fake and not is_bot):
        return "REVIEW"

    # Rule 3: Dual benign consensus with low weighted risk -> APPROVE.
    if (not is_fake) and (not is_bot):
        if weighted_risk < SYNTHETIC_APPROVE_RISK_MAX:
            return "APPROVE"
        return "REVIEW"

    # Rule 4: Dual threat (fake & bot) -> RESTRICT only when strongly corroborated.
    if (
        min_conf >= SYNTHETIC_RESTRICT_CONF_THRESHOLD
        and interaction >= SYNTHETIC_RESTRICT_INTERACTION_THRESHOLD
        and weighted_risk >= SYNTHETIC_RESTRICT_RISK_THRESHOLD
    ):
        return "RESTRICT"

    return "REVIEW"


def _sample_matching_pairs(
    rng: np.random.Generator,
    count: int,
    sampler: Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]],
    expected_label: str,
) -> list[tuple[float, float]]:
    """Rejection-sample `(p_fake, p_bot)` pairs guaranteed to match `expected_label`."""
    collected: list[tuple[float, float]] = []
    batch_size = max(count * 3, 256)
    while len(collected) < count:
        pf_batch, pb_batch = sampler(rng, batch_size)
        for pf, pb in zip(pf_batch, pb_batch):
            pf_r = round(float(np.clip(pf, 0.001, 0.999)), 6)
            pb_r = round(float(np.clip(pb, 0.001, 0.999)), 6)
            if assign_synthetic_moderation_label(pf_r, pb_r) == expected_label:
                collected.append((pf_r, pb_r))
                if len(collected) == count:
                    break
    return collected


def generate_synthetic_meta_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a 5,000-row synthetic Meta-Dataset covering diverse, conflicting,
    uncertain, and borderline cases across APPROVE (45%), REVIEW (35%), and
    RESTRICT (20%).

    Raises ValueError if n_samples != 5000 unless exact proportional scaling matches,
    though the project specification fixes n_samples=5000.
    """
    if n_samples != 5000:
        raise ValueError(
            f"Expected n_samples=5000 for exact 2250/1750/1000 distribution, got {n_samples}."
        )

    rng = np.random.default_rng(random_state)

    # Scenario quotas sum to:
    #   APPROVE  = 1600 + 650 = 2250 (45%)
    #   REVIEW   = 450 + 500 + 500 + 300 = 1750 (35%)
    #   RESTRICT = 700 + 300 = 1000 (20%)
    scenario_specs = [
        (
            "human_real_clear",
            "APPROVE",
            1600,
            lambda r, n: (
                0.005 + 0.145 * r.beta(1.5, 4.5, size=n),
                0.005 + 0.145 * r.beta(1.5, 4.5, size=n),
            ),
        ),
        (
            "human_real_moderate",
            "APPROVE",
            650,
            lambda r, n: (
                r.uniform(0.14, 0.295, size=n),
                r.uniform(0.05, 0.295, size=n),
            ),
        ),
        (
            "bot_real_conflict",
            "REVIEW",
            450,
            lambda r, n: (
                0.01 + 0.28 * r.beta(1.6, 3.2, size=n),
                0.705 + 0.29 * r.beta(3.0, 1.5, size=n),
            ),
        ),
        (
            "human_fake_conflict",
            "REVIEW",
            500,
            lambda r, n: (
                0.705 + 0.29 * r.beta(3.0, 1.5, size=n),
                0.01 + 0.28 * r.beta(1.6, 3.2, size=n),
            ),
        ),
        (
            "low_confidence_uncertain",
            "REVIEW",
            500,
            lambda r, n: (
                # Mix of dual-uncertainty near 0.5 and single-agent uncertainty in (0.30, 0.70)
                np.where(
                    r.random(n) < 0.5,
                    r.uniform(0.31, 0.69, size=n),
                    np.where(r.random(n) < 0.5, r.uniform(0.05, 0.28, size=n), r.uniform(0.72, 0.95, size=n)),
                ),
                np.where(
                    r.random(n) < 0.7,
                    r.uniform(0.31, 0.69, size=n),
                    r.uniform(0.05, 0.95, size=n),
                ),
            ),
        ),
        (
            "borderline_moderate_dual",
            "REVIEW",
            300,
            lambda r, n: (
                np.where(r.random(n) < 0.75, r.uniform(0.70, 0.84, size=n), r.uniform(0.24, 0.298, size=n)),
                np.where(r.random(n) < 0.75, r.uniform(0.70, 0.84, size=n), r.uniform(0.24, 0.298, size=n)),
            ),
        ),
        (
            "bot_fake_dual_high",
            "RESTRICT",
            700,
            lambda r, n: (
                0.85 + 0.145 * r.beta(3.2, 1.4, size=n),
                0.85 + 0.145 * r.beta(3.2, 1.4, size=n),
            ),
        ),
        (
            "bot_fake_strong_corroborated",
            "RESTRICT",
            300,
            lambda r, n: (
                r.uniform(0.78, 0.94, size=n),
                r.uniform(0.78, 0.94, size=n),
            ),
        ),
    ]

    raw_records: list[dict] = []
    for scenario_name, expected_label, quota, sampler in scenario_specs:
        pairs = _sample_matching_pairs(rng, quota, sampler, expected_label)
        for p_fake, p_bot in pairs:
            fn_conf = max(p_fake, 1.0 - p_fake)
            bot_conf = max(p_bot, 1.0 - p_bot)
            fn_out = {
                "prediction": "fake" if p_fake >= 0.5 else "real",
                "confidence": fn_conf,
                "scores": {"fake": p_fake, "real": 1.0 - p_fake},
            }
            bot_out = {
                "prediction": "bot" if p_bot >= 0.5 else "human",
                "confidence": bot_conf,
                "scores": {"bot": p_bot, "human": 1.0 - p_bot},
            }
            feats = extract_meta_features_from_outputs(fn_out, bot_out)
            raw_records.append(
                {
                    "fake_news_prediction": fn_out["prediction"],
                    "bot_prediction": bot_out["prediction"],
                    **feats,
                    "moderation_label": expected_label,
                    "label_source": "synthetic_bootstrap",
                    "annotator_id": SYNTHETIC_POLICY_VERSION,
                    "is_synthetic": True,
                    "data_origin": "synthetic",
                    "scenario_type": scenario_name,
                }
            )

    # Shuffle deterministically and assign unique case/provenance identifiers
    perm = rng.permutation(len(raw_records))
    ordered_records = []
    for seq_idx, orig_idx in enumerate(perm, start=1):
        rec = raw_records[int(orig_idx)]
        ordered_records.append(
            {
                "case_id": f"SYN-{seq_idx:05d}",
                "source_text_id": f"syn-txt-{seq_idx:05d}",
                "source_account_id": f"syn-acc-{seq_idx:05d}",
                **rec,
            }
        )

    df = pd.DataFrame(ordered_records)
    validate_meta_features_dataset(
        df,
        require_labels=True,
        require_human_annotated=False,
        require_all_classes=True,
    )
    return df


def save_synthetic_meta_dataset(
    output_path: str = SYNTHETIC_DATASET_DEFAULT_PATH,
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate and persist the 5,000-row synthetic Meta-Dataset CSV."""
    df = generate_synthetic_meta_dataset(n_samples=n_samples, random_state=random_state)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    saved_df = save_synthetic_meta_dataset()
    counts = saved_df["moderation_label"].value_counts().to_dict()
    print(f"[SYNTHETIC] Saved {len(saved_df)} rows to '{SYNTHETIC_DATASET_DEFAULT_PATH}'. Counts={counts}")

# Agent 3 — Supervisor & Decision Engine

## 1. Architecture

Agent 3 is implemented in [`src/moderation/`](../src/moderation/) and acts as the moderation coordinator above the two independent classification agents:

- **Agent 1 (`FakeNewsDetector`)**: Evaluates content credibility (`"fake"` vs `"real"`).
- **Agent 2 (`BotDetector`)**: Evaluates account automation likelihood (`"bot"` vs `"human"`).
- **Agent 3 (`SupervisorAgent`)**: Combines both outputs **without conflating bot accounts with fake news** (a human can share fake news; an automated account can share legitimate news).

### Module Layout
- [`src/moderation/policy.py`](../src/moderation/policy.py): `SupervisorPolicy` dataclass holding configurable confidence thresholds, risk weights, and policy flags.
- [`src/moderation/decision_engine.py`](../src/moderation/decision_engine.py): `SupervisorAgent` and `evaluate_threat()`, responsible for schema validation, independent threat signal extraction, risk scoring, moderation decisions, and structured explanations.
- [`src/moderation/human_review.py`](../src/moderation/human_review.py): `ReviewQueue` and `ReviewCase` for managing cases flagged with `human_review_required=True` and recording authorized reviewer decisions separately from the AI decision.

---

## 2. Confidence Thresholds

Defined in [`SupervisorPolicy`](../src/moderation/policy.py):

| Constant / Field | Default | Meaning |
|---|---|---|
| `HIGH_CONFIDENCE` (`high_confidence`) | `0.85` | Predictions with `confidence >= 0.85` are treated as **HIGH** confidence. |
| `MEDIUM_CONFIDENCE` (`medium_confidence`) | `0.60` | Predictions with `0.60 <= confidence < 0.85` are **MEDIUM**; strictly `< 0.60` are **LOW**. |

All thresholds and weights can be overridden by passing a custom `SupervisorPolicy` to `SupervisorAgent(policy=...)`.

---

## 3. Decision-Making Logic

`SupervisorAgent.evaluate()` inspects **both** predicted labels and confidence tiers (never `risk_score` alone):

1. **Uncertainty Gate (`review_on_uncertainty=True`):**
   - If either Agent 1 or Agent 2 has `MEDIUM` or `LOW` confidence, the case is routed to **`REVIEW`** (`human_review_required=True`).
   - Uncertain predictions are **never** automatically approved (`APPROVE`) or restricted (`RESTRICT`).
2. **High-Confidence Matrix (both agents `confidence >= HIGH_CONFIDENCE`):**

| Fake News (Agent 1) | Bot Detection (Agent 2) | Decision | `human_review_required` | Policy Intent |
|---|---|---|---|---|
| `real` | `human` | `APPROVE` | `False` | Safe benign content & human account |
| `fake` | `human` | `REVIEW` | `True` | Potential misinformation by human; requires human judgment |
| `real` | `bot` | `REVIEW` | `True` | Automated account sharing real news; **bot alone never causes `RESTRICT`** |
| `fake` | `bot` | `RESTRICT` | `True` | Temporary restriction / reduced distribution pending human review |

> **Important:** `RESTRICT` represents a temporary restriction or reduced distribution pending further review — **not** permanent account deletion.

---

## 4. Risk-Scoring Formula

Each agent supplies an independent threat signal in `[0.0, 1.0]`:
- `P_signal(fake)` = `scores["fake"]` (or derived from `(prediction, confidence)` if `scores` is omitted).
- `P_signal(bot)` = `scores["bot"]` (or derived from `(prediction, confidence)` if `scores` is omitted).

With normalized weights $w_{\text{fake}} = 0.60$ and $w_{\text{bot}} = 0.40$ ($w_{\text{fake}} + w_{\text{bot}} = 1.0$):

$$\text{risk\_score} = \text{round}\Big(\text{clip}\big(100 \times (w_{\text{fake}} \cdot P_{\text{signal}}(\text{fake}) + w_{\text{bot}} \cdot P_{\text{signal}}(\text{bot})),\ 0.0,\ 100.0\big),\ 2\Big)$$

Risk level mapping (configurable):
- `LOW`: `risk_score < 35.0`
- `MEDIUM`: `35.0 <= risk_score < 70.0`
- `HIGH`: `risk_score >= 70.0`

### Risk Score Limitations
- **`risk_score` is a heuristic moderation index, not a calibrated joint probability.**
- Multinomial Naive Bayes probabilities can be overconfident near `0.0` or `1.0` due to conditional independence assumptions.
- PyTorch GRU softmax scores are not post-hoc temperature-calibrated.
- Content credibility and account automation are correlated in some campaigns, so a weighted linear sum does not model joint interactions.

---

## 5. Output Schema

```json
{
    "agent": "supervisor",
    "decision": "APPROVE | REVIEW | RESTRICT",
    "risk_level": "LOW | MEDIUM | HIGH",
    "risk_score": 0.0,
    "reasons": [
        "..."
    ],
    "human_review_required": true,
    "input_predictions": {
        "fake_news": {},
        "bot_detection": {}
    }
}
```

---

## 6. Human Review Workflow

Implemented via [`ReviewQueue`](../src/moderation/human_review.py):

1. `case = queue.submit(supervisor_output)` enqueues any case where `human_review_required` is `True` (returns `None` for `APPROVE`).
2. The `ReviewCase` preserves `ai_decision`, `risk_level`, `risk_score`, `review_reasons`, and `input_predictions`.
3. An authorized reviewer calls:
   ```python
   queue.resolve(case.case_id, human_decision="APPROVE | REJECT | RESTRICT", reviewer_id="mod_id", notes="...")
   ```
4. `human_decision`, `reviewer_id`, `reviewer_notes`, and `reviewed_at` are recorded separately without overwriting `ai_decision`.

---

## 7. Example Scenarios

1. **Legitimate News + Human Account (`real` @ `0.94`, `human` @ `0.91`)**
   - `decision`: `"APPROVE"`, `risk_level`: `"LOW"`, `risk_score`: `7.2`, `human_review_required`: `False`.
2. **Misinformation + Human Account (`fake` @ `0.95`, `human` @ `0.92`)**
   - `decision`: `"REVIEW"`, `risk_level`: `"MEDIUM"`, `risk_score`: `60.2`, `human_review_required`: `True`.
3. **Legitimate News + Automated Account (`real` @ `0.93`, `bot` @ `0.96`)**
   - `decision`: `"REVIEW"`, `risk_level`: `"MEDIUM"`, `risk_score`: `42.6`, `human_review_required`: `True`.
4. **Coordinated Bot Misinformation (`fake` @ `0.96`, `bot` @ `0.94`)**
   - `decision`: `"RESTRICT"`, `risk_level`: `"HIGH"`, `risk_score`: `95.2`, `human_review_required`: `True`.
5. **Uncertain Prediction (`fake` @ `0.54`, `bot` @ `0.95`)**
   - `decision`: `"REVIEW"`, `human_review_required`: `True` (uncertainty prevents automated `RESTRICT`).

---

## 8. Current Limitations

- **No Probability Calibration:** Upstream model confidences are uncalibrated; thresholds (`0.85` / `0.60`) should be tuned once labeled evaluation datasets and validation splits are restored.
- **In-Memory Review Queue:** `ReviewQueue` stores cases in process memory; a persistent database or backend queue is required before multi-worker deployment.
- **No Text/Account Joint Context:** Agent 3 evaluates the two model outputs independently and does not inspect cross-features such as temporal coordination across multiple accounts or URL domain reputation.
- **Not Production-Ready:** Passing unit and smoke integration tests verifies interface and policy behavior only; production deployment requires dataset-backed calibration, adversarial testing, authentication/RBAC for human reviewers, and monitoring.

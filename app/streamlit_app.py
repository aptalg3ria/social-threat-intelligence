"""
app/streamlit_app.py
--------------------
Social Threat Intelligence Platform — Streamlit Dashboard.

Integrates the frozen Agent 1 (FakeNewsDetector), frozen Agent 2 (BotDetector),
Agent 3 (synthetic-trained Logistic Regression Meta-Classifier + guardrails),
and Human Review Queue via the unified SocialThreatPipeline interface.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import logging
import os
import sys
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessing import ACCOUNT_FEATURES  # noqa: E402
from src.detection.bot_detector import (  # noqa: E402
    BotDetector,
    DEFAULT_META_PATH as BOT_META_REL,
    DEFAULT_MODEL_PATH as BOT_MODEL_REL,
    DEFAULT_SCALER_PATH as BOT_SCALER_REL,
)
from src.detection.fake_news_detector import (  # noqa: E402
    DEFAULT_MODEL_PATH as FN_MODEL_REL,
    DEFAULT_VECTORIZER_PATH as FN_VECT_REL,
    FakeNewsDetector,
)
from src.models.logistic_regression import (  # noqa: E402
    META_CLASSES,
    META_FEATURE_COLUMNS,
    MetaLogisticRegression,
)
from src.moderation.decision_engine import (  # noqa: E402
    SupervisorAgent,
    VALID_RISK_LEVELS,
)
from src.moderation.human_review import ReviewQueue, VALID_HUMAN_DECISIONS  # noqa: E402
from src.moderation.pipeline import (  # noqa: E402
    DEFAULT_SYNTHETIC_META_MODEL_PATH,
    REQUIRED_ACCOUNT_METADATA_FIELDS,
    SocialThreatPipeline,
    validate_account_metadata,
    validate_post_text,
)
from src.moderation.policy import HIGH_CONFIDENCE, MEDIUM_CONFIDENCE  # noqa: E402


PAGES: tuple[str, ...] = (
    "Overview Dashboard",
    "Analyze a Post",
    "Analysis Results",
    "Human Review Queue",
    "System Information",
)

DEMO_PRESETS: dict[str, dict[str, Any]] = {
    "Human + Real News": {
        "case_id": "DEMO-HR-001",
        "post_text": (
            "WASHINGTON (Reuters) - The U.S. Federal Reserve held benchmark interest "
            "rates steady on Wednesday following its two-day policy meeting, citing steady "
            "labor market conditions and moderating inflation data in the quarterly report."
        ),
        "screen_name": "econ_correspondent",
        "description": "Senior economics correspondent and policy analyst covering central banks.",
        "followers_count": 1250,
        "friends_count": 480,
        "statuses_count": 3400,
        "favourites_count": 2100,
        "average_tweets_per_day": 1.4,
        "account_age_days": 2400,
        "created_date": date(2018, 4, 10),
        "created_time": time(14, 20),
        "verified": True,
        "default_profile": False,
        "default_profile_image": False,
        "geo_enabled": True,
    },
    "Bot + Real News": {
        "case_id": "DEMO-BR-002",
        "post_text": (
            "WASHINGTON (Reuters) - The U.S. Federal Reserve held benchmark interest "
            "rates steady on Wednesday following its two-day policy meeting, citing steady "
            "labor market conditions and moderating inflation data in the quarterly report."
        ),
        "screen_name": "news_bot_9948271",
        "description": "",
        "followers_count": 15,
        "friends_count": 4800,
        "statuses_count": 45000,
        "favourites_count": 1,
        "average_tweets_per_day": 180.0,
        "account_age_days": 250,
        "created_date": date(2023, 11, 2),
        "created_time": time(3, 15),
        "verified": False,
        "default_profile": True,
        "default_profile_image": True,
        "geo_enabled": False,
    },
    "Human + Fake News": {
        "case_id": "DEMO-HF-003",
        "post_text": (
            "BREAKING: Secret government microchip inside tap water exposed!! "
            "Share before they delete this! Mainstream media refuses to report "
            "this Big Pharma conspiracy cure they are hiding from the public!"
        ),
        "screen_name": "citizen_observer",
        "description": "Independent writer interested in local community events and public policy.",
        "followers_count": 1250,
        "friends_count": 480,
        "statuses_count": 3400,
        "favourites_count": 2100,
        "average_tweets_per_day": 1.4,
        "account_age_days": 2400,
        "created_date": date(2018, 4, 10),
        "created_time": time(14, 20),
        "verified": True,
        "default_profile": False,
        "default_profile_image": False,
        "geo_enabled": True,
    },
    "Bot + Fake News": {
        "case_id": "DEMO-BF-004",
        "post_text": (
            "BREAKING: Secret government microchip inside tap water exposed!! "
            "Share before they delete this! Mainstream media refuses to report "
            "this Big Pharma conspiracy cure they are hiding from the public!"
        ),
        "screen_name": "news_bot_9948271",
        "description": "",
        "followers_count": 15,
        "friends_count": 4800,
        "statuses_count": 45000,
        "favourites_count": 1,
        "average_tweets_per_day": 180.0,
        "account_age_days": 250,
        "created_date": date(2023, 11, 2),
        "created_time": time(3, 15),
        "verified": False,
        "default_profile": True,
        "default_profile_image": True,
        "geo_enabled": False,
    },
}

_FORM_DEFAULTS: dict[str, Any] = {
    "form_case_id": "",
    "form_post_text": "",
    "form_screen_name": "",
    "form_description": "",
    "form_followers_count": None,
    "form_friends_count": None,
    "form_statuses_count": None,
    "form_favourites_count": None,
    "form_average_tweets_per_day": None,
    "form_account_age_days": None,
    "form_created_date": None,
    "form_created_time": None,
    "form_verified": False,
    "form_default_profile": False,
    "form_default_profile_image": False,
    "form_geo_enabled": False,
}


def _abs_path(rel_path: str) -> str:
    return os.path.join(PROJECT_ROOT, rel_path)


def _init_session_state() -> None:
    if "review_queue" not in st.session_state:
        st.session_state["review_queue"] = ReviewQueue()
    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []
    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = PAGES[0]
    for key, default_val in _FORM_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def _apply_preset(preset_name: str) -> None:
    preset = DEMO_PRESETS[preset_name]
    st.session_state["form_case_id"] = preset["case_id"]
    st.session_state["form_post_text"] = preset["post_text"]
    st.session_state["form_screen_name"] = preset["screen_name"]
    st.session_state["form_description"] = preset["description"]
    st.session_state["form_followers_count"] = preset["followers_count"]
    st.session_state["form_friends_count"] = preset["friends_count"]
    st.session_state["form_statuses_count"] = preset["statuses_count"]
    st.session_state["form_favourites_count"] = preset["favourites_count"]
    st.session_state["form_average_tweets_per_day"] = preset["average_tweets_per_day"]
    st.session_state["form_account_age_days"] = preset["account_age_days"]
    st.session_state["form_created_date"] = preset["created_date"]
    st.session_state["form_created_time"] = preset["created_time"]
    st.session_state["form_verified"] = preset["verified"]
    st.session_state["form_default_profile"] = preset["default_profile"]
    st.session_state["form_default_profile_image"] = preset["default_profile_image"]
    st.session_state["form_geo_enabled"] = preset["geo_enabled"]


def _clear_form() -> None:
    for key, default_val in _FORM_DEFAULTS.items():
        st.session_state[key] = default_val


@st.cache_resource(show_spinner=False)
def load_cached_models() -> dict[str, Any]:
    """
    Load frozen Agent 1, frozen Agent 2, and synthetic-trained Agent 3 once per server
    process, catching errors cleanly so no raw stack traces leak to the UI.
    """
    status: dict[str, Any] = {
        "ready": False,
        "fake_news_detector": None,
        "bot_detector": None,
        "meta_classifier": None,
        "rule_supervisor_available": False,
        "errors": [],
    }

    try:
        status["fake_news_detector"] = FakeNewsDetector(
            model_path=_abs_path(FN_MODEL_REL),
            vectorizer_path=_abs_path(FN_VECT_REL),
        )
    except Exception as exc:
        logger.exception("Agent 1 (Fake News) model loading failed")
        status["errors"].append(f"Agent 1 (Fake News) load failed: {exc}")

    try:
        status["bot_detector"] = BotDetector(
            model_path=_abs_path(BOT_MODEL_REL),
            scaler_path=_abs_path(BOT_SCALER_REL),
            meta_path=_abs_path(BOT_META_REL),
        )
    except Exception as exc:
        logger.exception("Agent 2 (Bot Detection) model loading failed")
        status["errors"].append(f"Agent 2 (Bot Detection) load failed: {exc}")

    try:
        status["meta_classifier"] = MetaLogisticRegression.load(
            _abs_path(DEFAULT_SYNTHETIC_META_MODEL_PATH)
        )
    except Exception as exc:
        logger.exception("Agent 3 (Decision & Moderation Agent) model loading failed")
        status["errors"].append(f"Agent 3 (Decision & Moderation Agent) load failed: {exc}")

    try:
        _ = SupervisorAgent()
        status["rule_supervisor_available"] = True
    except Exception:
        logger.exception("Rule-Based Supervisor initialization failed")
        status["rule_supervisor_available"] = False

    status["ready"] = (
        status["fake_news_detector"] is not None
        and status["bot_detector"] is not None
        and status["meta_classifier"] is not None
    )
    return status


def get_session_pipeline(models: dict[str, Any]) -> SocialThreatPipeline | None:
    """Build a SocialThreatPipeline binding cached agents to the session's ReviewQueue."""
    if not models.get("ready"):
        return None
    return SocialThreatPipeline(
        fake_news_detector=models["fake_news_detector"],
        bot_detector=models["bot_detector"],
        meta_classifier=models["meta_classifier"],
        meta_model_path=_abs_path(DEFAULT_SYNTHETIC_META_MODEL_PATH),
        review_queue=st.session_state["review_queue"],
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .sti-banner {
            padding: 0.7rem 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #d97706;
            background-color: rgba(217, 119, 6, 0.12);
            margin-bottom: 1rem;
            font-size: 0.92rem;
        }
        .sti-badge {
            display: inline-block;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 0.03em;
        }
        .sti-badge-approve {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }
        .sti-badge-review {
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
        }
        .sti-badge-restrict {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
        .sti-card {
            padding: 1rem 1.15rem;
            border-radius: 0.6rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background-color: rgba(128, 128, 128, 0.04);
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _decision_badge_html(decision: str) -> str:
    css_map = {
        "APPROVE": "sti-badge-approve",
        "REVIEW": "sti-badge-review",
        "RESTRICT": "sti-badge-restrict",
    }
    cls = css_map.get(decision, "sti-badge-review")
    return f'<span class="sti-badge {cls}">{decision}</span>'


def render_analysis_result_cards(record: dict[str, Any]) -> None:
    """Render structured result cards for Agent 1, Agent 2, Agent 3, and Queue status."""
    fn = record["fake_news"]
    bd = record["bot_detection"]
    mod = record["moderation"]
    rq = record["review_queue"]
    meta_feats = record["meta_features"]

    decision = mod["decision"]
    if decision == "APPROVE":
        st.success(f"Final Moderation Decision: **APPROVE** (Case `{record['case_id']}`)")
    elif decision == "REVIEW":
        st.warning(
            f"Final Moderation Decision: **REVIEW** — Escalated to Human Review Queue "
            f"(Case `{record['case_id']}`)"
        )
    else:
        st.error(
            f"Final Moderation Decision: **RESTRICT** — Temporary restriction flagged for "
            f"Human Review (Case `{record['case_id']}`)"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="sti-card">', unsafe_allow_html=True)
        st.subheader("Agent 1 — Fake News")
        st.metric("Prediction", fn["prediction"].upper())
        st.metric("Confidence", f"{fn['confidence']:.2%}")
        st.write(f"**Fake Probability:** `{fn['scores']['fake']:.4f}`")
        st.progress(min(max(float(fn["scores"]["fake"]), 0.0), 1.0))
        st.write(f"**Real Probability:** `{fn['scores']['real']:.4f}`")
        st.progress(min(max(float(fn["scores"]["real"]), 0.0), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sti-card">', unsafe_allow_html=True)
        st.subheader("Agent 2 — Bot Detection")
        st.metric("Prediction", bd["prediction"].upper())
        st.metric("Confidence", f"{bd['confidence']:.2%}")
        st.write(f"**Bot Probability:** `{bd['scores']['bot']:.4f}`")
        st.progress(min(max(float(bd["scores"]["bot"]), 0.0), 1.0))
        st.write(f"**Human Probability:** `{bd['scores']['human']:.4f}`")
        st.progress(min(max(float(bd["scores"]["human"]), 0.0), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="sti-card">', unsafe_allow_html=True)
        st.subheader("Agent 3 — Decision & Moderation Agent")
        st.markdown(_decision_badge_html(decision), unsafe_allow_html=True)
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Risk Score", f"{mod['risk_score']:.2f} / 100")
        m2.metric("Risk Level", mod["risk_level"])
        st.metric("Decision Confidence", f"{mod['confidence']:.2%}")
        st.write(
            f"**Guardrail Override:** `{'Yes' if mod['guardrail_applied'] else 'No'}` "
            f"| **Human Review:** `{'Required' if mod['human_review_required'] else 'Not required'}`"
        )
        if rq["queued"]:
            st.info(f"Queued for review: `{rq['case_id']}` ({rq['status']})")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Explanation, Review Reasons & Meta-Classifier Features", expanded=True):
        st.markdown("**Moderation Reasons:**")
        for reason in mod["reasons"]:
            st.write(f"- {reason}")

        c_prob, c_feat = st.columns(2)
        with c_prob:
            st.markdown("**Decision & Moderation Agent — Class Probabilities:**")
            prob_df = pd.DataFrame(
                [{"Decision": k, "Probability": v} for k, v in mod["probabilities"].items()]
            )
            st.dataframe(prob_df, hide_index=True, width="stretch")
        with c_feat:
            st.markdown("**Extracted 6 Meta-Features:**")
            feat_df = pd.DataFrame(
                [{"Feature": k, "Value": v} for k, v in meta_feats.items()]
            )
            st.dataframe(feat_df, hide_index=True, width="stretch")


def render_overview_page() -> None:
    st.title("Overview Dashboard")
    st.caption(
        "Live session moderation telemetry computed strictly from analyses executed "
        "during this session (no fabricated historical data)."
    )

    history: list[dict[str, Any]] = st.session_state["analysis_history"]
    total = len(history)
    approved = sum(1 for r in history if r["moderation"]["decision"] == "APPROVE")
    review = sum(1 for r in history if r["moderation"]["decision"] == "REVIEW")
    restrict = sum(1 for r in history if r["moderation"]["decision"] == "RESTRICT")
    bots = sum(1 for r in history if r["bot_detection"]["prediction"] == "bot")
    fakes = sum(1 for r in history if r["fake_news"]["prediction"] == "fake")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Analyzed", total)
    c2.metric("Approved", approved)
    c3.metric("Review Required", review)
    c4.metric("Restricted", restrict)
    c5.metric("Bot Detections", bots)
    c6.metric("Fake News", fakes)

    if total == 0:
        st.info(
            "No posts have been analyzed in this session yet. Go to **Analyze a Post** "
            "to submit content and account metadata."
        )
        return

    st.divider()
    ch1, ch2, ch3 = st.columns(3)

    with ch1:
        st.subheader("Moderation Decisions")
        dec_df = pd.DataFrame(
            {
                "Decision": ["APPROVE", "REVIEW", "RESTRICT"],
                "Count": [approved, review, restrict],
            }
        )
        chart = (
            alt.Chart(dec_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Decision:N", sort=["APPROVE", "REVIEW", "RESTRICT"]),
                y=alt.Y("Count:Q", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color(
                    "Decision:N",
                    scale=alt.Scale(
                        domain=["APPROVE", "REVIEW", "RESTRICT"],
                        range=["#16a34a", "#d97706", "#dc2626"],
                    ),
                    legend=None,
                ),
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")

    with ch2:
        st.subheader("Bot vs Human")
        humans = total - bots
        bot_df = pd.DataFrame(
            {"Account Class": ["Human", "Bot"], "Count": [humans, bots]}
        )
        chart = (
            alt.Chart(bot_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Account Class:N", sort=["Human", "Bot"]),
                y=alt.Y("Count:Q", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color(
                    "Account Class:N",
                    scale=alt.Scale(domain=["Human", "Bot"], range=["#2563eb", "#ea580c"]),
                    legend=None,
                ),
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")

    with ch3:
        st.subheader("Fake vs Real News")
        reals = total - fakes
        fn_df = pd.DataFrame(
            {"News Class": ["Real", "Fake"], "Count": [reals, fakes]}
        )
        chart = (
            alt.Chart(fn_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("News Class:N", sort=["Real", "Fake"]),
                y=alt.Y("Count:Q", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color(
                    "News Class:N",
                    scale=alt.Scale(domain=["Real", "Fake"], range=["#0d9488", "#e11d48"]),
                    legend=None,
                ),
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")

    st.subheader("Recent Analysis Activity")
    recent_rows = [
        {
            "Timestamp (UTC)": item["analyzed_at"],
            "Case ID": item["case_id"],
            "Agent 1 (News)": f"{item['fake_news']['prediction'].upper()} ({item['fake_news']['confidence']:.2%})",
            "Agent 2 (Account)": f"{item['bot_detection']['prediction'].upper()} ({item['bot_detection']['confidence']:.2%})",
            "Moderation Decision": item["moderation"]["decision"],
            "Risk Score": item["moderation"]["risk_score"],
            "Risk Level": item["moderation"]["risk_level"],
            "Queued": item["review_queue"]["queued"],
        }
        for item in reversed(history[-15:])
    ]
    st.dataframe(pd.DataFrame(recent_rows), hide_index=True, width="stretch")


def render_analyze_page(pipeline: SocialThreatPipeline | None, models: dict[str, Any]) -> None:
    st.title("Analyze a Post")
    st.write(
        "Provide the post text and complete Twitter/X account metadata required by "
        "`SocialThreatPipeline`. Missing values are never silently replaced with zeros."
    )

    if not models["ready"] or pipeline is None:
        st.error(
            "Pipeline models could not be loaded. Check **System Information** for details."
        )
        for err in models.get("errors", []):
            st.warning(err)
        return

    st.markdown("**Quick-Fill Test Presets (optional):**")
    p_cols = st.columns(5)
    for idx, preset_name in enumerate(DEMO_PRESETS):
        p_cols[idx].button(
            preset_name,
            on_click=_apply_preset,
            args=(preset_name,),
            width="stretch",
        )
    p_cols[4].button("Clear Form", on_click=_clear_form, width="stretch")

    with st.form("analyze_post_form"):
        st.subheader("Post Information")
        col_id, col_txt = st.columns([1, 3])
        with col_id:
            case_id_input = st.text_input(
                "Optional Post / Case Identifier",
                key="form_case_id",
                placeholder="e.g., POST-1001",
            )
        with col_txt:
            post_text = st.text_area(
                "Post Text *",
                key="form_post_text",
                height=110,
                placeholder="Paste the news headline or social media post text...",
            )

        st.subheader("Account Information (All 13 Raw Fields Required)")
        c1, c2 = st.columns(2)
        with c1:
            screen_name = st.text_input(
                "Screen Name (@handle without @) *",
                key="form_screen_name",
                placeholder="e.g., econ_correspondent",
            )
            followers_count = st.number_input(
                "Followers Count *",
                min_value=0,
                step=1,
                key="form_followers_count",
                placeholder="Enter followers count",
            )
            friends_count = st.number_input(
                "Friends (Following) Count *",
                min_value=0,
                step=1,
                key="form_friends_count",
                placeholder="Enter following count",
            )
            statuses_count = st.number_input(
                "Total Statuses (Tweets) Count *",
                min_value=0,
                step=1,
                key="form_statuses_count",
                placeholder="Enter statuses count",
            )
            favourites_count = st.number_input(
                "Favourites (Likes) Count *",
                min_value=0,
                step=1,
                key="form_favourites_count",
                placeholder="Enter favourites count",
            )

        with c2:
            average_tweets_per_day = st.number_input(
                "Average Tweets per Day *",
                min_value=0.0,
                step=0.1,
                format="%.2f",
                key="form_average_tweets_per_day",
                placeholder="Enter avg tweets/day",
            )
            account_age_days = st.number_input(
                "Account Age (Days) *",
                min_value=0,
                step=1,
                key="form_account_age_days",
                placeholder="Enter account age in days",
            )
            d_col, t_col = st.columns(2)
            created_date = d_col.date_input(
                "Account Created Date *",
                key="form_created_date",
            )
            created_time = t_col.time_input(
                "Account Created Time *",
                key="form_created_time",
            )
            description = st.text_area(
                "Profile Bio / Description * (blank string allowed if bio is empty)",
                key="form_description",
                height=78,
                placeholder="Enter account bio or leave empty if account has no bio",
            )

        b1, b2, b3, b4 = st.columns(4)
        verified = b1.checkbox("Verified Account", key="form_verified")
        default_profile = b2.checkbox("Default Profile Theme", key="form_default_profile")
        default_profile_image = b3.checkbox("Default Profile Avatar", key="form_default_profile_image")
        geo_enabled = b4.checkbox("Geolocation Enabled", key="form_geo_enabled")

        submitted = st.form_submit_button("Analyze Post", type="primary", width="stretch")

    if submitted:
        if created_date is None or created_time is None:
            st.error(
                "Please select both **Account Created Date** and **Account Created Time** "
                "before running analysis."
            )
            return

        created_at_str = f"{created_date.isoformat()} {created_time.strftime('%H:%M:%S')}"
        account_data = {
            "followers_count": followers_count,
            "friends_count": friends_count,
            "statuses_count": statuses_count,
            "favourites_count": favourites_count,
            "average_tweets_per_day": average_tweets_per_day,
            "account_age_days": account_age_days,
            "default_profile": bool(default_profile),
            "default_profile_image": bool(default_profile_image),
            "geo_enabled": bool(geo_enabled),
            "verified": bool(verified),
            "created_at": created_at_str,
            "description": description if description is not None else "",
            "screen_name": screen_name,
        }

        try:
            validate_post_text(post_text)
            validate_account_metadata(account_data)
        except (TypeError, ValueError) as exc:
            logger.exception("Analyze Post input validation failed")
            st.error(f"Input validation failed: {exc}")
            return

        base_case_id = (
            case_id_input.strip()
            if isinstance(case_id_input, str) and case_id_input.strip()
            else f"CASE-{len(st.session_state['analysis_history']) + 1:04d}"
        )
        # Ensure unique case_id across both analysis_history and ReviewQueue
        existing_ids = {item["case_id"] for item in st.session_state["analysis_history"]}
        existing_ids.update(c.case_id for c in st.session_state["review_queue"].list_cases())
        effective_case_id = base_case_id
        suffix = 2
        while effective_case_id in existing_ids:
            effective_case_id = f"{base_case_id}-{suffix}"
            suffix += 1

        with st.spinner("Running Agent 1, Agent 2, and Agent 3 moderation pipeline..."):
            try:
                result = pipeline.run(
                    text=post_text,
                    account_data=account_data,
                    case_id=effective_case_id,
                    enqueue_review=True,
                )
            except (TypeError, ValueError, FileNotFoundError, RuntimeError) as exc:
                logger.exception(
                    "Pipeline execution failed for case_id=%s", effective_case_id
                )
                st.error(f"Analysis failed: {exc}")
                return
            except Exception:
                logger.exception(
                    "Unexpected exception while running the analysis pipeline for case_id=%s",
                    effective_case_id,
                )
                st.error("An unexpected error occurred while running the analysis pipeline.")
                return

        record = {
            "case_id": effective_case_id,
            "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "post_text": post_text,
            "account_screen_name": screen_name,
            **result,
        }
        st.session_state["latest_result"] = record
        st.session_state["analysis_history"].append(record)

    if st.session_state["latest_result"] is not None:
        st.divider()
        st.subheader("Latest Pipeline Output")
        render_analysis_result_cards(st.session_state["latest_result"])


def render_results_page() -> None:
    st.title("Analysis Results")
    history: list[dict[str, Any]] = st.session_state["analysis_history"]

    if not history:
        st.info(
            "No analysis results available yet. Submit a post on **Analyze a Post** to view "
            "detailed Agent 1, Agent 2, and Agent 3 predictions."
        )
        return

    case_ids = [item["case_id"] for item in reversed(history)]
    selected_id = st.selectbox("Select Analyzed Case", options=case_ids, index=0)
    selected_record = next(item for item in reversed(history) if item["case_id"] == selected_id)

    with st.expander("Input Post Preview", expanded=False):
        st.write(f"**Case ID:** `{selected_record['case_id']}` | **Screen Name:** `@{selected_record['account_screen_name']}`")
        st.write(selected_record["post_text"])

    render_analysis_result_cards(selected_record)


def render_human_review_queue_page() -> None:
    st.title("Human Review Queue")
    flash_msg = st.session_state.pop("hrq_flash_message", None)
    if flash_msg:
        st.success(flash_msg)

    queue: ReviewQueue = st.session_state["review_queue"]
    pending = queue.get_pending()
    resolved = queue.get_resolved()

    k1, k2, k3 = st.columns(3)
    k1.metric("Pending Cases", len(pending))
    k2.metric("Resolved Cases", len(resolved))
    k3.metric("Total Queue Records", len(queue.list_cases()))

    st.subheader("Pending Moderation Cases")
    if not pending:
        st.info(
            "The Human Review Queue is currently empty. Any post flagged as **REVIEW** "
            "or temporary **RESTRICT** will appear here for manual inspection."
        )
    else:
        pending_rows = [
            {
                "Case ID": c.case_id,
                "AI Decision": c.ai_decision,
                "Risk Level": c.risk_level,
                "Risk Score": c.risk_score,
                "Agent 1 (Fake News)": (
                    f"{c.input_predictions.get('fake_news', {}).get('prediction', 'n/a').upper()} "
                    f"({c.input_predictions.get('fake_news', {}).get('confidence', 0.0):.2%})"
                ),
                "Agent 2 (Bot Detection)": (
                    f"{c.input_predictions.get('bot_detection', {}).get('prediction', 'n/a').upper()} "
                    f"({c.input_predictions.get('bot_detection', {}).get('confidence', 0.0):.2%})"
                ),
                "Review Reasons": " | ".join(c.review_reasons),
            }
            for c in pending
        ]
        st.dataframe(pd.DataFrame(pending_rows), hide_index=True, width="stretch")

        st.divider()
        st.subheader("Inspect & Resolve Pending Case")
        selected_cid = st.selectbox(
            "Select Pending Case ID",
            options=[c.case_id for c in pending],
            key="hrq_selected_case_id",
        )
        case = queue.get_case(selected_cid)

        st.markdown(
            f"**Original AI Decision (Preserved):** {_decision_badge_html(case.ai_decision)} "
            f"&nbsp;&nbsp;|&nbsp;&nbsp; **Risk Level:** `{case.risk_level}` "
            f"&nbsp;&nbsp;|&nbsp;&nbsp; **Risk Score:** `{case.risk_score:.2f}/100`",
            unsafe_allow_html=True,
        )

        fn_pred = case.input_predictions.get("fake_news", {})
        bd_pred = case.input_predictions.get("bot_detection", {})

        c_a1, c_a2 = st.columns(2)
        with c_a1:
            st.markdown("**Agent 1 Output (Fake News Detection):**")
            st.json(fn_pred)
        with c_a2:
            st.markdown("**Agent 2 Output (Bot Detection):**")
            st.json(bd_pred)

        st.markdown("**Review Reasons:**")
        for r in case.review_reasons:
            st.write(f"- {r}")

        with st.form("resolve_review_form"):
            r1, r2 = st.columns(2)
            human_decision = r1.selectbox(
                "Human Reviewer Decision",
                options=list(VALID_HUMAN_DECISIONS),
                index=0,
            )
            reviewer_id = r2.text_input(
                "Reviewer ID *",
                placeholder="e.g., mod_analyst_01",
            )
            reviewer_notes = st.text_area(
                "Reviewer Notes",
                placeholder="Document justification for your final moderation disposition...",
            )
            submit_review = st.form_submit_button(
                "Submit Human Decision", type="primary", width="stretch"
            )

        if submit_review:
            try:
                resolved_case = queue.resolve(
                    case_id=selected_cid,
                    human_decision=human_decision,
                    reviewer_id=reviewer_id,
                    notes=reviewer_notes,
                )
                st.session_state["hrq_flash_message"] = (
                    f"Case `{resolved_case.case_id}` resolved as **{resolved_case.human_decision}** "
                    f"by `{resolved_case.reviewer_id}` (original AI decision `{resolved_case.ai_decision}` preserved)."
                )
                st.rerun()
            except (KeyError, ValueError) as exc:
                logger.exception("Failed to resolve review case %s", selected_cid)
                st.error(f"Could not resolve review case: {exc}")
            except Exception:
                logger.exception("Unexpected error resolving review case %s", selected_cid)
                st.error("An unexpected error occurred while resolving the review case.")

    if resolved:
        st.divider()
        st.subheader("Resolved Cases History")
        resolved_rows = [
            {
                "Case ID": c.case_id,
                "Original AI Decision": c.ai_decision,
                "Human Decision": c.human_decision,
                "Reviewer ID": c.reviewer_id,
                "Risk Score": c.risk_score,
                "Notes": c.reviewer_notes,
                "Reviewed At": c.reviewed_at,
            }
            for c in reversed(resolved)
        ]
        st.dataframe(pd.DataFrame(resolved_rows), hide_index=True, width="stretch")


def render_system_info_page(models: dict[str, Any]) -> None:
    st.title("System Information")
    st.caption(
        "Runtime architecture, model loading status, available feature specifications, "
        "moderation classes, and Human Review Queue status."
    )

    queue: ReviewQueue = st.session_state["review_queue"]
    loaded_agents = sum(
        1 for key in ("fake_news_detector", "bot_detector", "meta_classifier")
        if models.get(key) is not None
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Pipeline Status", "Operational" if models.get("ready") else "Unavailable")
    s2.metric("Agents Loaded", f"{loaded_agents} / 3")
    s3.metric("Moderation Classes", ", ".join(META_CLASSES))
    s4.metric(
        "Human Review Queue",
        f"{len(queue.get_pending())} Pending / {len(queue.get_resolved())} Resolved",
    )

    st.divider()
    st.subheader("Agent & Pipeline Status")
    components = [
        {
            "Component": "Agent 1 — Fake News Detection",
            "Role & Architecture": "Multinomial Naive Bayes + TF-IDF (max_features=5000, n-grams 1-2)",
            "Status": "Loaded & Active" if models.get("fake_news_detector") is not None else "Load Error",
        },
        {
            "Component": "Agent 2 — Bot Detection",
            "Role & Architecture": "PyTorch GRU/RNN (seq_len=5, input_size=3, 15 features) + StandardScaler",
            "Status": "Loaded & Active" if models.get("bot_detector") is not None else "Load Error",
        },
        {
            "Component": "Agent 3 — Decision & Moderation Agent",
            "Role & Architecture": "Logistic Regression Meta-Classifier (6 meta-features) + Safety Guardrails",
            "Status": "Loaded & Active" if models.get("meta_classifier") is not None else "Load Error",
        },
        {
            "Component": "End-to-End Pipeline",
            "Role & Architecture": "SocialThreatPipeline (Input Validation -> Agents 1 & 2 -> Agent 3 -> Queue)",
            "Status": "Ready" if models.get("ready") else "Degraded",
        },
        {
            "Component": "Rule-Based Supervisor (Baseline)",
            "Role & Architecture": f"SupervisorAgent (HIGH>={HIGH_CONFIDENCE}, MEDIUM>={MEDIUM_CONFIDENCE})",
            "Status": "Available" if models.get("rule_supervisor_available") else "Unavailable",
        },
        {
            "Component": "Human Review Queue",
            "Role & Architecture": "ReviewQueue (preserves AI decision separately from human disposition)",
            "Status": f"Active ({len(queue.get_pending())} pending / {len(queue.get_resolved())} resolved)",
        },
    ]
    st.dataframe(pd.DataFrame(components), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Moderation Decision Classes & Available Features")
    col_cls, col_feat = st.columns(2)
    with col_cls:
        st.markdown("**Moderation Decision Classes**")
        class_rows = [
            {"Class": "APPROVE", "Meaning": "Benign content and human account with high confidence."},
            {"Class": "REVIEW", "Meaning": "Single-signal threat, conflicting signals, or uncertain predictions routed for manual review."},
            {"Class": "RESTRICT", "Meaning": "High-confidence misinformation combined with bot activity, flagged for temporary restriction pending review."},
        ]
        st.dataframe(pd.DataFrame(class_rows), hide_index=True, width="stretch")
        st.write(
            f"**Risk Levels:** `{', '.join(VALID_RISK_LEVELS)}` &nbsp;|&nbsp; "
            f"**Human Reviewer Decisions:** `{', '.join(VALID_HUMAN_DECISIONS)}`"
        )

    with col_feat:
        st.markdown("**Agent 3 — Decision & Moderation Agent Features (6)**")
        st.dataframe(
            pd.DataFrame({"Meta-Feature": list(META_FEATURE_COLUMNS)}),
            hide_index=True,
            width="stretch",
        )

    with st.expander("Available Raw & Engineered Agent 2 Features", expanded=False):
        c_raw, c_eng = st.columns(2)
        with c_raw:
            st.markdown(f"**Required Raw Account Fields ({len(REQUIRED_ACCOUNT_METADATA_FIELDS)})**")
            st.dataframe(
                pd.DataFrame({"Raw Field": list(REQUIRED_ACCOUNT_METADATA_FIELDS)}),
                hide_index=True,
                width="stretch",
            )
        with c_eng:
            st.markdown(f"**Engineered Bot Detection Features ({len(ACCOUNT_FEATURES)})**")
            st.dataframe(
                pd.DataFrame({"Feature": list(ACCOUNT_FEATURES)}),
                hide_index=True,
                width="stretch",
            )

    if models.get("errors"):
        st.subheader("Reported Loading Warnings / Errors")
        for err in models["errors"]:
            st.error(err)


def main() -> None:
    st.set_page_config(
        page_title="Social Threat Intelligence Platform",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _init_session_state()

    models = load_cached_models()
    pipeline = get_session_pipeline(models)
    queue: ReviewQueue = st.session_state["review_queue"]
    pending_count = len(queue.get_pending())
    history_count = len(st.session_state["analysis_history"])

    with st.sidebar:
        st.title("🛡️ Social Threat Intelligence")
        st.caption("Multi-Agent Moderation Platform")

        selected_page = st.radio(
            "Navigation",
            options=list(PAGES),
            key="active_page",
        )

        st.divider()
        st.metric("Session Posts Analyzed", history_count)
        st.metric("Pending Human Reviews", pending_count)

    if selected_page == "Overview Dashboard":
        render_overview_page()
    elif selected_page == "Analyze a Post":
        render_analyze_page(pipeline, models)
    elif selected_page == "Analysis Results":
        render_results_page()
    elif selected_page == "Human Review Queue":
        render_human_review_queue_page()
    elif selected_page == "System Information":
        render_system_info_page(models)


if __name__ == "__main__":
    main()

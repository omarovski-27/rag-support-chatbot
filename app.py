"""
app.py — Streamlit chat UI for the RAG Support Chatbot.

Run:
    C:\\dev\\apg-rag-venv\\Scripts\\streamlit.exe run app.py

What this file wires together
------------------------------
  memory.py  → ask_with_memory()     multi-turn RAG + history
  tools.py   → get_tracking_status() parcel lookup
               escalate_to_human()   human handoff
               detect_sentiment()    per-turn mood
  logger.py  → ConversationLogger    analytics + CSAT

Session lifecycle
-----------------
  1. User opens the app → new session_id + ConversationLogger created
  2. User types a message → sentiment detected, chain invoked, turn logged
  3. If answer contains a tracking lookup trigger → tool called, card shown
  4. If Claude calls escalate_to_human → handoff card shown, input disabled
  5. User clicks End Chat → CSAT widget shown, close() called, history cleared
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import re
import uuid

import streamlit as st
from dotenv import load_dotenv

from src.logger import ConversationLogger, timer
import time
from src.memory import clear_session, stream_with_memory
from src.retriever import get_reranking_retriever
from src.tools import (
    ESCALATION_TOOL,
    TRACKING_TOOL,
    detect_sentiment,
    escalate_to_human,
    get_tracking_status,
)

load_dotenv()


@st.cache_resource
def _load_retriever():
    """Build retriever once per process; survives Streamlit reruns and hot-reloads."""
    return get_reranking_retriever()


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Support Chatbot",
    page_icon="📦",
    layout="centered",
)

# Warm the retriever on first load; @st.cache_resource returns instantly thereafter.
_retriever = _load_retriever()

# ── Session state bootstrap ──────────────────────────────────────────────────
# st.session_state persists across Streamlit reruns within the same browser tab.

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "logger" not in st.session_state:
    st.session_state.logger = ConversationLogger()
    # Keep session_id in sync with logger
    st.session_state.session_id = st.session_state.logger.session_id

if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant"|"tool", "content": str, "meta": dict}
    st.session_state.messages = []

if "escalated" not in st.session_state:
    st.session_state.escalated = False

if "session_closed" not in st.session_state:
    st.session_state.session_closed = False

if "show_csat" not in st.session_state:
    st.session_state.show_csat = False

# ── Helper: detect if user is asking for tracking ────────────────────────────

_TRACKING_RE = re.compile(
    r"\b([A-Z]{2,4}[\-]?\d{8,20}|\d{10,20})\b"
)

def _extract_tracking_number(text: str) -> str | None:
    """Return the first plausible tracking number in *text*, or None."""
    match = _TRACKING_RE.search(text)
    return match.group(0) if match else None

_TRACKING_KEYWORDS = {"track", "tracking", "parcel", "shipment", "where is my", "locate"}

def _looks_like_tracking_query(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _TRACKING_KEYWORDS)

# ── Helper: detect escalation intent ────────────────────────────────────────

_ESCALATION_RE = re.compile(
    r"\b(human|agent|person|supervisor|manager|representative|rep|speak to someone|"
    r"talk to someone|real person|live agent|escalate)\b",
    re.IGNORECASE,
)

def _looks_like_escalation(text: str) -> bool:
    return bool(_ESCALATION_RE.search(text))

# ── Render helpers ────────────────────────────────────────────────────────────

_SENTIMENT_EMOJI = {"positive": "😊", "frustrated": "😤", "angry": "😠"}


def _render_tracking_card(tracking_number: str, result: dict) -> None:
    status = result.get("status", "Unknown")
    carrier = result.get("carrier", "")
    eta = result.get("estimated_delivery") or "N/A"
    duty = result.get("requires_duty_payment", False)

    color = "🔴" if duty else ("🟢" if status == "Delivered" else "🟡")
    st.markdown(f"**{color} Tracking: `{tracking_number}`**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status)
    col2.metric("Carrier", carrier)
    col3.metric("Est. Delivery", eta)
    if duty:
        st.warning("⚠ Duty payment required to release this shipment.")
    st.caption(result.get("message", ""))


def _render_escalation_card(result: dict) -> None:
    st.error("🙋 **Connecting you to a human agent**")
    st.markdown(result.get("handoff_message", ""))
    st.info(f"Reason: {result.get('reason', '')}")


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("📦 RAG Support Chatbot")
st.caption("Customer support powered by Local RAG · PoC")

# Sidebar — session info for demo/debug visibility
with st.sidebar:
    st.markdown("### Session Info")
    st.code(st.session_state.session_id[:8] + "...", language=None)
    st.markdown(f"**Turns:** {st.session_state.logger.turn_number}")
    if st.session_state.escalated:
        st.error("⚠ Escalated to human")
    if st.session_state.session_closed:
        st.success("✓ Session closed")

    st.divider()
    st.markdown("**Quick test tracking numbers:**")
    st.code("1234567890")       # → Held at Customs
    st.code("APG-99887766")     # → Delivered
    st.code("JD014600006281471990")  # → Held at Customs

# ── Render existing messages ──────────────────────────────────────────────────

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    meta = msg.get("meta", {})

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
            if meta.get("sentiment") and meta["sentiment"] != "neutral":
                _SENTIMENT_EMOJI = {"positive": "😊", "frustrated": "😤", "angry": "😠"}
                st.caption(f"Sentiment: {_SENTIMENT_EMOJI.get(meta['sentiment'], '')} {meta['sentiment']}")

    elif role == "assistant":
        with st.chat_message("assistant", avatar="📦"):
            st.markdown(content)
            if meta.get("latency_ms"):
                st.caption(f"⚡ {meta['latency_ms']:.0f}ms · {meta.get('chunks', 0)} chunks from: {', '.join(meta.get('topics', []))}")

    elif role == "tool_tracking":
        with st.chat_message("assistant", avatar="📦"):
            _render_tracking_card(content, meta)

    elif role == "tool_escalation":
        with st.chat_message("assistant", avatar="📦"):
            _render_escalation_card(meta)

# ── CSAT widget ───────────────────────────────────────────────────────────────

def _show_csat_and_close(outcome: str) -> None:
    st.divider()
    st.markdown("### How did we do?")
    rating = st.feedback("stars", key="csat_widget")
    if rating is not None:
        # st.feedback returns 0-4, convert to 1-5
        stars = rating + 1
        st.session_state.logger.close(outcome=outcome, csat_rating=stars)
        clear_session(st.session_state.session_id)
        st.session_state.session_closed = True
        st.session_state.show_csat = False
        st.success(f"Thank you for your {stars}★ rating! Chat session saved.")
        st.rerun()

# ── End Chat button ───────────────────────────────────────────────────────────

if not st.session_state.session_closed and st.session_state.messages:
    if st.button("End Chat", type="secondary"):
        st.session_state.show_csat = True

if st.session_state.show_csat and not st.session_state.session_closed:
    outcome = "escalated" if st.session_state.escalated else "resolved"
    _show_csat_and_close(outcome)

# ── Chat input ────────────────────────────────────────────────────────────────

input_disabled = st.session_state.escalated or st.session_state.session_closed
placeholder = (
    "Session ended." if st.session_state.session_closed
    else "A human agent will be with you shortly." if st.session_state.escalated
    else "Type your question here…"
)

if user_input := st.chat_input(placeholder, disabled=input_disabled):

    sentiment = detect_sentiment(user_input)

    # ── Render user message immediately ──────────────────────────────────
    with st.chat_message("user"):
        st.markdown(user_input)
        if sentiment != "neutral":
            st.caption(f"Sentiment: {_SENTIMENT_EMOJI.get(sentiment, '')} {sentiment}")

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "meta": {"sentiment": sentiment},
    })

    # ── Check for explicit escalation request ────────────────────────────
    if _looks_like_escalation(user_input) or sentiment == "angry":
        reason = (
            "Customer requested a human agent."
            if _looks_like_escalation(user_input)
            else "Customer sentiment detected as angry."
        )
        result = escalate_to_human(reason=reason, customer_sentiment=sentiment)

        with st.chat_message("assistant", avatar="📦"):
            _render_escalation_card(result)

        st.session_state.messages.append({
            "role": "tool_escalation",
            "content": "",
            "meta": result,
        })
        st.session_state.escalated = True

        st.session_state.logger.log_turn(
            user_input, [], result["handoff_message"],
            tool_called="escalate_to_human",
            sentiment=sentiment,
        )
        st.rerun()

    # ── Check for tracking number in the message ──────────────────────────
    tracking_number = _extract_tracking_number(user_input)
    if tracking_number or _looks_like_tracking_query(user_input):
        if not tracking_number:
            # Ask Claude + also show a prompt for the tracking number
            pass  # fall through to RAG answer; no number to look up yet
        else:
            track_result = get_tracking_status(tracking_number)
            with st.chat_message("assistant", avatar="📦"):
                _render_tracking_card(tracking_number, track_result)
            st.session_state.messages.append({
                "role": "tool_tracking",
                "content": tracking_number,
                "meta": track_result,
            })

    # ── RAG answer (streamed) ─────────────────────────────────────────────
    with st.chat_message("assistant", avatar="📦"):
        with st.spinner("Searching knowledge base…"):
            # Retrieval is blocking and completes inside this spinner block.
            # stream_with_memory returns a token generator; LLM streaming starts below.
            # usage dict is empty until the generator is fully consumed.
            token_iter, docs, t0, usage = stream_with_memory(
                user_input,
                session_id=st.session_state.session_id,
                retriever=_retriever,
            )
        response_placeholder = st.empty()
        full_response = ""
        for token in token_iter:
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        answer = full_response
        latency = (time.perf_counter() - t0) * 1000
        topics = list({d.metadata.get("topic", "?") for d in docs})
        st.caption(f"⚡ {latency:.0f}ms · {len(docs)} chunks from: {', '.join(topics)}")

    # usage dict is now populated (generator fully consumed above)
    input_tokens  = usage.get("input_tokens",  0)
    output_tokens = usage.get("output_tokens", 0)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": {"latency_ms": latency, "chunks": len(docs), "topics": topics},
    })

    st.session_state.logger.log_turn(
        user_input,
        docs,
        answer,
        latency_ms=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        sentiment=sentiment,
    )


import html
import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, rag_pipeline  # noqa: E402

load_dotenv()

st.set_page_config(
    page_title="Atlas - Knowledge Assistant",
    page_icon="A",
    layout="wide",
)


# --- Styling ---------------------------------------------------------------
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root{
            --ink:#1A1F2E; --muted:#6B7280; --primary:#2563EB; --navy:#1E3A5F;
            --success:#059669; --bg:#F7F8FA; --card:#FFFFFF; --border:#E6E8EC;
        }

        html, body, .stApp, [class*="css"]{
            font-family:'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        }
        .stApp{ background:var(--bg); }

        /* Tighten chrome for an app-like feel */
        [data-testid="stToolbar"]{ display:none; }
        [data-testid="stDecoration"]{ display:none; }
        [data-testid="stHeader"]{ background:transparent; height:0; }
        #MainMenu, footer{ visibility:hidden; }
        .block-container{ padding-top:1.6rem; padding-bottom:3rem; max-width:980px; }

        /* App bar */
        .rag-header{
            display:flex; align-items:center; justify-content:space-between;
            padding:6px 0 20px; border-bottom:1px solid var(--border); margin-bottom:6px;
        }
        .rag-brand{ display:flex; align-items:center; gap:14px; }
        .rag-logo{
            width:44px; height:44px; border-radius:11px;
            background:linear-gradient(135deg,var(--navy),var(--primary));
            color:#fff; display:flex; align-items:center; justify-content:center;
            font-weight:800; font-size:21px; letter-spacing:-0.5px;
            box-shadow:0 4px 12px rgba(37,99,235,.25);
        }
        .rag-brand-text{ display:flex; flex-direction:column; line-height:1.2; }
        .rag-product{ font-weight:700; font-size:18px; color:var(--ink); letter-spacing:-0.2px; }
        .rag-sub{ font-size:12.5px; color:var(--muted); font-weight:500; }
        .rag-env{
            display:flex; align-items:center; gap:8px; font-size:12.5px; font-weight:500;
            color:var(--muted); background:var(--card); border:1px solid var(--border);
            padding:7px 13px; border-radius:999px;
        }
        .rag-dot{ width:8px; height:8px; border-radius:50%; background:var(--success);
            display:inline-block; box-shadow:0 0 0 3px rgba(5,150,105,.15); }

        /* Eyebrow labels */
        .rag-eyebrow{ font-size:11px; font-weight:700; letter-spacing:1.5px;
            text-transform:uppercase; color:var(--muted); margin:0 0 9px; }

        /* Lead text */
        .rag-lead{ font-size:15px; color:var(--muted); line-height:1.6; margin:14px 0 18px; }

        /* Answer card (signature element) */
        .rag-answer-card{
            background:var(--card); border:1px solid var(--border);
            border-left:3px solid var(--primary); border-radius:14px;
            padding:22px 24px; margin-top:6px;
            box-shadow:0 1px 2px rgba(16,24,40,.04), 0 6px 20px rgba(16,24,40,.06);
        }
        .rag-answer-text{ font-size:16px; line-height:1.62; color:var(--ink); }
        .rag-answer-meta{ display:flex; align-items:center; gap:10px; margin-top:18px; flex-wrap:wrap; }
        .rag-badge{ display:inline-flex; align-items:center; gap:7px; font-size:12.5px;
            font-weight:600; padding:5px 12px; border-radius:999px; }
        .rag-badge-grounded{ background:rgba(5,150,105,.10); color:var(--success); }
        .rag-pill{ display:inline-flex; align-items:center; gap:6px; font-size:12.5px;
            font-weight:600; color:var(--muted); background:var(--bg);
            border:1px solid var(--border); padding:5px 12px; border-radius:999px; }

        /* Source cards */
        .rag-source-card{ background:var(--card); border:1px solid var(--border);
            border-radius:12px; padding:15px 16px; margin-bottom:14px; }
        .rag-source-head{ display:flex; align-items:center; gap:9px; margin-bottom:10px; }
        .rag-source-tag{ font-size:10.5px; font-weight:700; letter-spacing:.5px;
            color:var(--primary); background:rgba(37,99,235,.08);
            padding:3px 8px; border-radius:6px; text-transform:uppercase; }
        .rag-source-name{ font-weight:600; font-size:13.5px; color:var(--navy); }
        .rag-source-body{ font-size:13px; line-height:1.55; color:var(--muted);
            max-height:150px; overflow:auto; white-space:pre-wrap; }

        /* Empty state */
        .rag-empty{ border:1px dashed #D4D8DF; border-radius:14px; padding:24px 26px;
            background:var(--card); color:var(--muted); font-size:14px; line-height:1.65; }
        .rag-empty strong{ color:var(--ink); }

        /* Metric cards */
        .rag-metric{ background:var(--card); border:1px solid var(--border);
            border-radius:12px; padding:18px 20px; }
        .rag-metric-val{ font-size:28px; font-weight:800; color:var(--ink); letter-spacing:-1px; }
        .rag-metric-val span{ font-size:15px; font-weight:600; color:var(--muted); }
        .rag-metric-label{ font-size:11px; font-weight:600; color:var(--muted);
            text-transform:uppercase; letter-spacing:.7px; margin-top:5px; }

        /* Widgets */
        .stButton>button{ border-radius:9px; font-weight:600; }
        .stTextInput input{ border-radius:9px; }
        .stTabs [data-baseweb="tab-list"]{ gap:4px; }
        .stTabs [data-baseweb="tab"]{ font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    connected = bool(os.getenv("OPENAI_API_KEY"))
    status = (
        f'<span class="rag-dot"></span>Connected &middot; {html.escape(config.CHAT_MODEL)}'
        if connected
        else '<span class="rag-dot" style="background:#DC2626;box-shadow:0 0 0 3px rgba(220,38,38,.15)"></span>No API key'
    )
    st.markdown(
        f"""
        <div class="rag-header">
            <div class="rag-brand">
                <div class="rag-logo">A</div>
                <div class="rag-brand-text">
                    <div class="rag-product">Atlas</div>
                    <div class="rag-sub">Acme Corp &middot; Knowledge Assistant</div>
                </div>
            </div>
            <div class="rag-env">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _nl2html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def render_answer(answer: str, n_sources: int, latency: float) -> None:
    st.markdown(
        f"""
        <div class="rag-answer-card">
            <div class="rag-eyebrow">Answer</div>
            <div class="rag-answer-text">{_nl2html(answer)}</div>
            <div class="rag-answer-meta">
                <span class="rag-badge rag-badge-grounded">&#9679; Grounded in {n_sources} source{'s' if n_sources != 1 else ''}</span>
                <span class="rag-pill">&#9201; {latency:.2f}s</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source(doc) -> None:
    name = html.escape(doc.metadata.get("source", "unknown"))
    body = _nl2html(doc.page_content)
    st.markdown(
        f"""
        <div class="rag-source-card">
            <div class="rag-source-head">
                <span class="rag-source-tag">Source</span>
                <span class="rag-source-name">{name}</span>
            </div>
            <div class="rag-source-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(value, label: str, unit: str = "") -> str:
    unit_html = f"<span>{unit}</span>" if unit else ""
    return (
        f'<div class="rag-metric"><div class="rag-metric-val">{value}{unit_html}</div>'
        f'<div class="rag-metric-label">{label}</div></div>'
    )


# --- Cached resources ------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return rag_pipeline.load_index()


# --- App -------------------------------------------------------------------
inject_styles()
render_header()

# Sidebar: system status / configuration
with st.sidebar:
    st.markdown('<div class="rag-eyebrow">System</div>', unsafe_allow_html=True)
    st.write(f"**Chat model**  \n{config.CHAT_MODEL}")
    st.write(f"**Embeddings**  \n{config.EMBEDDING_MODEL}")
    st.write(f"**Sources per answer**  \n{config.TOP_K}")
    st.divider()
    index_exists = config.VECTORSTORE_DIR.exists()
    st.write(f"**Index**  \n{'Ready' if index_exists else 'Not built'}")
    if not index_exists:
        st.warning("Build the index: `python scripts/build_index.py`")
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Add OPENAI_API_KEY to a .env file to enable answers.")
    st.caption("Reference implementation — LangChain + GPT-4 family.")

tab_assistant, tab_eval = st.tabs(["Assistant", "Evaluation"])

with tab_assistant:
    st.markdown(
        '<p class="rag-lead">Ask about company policies and internal documentation. '
        "Every answer is generated only from indexed sources and shows where it came from.</p>",
        unsafe_allow_html=True,
    )

    question = st.text_input(
        "Your question",
        placeholder="e.g. How many vacation days do I get?",
        label_visibility="collapsed",
    )
    ask = st.button("Ask", type="primary")

    # Sample question chips — also serve as the empty-state invitation.
    samples = [
        "How many vacation days do I get?",
        "What is the password policy?",
        "How often is analytics data refreshed?",
    ]
    chip_cols = st.columns(len(samples))
    chip_clicked = None
    for col, sample in zip(chip_cols, samples):
        if col.button(sample, use_container_width=True):
            chip_clicked = sample

    active_q = chip_clicked or (question.strip() if (ask and question.strip()) else None)

    if active_q:
        if not config.VECTORSTORE_DIR.exists():
            st.error("No index found. Run `python scripts/build_index.py` first.")
        else:
            try:
                with st.spinner("Retrieving sources and generating an answer..."):
                    answer, docs, latency = rag_pipeline.answer_question(
                        active_q, get_vectorstore()
                    )
                render_answer(answer, len(docs), latency)
                st.markdown(
                    '<div class="rag-eyebrow" style="margin-top:26px">Sources</div>',
                    unsafe_allow_html=True,
                )
                col_l, col_r = st.columns(2)
                for i, d in enumerate(docs):
                    with (col_l if i % 2 == 0 else col_r):
                        render_source(d)
            except Exception as exc:
                st.error(f"Couldn't generate an answer: {exc}")
    else:
        st.markdown(
            '<div class="rag-empty">Start with a question above, or pick a sample. '
            "Atlas answers only from the indexed documents — try an off-topic question "
            "(for example, <strong>“What is the CEO's salary?”</strong>) to see it decline "
            "rather than guess.</div>",
            unsafe_allow_html=True,
        )

with tab_eval:
    st.markdown(
        '<p class="rag-lead">Quality, latency, and consistency measured on a synthetic '
        "Q&amp;A set generated from the documents. The documents act as ground truth.</p>",
        unsafe_allow_html=True,
    )

    if config.EVAL_RESULTS_PATH.exists():
        results = json.loads(config.EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        summary = results.get("summary", {})
        cards = [
            render_metric(summary.get("n_questions", "-"), "Questions"),
            render_metric(summary.get("avg_quality_1to5", "-"), "Avg quality", " / 5"),
            render_metric(summary.get("avg_latency_seconds", "-"), "Avg latency", "s"),
            render_metric(
                int(summary.get("consistency_rate", 0) * 100)
                if isinstance(summary.get("consistency_rate"), (int, float))
                else "-",
                "Consistency",
                "%",
            ),
        ]
        cols = st.columns(4)
        for col, card in zip(cols, cards):
            col.markdown(card, unsafe_allow_html=True)

        st.markdown(
            '<div class="rag-eyebrow" style="margin-top:26px">Per-question detail</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(results.get("items", []), use_container_width=True)
    else:
        st.markdown(
            '<div class="rag-empty">No evaluation results yet. Generate them with:<br><br>'
            "<code>python -m src.synthetic_qa</code><br><code>python -m src.evaluate</code></div>",
            unsafe_allow_html=True,
        )

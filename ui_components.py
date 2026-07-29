"""Shared presentation primitives for the Streamlit application."""

from __future__ import annotations

import html

import streamlit as st


GLOBAL_STYLES = """
<style>
:root {
  --surface-0: #020617;
  --surface-1: #0f172a;
  --surface-2: #162033;
  --border: #25324a;
  --text: #f8fafc;
  --muted: #94a3b8;
  --accent: #38bdf8;
  --accent-strong: #0ea5e9;
}

html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
               "Segoe UI", sans-serif;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 82% 0%, rgba(14,165,233,.10), transparent 26rem),
    var(--surface-0);
  color: var(--text);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebarNav"] span { font-weight: 560; }
header[data-testid="stHeader"] { background: rgba(2,6,23,.72); }
.block-container {
  max-width: 1500px;
  padding-top: 1.35rem;
  padding-bottom: 2.5rem;
}

.page-kicker {
  color: var(--accent);
  font-size: .76rem;
  font-weight: 750;
  letter-spacing: .12em;
  margin-bottom: .35rem;
  text-transform: uppercase;
}
.page-title {
  color: var(--text);
  font-size: clamp(1.8rem, 3vw, 2.55rem);
  font-weight: 760;
  letter-spacing: -.035em;
  line-height: 1.08;
  margin: 0;
}
.page-subtitle {
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.6;
  margin: .7rem 0 1.25rem;
  max-width: 78rem;
}
.page-rule {
  background: linear-gradient(90deg, var(--accent), rgba(56,189,248,0));
  border: 0;
  height: 1px;
  margin: 0 0 1.1rem;
}
.section-title {
  color: var(--accent);
  font-size: 1.1rem;
  font-weight: 680;
  margin: 1.15rem 0 .45rem;
}
.info-box, .method-card {
  background: linear-gradient(145deg, rgba(15,23,42,.96), rgba(15,23,42,.72));
  border: 1px solid var(--border);
  border-radius: .75rem;
  padding: .9rem 1rem;
}
.method-card { min-height: 9.5rem; }
.method-card h4 { color: var(--text); margin: 0 0 .35rem; }
.method-card p { color: var(--muted); line-height: 1.5; margin: 0; }
.hash-value {
  color: #cbd5e1;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .78rem;
  overflow-wrap: anywhere;
}
[data-testid="stMetric"] {
  background: rgba(15,23,42,.72);
  border: 1px solid var(--border);
  border-radius: .7rem;
  padding: .65rem .8rem;
}
[data-testid="stMetricValue"] { font-size: 1.45rem; }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; }
div[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent-strong), #0284c7);
  border: 0;
  font-weight: 700;
}
div[data-testid="stDownloadButton"] > button {
  border-color: #334155;
  width: 100%;
}
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: .6rem; }

@media (max-width: 760px) {
  .block-container { padding-top: .8rem; }
  .page-subtitle { font-size: .94rem; }
}
</style>
"""


def apply_global_styles() -> None:
    """Install the shared visual system once from the application shell."""

    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)


def render_page_header(*, kicker: str, title: str, subtitle: str) -> None:
    """Render a consistent, escaped page heading."""

    st.markdown(
        (
            f'<div class="page-kicker">{html.escape(kicker)}</div>'
            f'<h1 class="page-title">{html.escape(title)}</h1>'
            f'<p class="page-subtitle">{html.escape(subtitle)}</p>'
            '<hr class="page-rule">'
        ),
        unsafe_allow_html=True,
    )


def render_section_title(title: str) -> None:
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )

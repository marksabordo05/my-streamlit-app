import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="4Ps Children Beneficiaries — Descriptive Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Color palette (Cool Cerulean / Baby Blue Eyes / Light Blue Grey / Platinum)
COLORS = ["#8EB1D1", "#A7C7E7", "#C4D8E5", "#6a9bbf", "#4f80a4", "#d6e8f5"]

# ── Plotly default theme: always white background, readable text ───────────────
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLORS

_plotly_layout = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", color="#2c3e50", size=12),
    title_font=dict(family="Inter, sans-serif", color="#1a2a3a", size=14),
    legend=dict(bgcolor="#ffffff", bordercolor="#C4D8E5", borderwidth=1,
                font=dict(color="#2c3e50")),
    xaxis=dict(gridcolor="#E8ECEF", linecolor="#C4D8E5",
               tickfont=dict(color="#2c3e50"), title_font=dict(color="#2c3e50")),
    yaxis=dict(gridcolor="#E8ECEF", linecolor="#C4D8E5",
               tickfont=dict(color="#2c3e50"), title_font=dict(color="#2c3e50")),
    margin=dict(t=50, b=30, l=10, r=10),
)

def _fig(fig):
    """Apply white-background layout to any Plotly figure and return it."""
    fig.update_layout(**_plotly_layout)
    return fig

def _empty_fig(title, x_title="", y_title=""):
    """A chart frame with no data: same title/axes/size as the real chart, plus a note."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        annotations=[dict(
            text="No data selected", x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#5a7a9a"),
        )],
        height=380,
    )
    return fig

def _df(dataframe, height=None):
    """Render a DataFrame as a clean white HTML table (bypasses Streamlit's dark grid).
    An empty DataFrame keeps its headers and shows a single 'No data' row."""
    df_display = dataframe.copy()
    for col in df_display.select_dtypes(include="float").columns:
        df_display[col] = df_display[col].map(
            lambda x: "—" if pd.isna(x) else (f"{x:,.4f}" if abs(x) < 1e6 else f"{x:,.0f}")
        )
    for col in df_display.select_dtypes(include="int").columns:
        df_display[col] = df_display[col].map(lambda x: f"{x:,}")

    rows = ""
    for _, row in df_display.iterrows():
        cells = "".join(f"<td style='padding:8px 14px;border-bottom:1px solid #E8ECEF;"
                        f"color:#2c3e50;font-size:0.875rem;'>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"

    if df_display.empty:
        rows = (f"<tr><td colspan='{max(len(df_display.columns), 1)}' "
                f"style='padding:14px;text-align:center;color:#7a8ba0;font-size:0.875rem;'>"
                f"No data</td></tr>")

    headers = "".join(
        f"<th style='padding:9px 14px;text-align:left;background:#f0f6fb;"
        f"color:#2c4a6e;font-size:0.78rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.04em;border-bottom:2px solid #C4D8E5;'>{c}</th>"
        for c in df_display.columns
    )

    scroll = f"max-height:{height}px;overflow-y:auto;" if height else ""
    html = (
        f"<div style='border:1px solid #C4D8E5;border-radius:10px;overflow:hidden;"
        f"background:#ffffff;{scroll}'>"
        f"<table style='width:100%;border-collapse:collapse;background:#ffffff;'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)

# ── Global CSS (main area) ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Main background ── */
.stApp, [data-testid="stAppViewContainer"] > .main {
    background: #f4f7fa;
    color: #2c3e50;
}
.block-container {
    max-width: 1600px;
    padding-top: 3.2rem;   /* clears the fixed top bar so the title is never cut off */
    padding-bottom: 2.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
[data-testid="stHeader"] {
    background: #f4f7fa;
    border-bottom: none !important;
    box-shadow: none !important;
}

/* ── Headings ── */
h1 {
    color: #1a2a3a !important;
    font-weight: 800 !important;
    font-size: 2.0rem !important;
    letter-spacing: -0.03em;
    line-height: 1.35 !important;
    margin-top: 0.6rem !important;
}
h2 { color: #1a2a3a !important; font-weight: 700 !important; font-size: 1.4rem !important; }
h3 { color: #2c4a6e !important; font-weight: 600 !important; }
p, label, .stMarkdown { color: #2c3e50; }
[data-testid="stCaptionContainer"] { color: #5a7a9a !important; font-size: 0.82rem; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #C4D8E5;
    border-top: 3px solid #8EB1D1;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(142,177,209,0.12);
}
[data-testid="stMetricLabel"] {
    color: #5a7a9a !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: #1a2a3a !important;
    font-weight: 800 !important;
    font-size: 1.9rem !important;
}
[data-testid="stMetricDelta"] { color: #4f80a4 !important; }

/* ── Download button (main area) ── */
.stDownloadButton > button {
    background: #8EB1D1 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
    transition: background 0.2s;
}
.stDownloadButton > button:hover {
    background: #6a9bbf !important;
    color: #ffffff !important;
}

/* ── Tabs ── */
div[data-testid="stTabs"] {
    background: #ffffff;
    border: 1px solid #C4D8E5;
    border-radius: 12px;
    padding: 6px 10px 0;
    box-shadow: 0 1px 6px rgba(142,177,209,0.1);
}
div[data-testid="stTabs"] button {
    color: #5a7a9a !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 8px 14px !important;
    border-radius: 8px 8px 0 0 !important;
    background: transparent !important;
}
div[data-testid="stTabs"] button:hover {
    background: #f0f6fb !important;
    color: #2c3e50 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2c4a6e !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #8EB1D1 !important;
    background: #f0f6fb !important;
}

/* ── DataFrames — force light theme on every iframe/canvas Streamlit renders ── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background: #ffffff !important;
    border: 1px solid #C4D8E5 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] > div {
    background: #ffffff !important;
    color: #2c3e50 !important;
}
.dvn-scroller, .gdg-style, .gdg-cell,
[class*="dvn-"], [class*="gdg-"] {
    background: #ffffff !important;
    color: #2c3e50 !important;
}
[class*="gdg-header"] {
    background: #f0f6fb !important;
    color: #2c4a6e !important;
    font-weight: 600 !important;
}

/* ── Alerts — no dark backgrounds ── */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left: 4px solid #8EB1D1 !important;
    background: #f0f6fb !important;
    color: #2c3e50 !important;
}
div[data-testid="stAlert"] > div { background: transparent !important; }
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span { color: #2c3e50 !important; }

/* ── General form inputs (main area): white background, light-blue border ── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="base-input"],
[data-baseweb="input"] {
    border-color: #C4D8E5 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    color: #2c3e50 !important;
}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within { border-color: #2F6FD0 !important; }

input, textarea, [data-baseweb="input"] input {
    color: #2c3e50 !important;
    background: #ffffff !important;
}

/* Number input container and +/- buttons */
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    color: #2c3e50 !important;
    border-color: #C4D8E5 !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] button {
    background: #f0f6fb !important;
    color: #4f80a4 !important;
    border-left: 1px solid #C4D8E5 !important;
    border-radius: 0 !important;
}
[data-testid="stNumberInput"] button:hover {
    background: #C4D8E5 !important;
}

/* ── Checkbox & radio ── */
[data-testid="stCheckbox"] span,
[data-testid="stRadio"] span { color: #2c3e50 !important; }

/* ── Divider & scrollbar ── */
hr { border-color: #C4D8E5; margin: 1.5rem 0; }
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: #f4f7fa; }
::-webkit-scrollbar-thumb { background: #C4D8E5; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #8EB1D1; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR STYLES — all sidebar / filter / slider CSS lives here.
# Change the colors in :root only.
# ══════════════════════════════════════════════════════════════════════════════
def apply_sidebar_styles():
    """Inject all sidebar CSS in one place. Change colors in :root only."""
    st.markdown("""
<style>
:root {
    --accent:        #2F6FD0;
    --accent-dark:   #2459A8;
    --accent-light:  #BFD4F2;
    --accent-bg:     #EEF5FB;   /* dropzone fill */
    --dropzone-line: #8EB1D1;   /* dashed border */
}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div { background:#ffffff !important; }
section[data-testid="stSidebar"] { border-right:2px solid #C4D8E5; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color:#2c3e50; }

.sb-header {
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#9aa5b4; margin:16px 0 8px;
}
section[data-testid="stSidebar"] hr {
    border:none !important; border-top:1px solid #e2e6ea !important; margin:12px 0 !important;
}

/* ───────── 1. FILE UPLOADER ───────── */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background:var(--accent-bg) !important;
    border:2px dashed var(--dropzone-line) !important;
    border-radius:12px !important;
    padding:14px !important;
}
/* strip inner layers so nothing paints a box behind the label */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
    padding:0 !important;
    display:flex !important;
    flex-direction:column !important;
    align-items:stretch !important;
    gap:10px !important;                 /* even spacing button <-> size text */
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div {
    background:transparent !important;
    padding:0 !important;
    display:flex !important;
    flex-direction:column !important;
    gap:10px !important;
}
/* the button */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    gap:8px !important;
    width:100% !important;
    margin:0 !important;
    padding:10px 18px !important;
    background:var(--accent) !important;
    color:#ffffff !important;
    border:none !important;
    border-radius:8px !important;
    font-size:0.85rem !important;
    font-weight:600 !important;
    cursor:pointer !important;
    outline:none !important;
    box-shadow:none !important;
    transition:background .18s ease, transform .18s ease, box-shadow .18s ease;
}
/* remove the white box behind the "Upload" label (and keep the icon white) */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button p,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button div {
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
    padding:0 !important;
    margin:0 !important;
    color:#ffffff !important;
    fill:#ffffff !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
    background:var(--accent-dark) !important;
    transform:translateY(-1px);
    box-shadow:0 4px 10px rgba(47,111,208,0.30) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:focus-visible {
    box-shadow:0 0 0 3px rgba(47,111,208,0.30) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    display:block !important;
    text-align:center !important;
    font-size:0.71rem !important;
    color:#7a8ba0 !important;
    background:transparent !important;
    margin:0 !important;
}

/* ───────── 2a. REGION MULTISELECT ───────── */
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background:#ffffff !important;
    border:1px solid var(--accent-light) !important;
    border-radius:8px !important;
    box-shadow:none !important;
}
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
    border-color:var(--accent) !important;           /* blue, not red */
    box-shadow:0 0 0 2px rgba(47,111,208,0.18) !important;
}
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] input {
    background:transparent !important; color:#2c3e50 !important;
}
/* selected tags */
section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background:var(--accent) !important;
    border:none !important;
    border-radius:6px !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"],
section[data-testid="stSidebar"] [data-baseweb="tag"] span,
section[data-testid="stSidebar"] [data-baseweb="tag"] svg {
    color:#ffffff !important;
    fill:#ffffff !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] [role="presentation"] svg {
    width:10px !important; height:10px !important;   /* small "×" */
}
/* dropdown list (rendered in a portal, so no sidebar prefix) */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] ul {
    background:#ffffff !important;
    border:1px solid var(--accent-light) !important;
    border-radius:8px !important;
}
[data-baseweb="popover"] [role="option"] {
    background:#ffffff !important; color:#2c3e50 !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background:var(--accent-bg) !important; color:var(--accent-dark) !important;
}

/* ───────── 2b. SLIDERS (sidebar year range + model-tab slider) ─────────
   Streamlit paints the fill with an inline red gradient; shifting its hue turns it blue.
   For the exact #2F6FD0 also set primaryColor in .streamlit/config.toml. */
[data-testid="stSlider"] [data-baseweb="slider"] div[style*="linear-gradient"] {
    filter:hue-rotate(214deg) saturate(0.85);
}
[data-testid="stSlider"] div[role="slider"] {
    background:var(--accent) !important;
    border-color:var(--accent) !important;
    box-shadow:0 0 0 4px rgba(47,111,208,0.22) !important;
}
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"],
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {
    color:var(--accent) !important;
    font-weight:600 !important;
}

/* ───────── 3. SIDEBAR COLLAPSE / EXPAND ARROW (always visible) ───────── */
/* Streamlit hides the « arrow until you hover the sidebar; force it to stay visible.
   Covers the collapse button (sidebar open) and the expand button (sidebar closed),
   including the older "collapsedControl" test id. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarHeader"] button,
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] button,
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] button {
    opacity:1 !important;
    visibility:visible !important;
}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] button {
    background:var(--accent-bg) !important;
    border:1px solid var(--accent-light) !important;
    border-radius:8px !important;
    transition:background .18s ease;
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="collapsedControl"] button:hover {
    background:var(--accent-light) !important;
}
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stExpandSidebarButton"] *,
[data-testid="collapsedControl"] * {
    color:var(--accent) !important;
    fill:var(--accent) !important;
}

/* ───────── 2c. RESET BUTTON ───────── */
section[data-testid="stSidebar"] .stButton > button {
    width:100% !important;
    background:var(--accent) !important;
    color:#ffffff !important;
    border:none !important;
    border-radius:8px !important;
    padding:0.55rem 1rem !important;
    font-weight:600 !important;
    opacity:1 !important;
    transition:background .18s ease, transform .18s ease, box-shadow .18s ease;
}
section[data-testid="stSidebar"] .stButton > button * {
    background:transparent !important; color:#ffffff !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background:var(--accent-dark) !important;
    transform:translateY(-1px);
    box-shadow:0 4px 10px rgba(47,111,208,0.28) !important;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color:#7a8ba0 !important; font-size:0.75rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Title ──────────────────────────────────────────────────────────────────────
st.title("4Ps Children Beneficiaries — Descriptive Analytics")
st.caption("Explore beneficiary counts by region and year, and compare four machine-learning classifiers.")

# ── Data loading ───────────────────────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/marksabordo05/beneficiaries-4ps/"
    "main/4Ps_children_structured.xlsx"
)

@st.cache_data(show_spinner="Loading dataset…")
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_excel(url)
    return _clean(df)

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Region"] = df["Region"].astype(str).str.strip()
    df = df[df["Region"].str.lower() != "philippines"].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Children"] = pd.to_numeric(df["Children"], errors="coerce")
    df = df.dropna(subset=["Region", "Year", "Children"])
    return df


# ── Filter helpers ─────────────────────────────────────────────────────────────
def reset_filters():
    """on_click callback: runs before widgets render, so writing widget keys is safe."""
    st.session_state["filter_regions"] = list(st.session_state["_all_regions"])
    st.session_state["filter_years"]   = tuple(st.session_state["_year_bounds"])


def _valid_years(value, bounds):
    """True if value is a (lo, hi) pair of ints inside bounds with lo <= hi."""
    try:
        lo, hi = int(value[0]), int(value[1])
        return len(value) == 2 and bounds[0] <= lo <= hi <= bounds[1]
    except (TypeError, ValueError, IndexError):
        return False


def resolve_filters(all_regions, year_bounds):
    """Return (regions, years, region_empty, year_fallback_used).
    An empty Region selection stays empty (the dashboard keeps its layout but shows no data);
    an invalid year range falls back to the full range."""
    regions = [r for r in (st.session_state.get("filter_regions") or []) if r in all_regions]
    region_empty = not regions

    years = st.session_state.get("filter_years")
    year_fallback = not _valid_years(years, year_bounds)
    years = tuple(year_bounds) if year_fallback else (int(years[0]), int(years[1]))

    return regions, years, region_empty, year_fallback


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:

    apply_sidebar_styles()

    # ── DATA SOURCE ────────────────────────────────────────────────────────────
    st.markdown("<p class='sb-header'>Data source</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a replacement file (optional)",
        type=["csv", "xlsx"],
    )
    st.caption("By default, the app loads the dataset directly from the linked GitHub repository.")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Load data — uploaded file takes priority
    if uploaded_file is not None:
        try:
            _raw = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") \
                   else pd.read_excel(uploaded_file)
            df = _clean(_raw)
            st.success(f"Using: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    else:
        try:
            df = load_data(DATA_URL)
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    # ── INTERACTIVE FILTERS ────────────────────────────────────────────────────
    st.markdown("<p class='sb-header'>Interactive filters</p>", unsafe_allow_html=True)

    all_regions = sorted(df["Region"].unique().tolist())
    year_min    = int(df["Year"].min())
    year_max    = int(df["Year"].max())
    year_bounds = (year_min, year_max)

    # Needed by reset_filters()
    st.session_state["_all_regions"] = all_regions
    st.session_state["_year_bounds"] = year_bounds

    # Seed widget keys on first load, and repair them if the dataset changed
    if "filter_regions" not in st.session_state:
        st.session_state["filter_regions"] = list(all_regions)
    elif not set(st.session_state["filter_regions"]).issubset(all_regions):
        st.session_state["filter_regions"] = list(all_regions)

    if not _valid_years(st.session_state.get("filter_years"), year_bounds):
        st.session_state["filter_years"] = year_bounds   # keeps slider from erroring on bad bounds

    st.multiselect("Region", all_regions, key="filter_regions")
    st.slider("Year range", year_min, year_max, key="filter_years")

    # Resolve once; everything below uses these values
    sel_regions, sel_years, region_empty, year_fallback = resolve_filters(all_regions, year_bounds)

    if region_empty:
        st.caption("No region selected, so no data is displayed.")
    if year_fallback:
        st.caption("Invalid year range, showing all years.")

    st.button("↺  Reset filters", on_click=reset_filters, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("National 'Philippines' total is always excluded from all views.")

# ── Apply filters (single filtered dataframe used by everything below) ─────────
# An empty region selection gives an empty `view`. The page layout (KPI cards, tabs,
# charts, tables) still renders; each element simply shows no data.
view = df[
    df["Region"].isin(sel_regions) &
    df["Year"].between(sel_years[0], sel_years[1])
].copy()
no_data = view.empty

_ch = view["Children"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Records",          f"{len(view):,}")
k2.metric("Regions",          f"{view['Region'].nunique():,}")
k3.metric("Average children", f"{_ch.mean():,.0f}"  if len(view) else "—")
k4.metric("Median children",  f"{_ch.median():,.0f}" if len(view) else "—")
k5.metric("Range (min–max)",
          f"{int(_ch.min()):,} – {int(_ch.max()):,}" if len(view) else "—")

st.markdown("<hr style='border-color:#C4D8E5;margin:0.6rem 0 1rem;'>", unsafe_allow_html=True)

# ── 7 Tabs ─────────────────────────────────────────────────────────────────────
(tab_overview, tab_trends, tab_dist,
 tab_scatter, tab_ml, tab_quality, tab_dataset) = st.tabs([
    "Overview",
    "Trend analysis",
    "Distribution",
    "Scatterplots",
    "Classification models",
    "Data quality",
    "Dataset",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Descriptive statistics")
    if no_data:
        desc = pd.DataFrame(columns=["Column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"])
    else:
        desc = view.describe().T.reset_index().rename(columns={"index": "Column"})
    _df(desc)

    st.subheader("Average children per region")
    if no_data:
        fig_avg = _empty_fig("Average Children Beneficiaries by Region", "Avg Children", "")
    else:
        avg_region = (
            view.groupby("Region", as_index=False)["Children"]
            .mean()
            .sort_values("Children")
        )
        fig_avg = px.bar(
            avg_region, x="Children", y="Region", orientation="h",
            title="Average Children Beneficiaries by Region",
            color="Children",
            color_continuous_scale=["#d6e8f5", "#8EB1D1", "#4f80a4"],
            labels={"Children": "Avg Children", "Region": ""},
        )
        fig_avg.update_coloraxes(showscale=False)
        fig_avg.update_layout(margin=dict(l=0, r=20, t=40, b=20))
    st.plotly_chart(_fig(fig_avg), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    st.subheader("Children beneficiaries over time")
    if no_data:
        fig_line = _empty_fig("Children Beneficiaries Over Time by Region", "Year", "Children Beneficiaries")
    else:
        trend = view.groupby(["Year", "Region"], as_index=False)["Children"].sum()
        fig_line = px.line(
            trend, x="Year", y="Children", color="Region", markers=True,
            title="Children Beneficiaries Over Time by Region",
            labels={"Children": "Children Beneficiaries", "Year": "Year"},
        )
        fig_line.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            hovermode="x unified",
        )
    st.plotly_chart(_fig(fig_line), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_dist:
    st.subheader("Distribution of Children Beneficiaries")
    if no_data:
        fig_hist = _empty_fig("Distribution of Children Beneficiaries", "Children Beneficiaries", "count")
    else:
        fig_hist = px.histogram(
            view, x="Children", nbins=20, marginal="box",
            title="Distribution of Children Beneficiaries",
            color_discrete_sequence=["#8EB1D1"],
            labels={"Children": "Children Beneficiaries"},
        )
        fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(_fig(fig_hist), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCATTERPLOTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_scatter:
    st.subheader("Year vs. Children Beneficiaries")
    if no_data:
        fig_scatter = _empty_fig("Year vs. Children Beneficiaries by Region", "Year", "Children Beneficiaries")
    else:
        fig_scatter = px.scatter(
            view, x="Year", y="Children", color="Region",
            hover_data=["Region", "Year", "Children"],
            title="Year vs. Children Beneficiaries by Region",
            labels={"Children": "Children Beneficiaries", "Year": "Year"},
        )
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.75))
        fig_scatter.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
        )
    st.plotly_chart(_fig(fig_scatter), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_quality:
    st.subheader("Data quality")

    if no_data:
        missing = pd.DataFrame(columns=["Column", "Missing values", "Missing %"])
    else:
        missing = (
            view.isna().sum()
            .reset_index()
            .rename(columns={"index": "Column", 0: "Missing values"})
        )
        missing["Missing %"] = (missing["Missing values"] / len(view) * 100).round(2)

    col_tbl, col_bar = st.columns([1, 2], gap="large")
    with col_tbl:
        st.markdown("**Missing values per column**")
        _df(missing)
        st.metric("Total records", f"{len(view):,}")

    with col_bar:
        if no_data:
            fig_miss = _empty_fig("Missing Values per Column", "", "Missing count")
        else:
            fig_miss = px.bar(
                missing, x="Column", y="Missing values",
                title="Missing Values per Column",
                color="Missing values",
                color_continuous_scale=["#d6e8f5", "#8EB1D1", "#4f80a4"],
                text="Missing values",
            )
            fig_miss.update_coloraxes(showscale=False)
            fig_miss.update_traces(textposition="outside")
            fig_miss.update_layout(yaxis_title="Missing count", xaxis_title="")
        st.plotly_chart(_fig(fig_miss), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab_dataset:
    st.subheader("Filtered dataset")
    st.caption(
        f"{len(view):,} records · {view['Region'].nunique():,} regions · "
        f"years {int(view['Year'].min()) if not view.empty else '—'}–"
        f"{int(view['Year'].max()) if not view.empty else '—'}"
    )
    _df(view.sort_values(["Region", "Year"]).reset_index(drop=True), height=480)
    st.download_button(
        label="⬇ Download filtered data as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="4ps_filtered.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CLASSIFICATION MODELS
# (kept last in the script because the "fewer than 2 classes" guard calls st.stop())
# ══════════════════════════════════════════════════════════════════════════════
def _ml_controls():
    """Model controls; same widget keys in both the empty and the normal branch."""
    st.markdown("#### Model controls")
    ctrl1, ctrl2 = st.columns(2, gap="large")
    with ctrl1:
        test_pct = st.slider(
            "Test set size (%)", min_value=10, max_value=40,
            value=20, step=5, key="ml_test_size",
        )
    with ctrl2:
        rand_state = st.number_input(
            "Random state", min_value=0, max_value=9999,
            value=42, step=1, key="ml_rand_state",
        )
    return test_pct / 100, int(rand_state)


with tab_ml:
    st.subheader("Beneficiary classification — four machine-learning algorithms")
    st.markdown(
        "This module classifies each region-year record as **Increase** or **No Increase** "
        "compared with the previous year. Only the previous year's value is used as a "
        "predictor, to prevent target leakage."
    )

    # ── Classification target selectbox ───────────────────────────────────────
    st.selectbox(
        "Classification target",
        ["Increase vs No Increase (derived)"],
        key="ml_target_sel",
    )
    st.caption(
        "**Derived target:** for each region, records are sorted by year. "
        "A record is labelled **Increase (1)** if its Children value is greater than "
        "the previous consecutive year's value for the same region, and "
        "**No Increase (0)** otherwise. Records with no prior consecutive year are dropped."
    )

    st.markdown("---")

    if no_data:
        # ── Same layout as below, but with no data ────────────────────────────
        st.markdown("#### Target class distribution")
        col_tbl, col_chart = st.columns([1, 2], gap="large")
        with col_tbl:
            _df(pd.DataFrame(columns=["Class", "Records"]))
        with col_chart:
            st.plotly_chart(
                _fig(_empty_fig("Records per Target Class", "", "Records")),
                use_container_width=True,
            )

        st.markdown("---")
        _ml_controls()
        st.markdown("---")

        empty_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        st.markdown("#### Model performance")
        _df(pd.DataFrame(columns=empty_cols))

        st.markdown("**Final comparison** (sorted by accuracy)")
        _df(pd.DataFrame(columns=empty_cols))

        st.markdown("#### Accuracy comparison")
        st.plotly_chart(
            _fig(_empty_fig("Accuracy Comparison — Four Classifiers", "Model", "Accuracy")),
            use_container_width=True,
        )

        st.markdown("#### Confusion matrices")
        _e1, _e2 = st.columns(2, gap="large"), st.columns(2, gap="large")
        for _c, _n in zip([_e1[0], _e1[1], _e2[0], _e2[1]],
                          ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]):
            _c.plotly_chart(_fig(_empty_fig(_n)), use_container_width=True)

        st.markdown("#### ROC curve comparison")
        st.plotly_chart(
            _fig(_empty_fig("ROC Curve Comparison", "False Positive Rate", "True Positive Rate")),
            use_container_width=True,
        )

        st.markdown("#### Reality check")
        rc1, rc2 = st.columns(2)
        rc1.metric("Majority-class baseline (random split)", "—")
        rc1.caption(
            "A trivial classifier that always predicts the majority class sets the bar "
            "that model scores should beat to be meaningful."
        )
        rc2.markdown("**Time-based split** (train ≤ 2014, test > 2014)  \nBaseline: **—**")
        with rc2:
            _df(pd.DataFrame(columns=["Model", "Accuracy", "F1-Score"]))

        st.warning(
            "⚠️ **High random-split scores may only reflect an era pattern.** "
            "If beneficiary counts consistently rise year-over-year, a model can appear "
            "accurate simply by learning 'it always increases' — not by understanding "
            "regional dynamics. Compare the random-split results with the time-based "
            "split and the majority-class baseline before drawing conclusions."
        )

    else:
        # ── Prepare labelled dataset ───────────────────────────────────────────
        prepared = df.sort_values(["Region", "Year"]).copy()
        prepared["Previous_Children"] = prepared.groupby("Region")["Children"].shift(1)
        prepared["Previous_Year"]     = prepared.groupby("Region")["Year"].shift(1)
        prepared = prepared[(prepared["Year"] - prepared["Previous_Year"]) == 1].copy()
        prepared["Increase"] = (prepared["Children"] > prepared["Previous_Children"]).astype(int)

        # Guard: need at least 2 classes
        if prepared["Increase"].nunique() < 2:
            st.warning(
                "The filtered data produces fewer than 2 target classes "
                "(Increase / No Increase). Adjust the sidebar filters to include "
                "more regions or a wider year range."
            )
            st.stop()

        # ── Target class distribution ──────────────────────────────────────────
        st.markdown("#### Target class distribution")

        counts = (
            prepared["Increase"]
            .map({0: "No Increase", 1: "Increase"})
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Records")
            .sort_values("Class")
        )

        col_tbl, col_chart = st.columns([1, 2], gap="large")

        with col_tbl:
            _df(counts)

        with col_chart:
            fig_dist = px.bar(
                counts, x="Class", y="Records",
                title="Records per Target Class",
                color="Class",
                color_discrete_map={"Increase": "#8EB1D1", "No Increase": "#A7C7E7"},
                text="Records",
            )
            fig_dist.update_traces(textposition="outside")
            fig_dist.update_layout(
                showlegend=True,
                legend_title_text="Class",
                yaxis_title="Records",
                xaxis_title="",
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(_fig(fig_dist), use_container_width=True)

        st.markdown("---")

        # ── Controls ───────────────────────────────────────────────────────────
        test_size, random_state = _ml_controls()

        # ── Run models ─────────────────────────────────────────────────────────
        st.markdown("---")

        X = prepared[["Region", "Year", "Previous_Children"]]
        y = prepared["Increase"]

        stratify_arg = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )

        pre = ColumnTransformer([
            ("region",  OneHotEncoder(handle_unknown="ignore"), ["Region"]),
            ("numeric", StandardScaler(),                       ["Year", "Previous_Children"]),
        ])

        MODEL_DEFS = {
            "Logistic Regression": LogisticRegression(max_iter=10000),
            "Decision Tree":       DecisionTreeClassifier(random_state=random_state),
            "Random Forest":       RandomForestClassifier(random_state=random_state),
            "SVM":                 SVC(probability=True, random_state=random_state),
        }

        # Cache key: tuple of control values so the cache busts when user changes them
        @st.cache_data(show_spinner=False)
        def _train_models(prepared_hash, ts, rs):
            _X = prepared[["Region", "Year", "Previous_Children"]]
            _y = prepared["Increase"]
            _strat = _y if _y.value_counts().min() >= 2 else None
            _Xtr, _Xte, _ytr, _yte = train_test_split(
                _X, _y, test_size=ts, random_state=rs, stratify=_strat
            )
            _pre = ColumnTransformer([
                ("region",  OneHotEncoder(handle_unknown="ignore"), ["Region"]),
                ("numeric", StandardScaler(),                       ["Year", "Previous_Children"]),
            ])
            _scores, _fitted, _probs = [], {}, {}
            for name, estimator in MODEL_DEFS.items():
                pipe = Pipeline([("pre", _pre), ("clf", estimator)])
                pipe.fit(_Xtr, _ytr)
                pred = pipe.predict(_Xte)
                prob = pipe.predict_proba(_Xte)[:, 1]
                _scores.append({
                    "Model":     name,
                    "Accuracy":  accuracy_score(_yte, pred),
                    "Precision": precision_score(_yte, pred, zero_division=0),
                    "Recall":    recall_score(_yte, pred, zero_division=0),
                    "F1-Score":  f1_score(_yte, pred, zero_division=0),
                    "ROC-AUC":   roc_auc_score(_yte, prob) if _yte.nunique() == 2 else float("nan"),
                })
                _fitted[name] = pred
                _probs[name]  = prob
            return pd.DataFrame(_scores), _fitted, _probs, _yte

        with st.spinner("Training four classifiers…"):
            score_df, fitted_preds, fitted_probs, y_test = _train_models(
                len(prepared), test_size, random_state
            )

        score_df = score_df.reset_index(drop=True)
        metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

        # ── Model performance table ────────────────────────────────────────────
        st.markdown("#### Model performance")

        def _highlight_best(s):
            is_best = s == s.max()
            return ["background-color:#d6e8f5; font-weight:700" if v else "" for v in is_best]

        styled = (
            score_df.style
            .apply(_highlight_best, subset=metric_cols)
            .format({c: "{:.4f}" for c in metric_cols})
        )
        _df(score_df)

        st.markdown("**Final comparison** (sorted by accuracy)")
        final_tbl = score_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
        _df(final_tbl)

        # ── Accuracy comparison bar chart ──────────────────────────────────────
        st.markdown("#### Accuracy comparison")
        fig_acc = px.bar(
            final_tbl, x="Model", y="Accuracy",
            title="Accuracy Comparison — Four Classifiers",
            color="Model",
            color_discrete_sequence=COLORS,
            text=final_tbl["Accuracy"].map(lambda v: f"{v*100:.1f}%"),
        )
        fig_acc.update_traces(textposition="outside")
        fig_acc.update_layout(
            showlegend=False,
            yaxis=dict(range=[0, 1.12], tickformat=".0%"),
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(_fig(fig_acc), use_container_width=True)

        # ── Confusion matrices 2×2 grid ────────────────────────────────────────
        st.markdown("#### Confusion matrices")
        model_names = list(MODEL_DEFS.keys())
        cm_row1, cm_row2 = st.columns(2, gap="large"), st.columns(2, gap="large")
        cm_grid = [cm_row1[0], cm_row1[1], cm_row2[0], cm_row2[1]]
        class_labels = ["No Increase", "Increase"]

        for col, name in zip(cm_grid, model_names):
            cm_arr = confusion_matrix(y_test, fitted_preds[name], labels=[0, 1])
            fig_cm = px.imshow(
                cm_arr,
                x=[f"Pred: {l}" for l in class_labels],
                y=[f"Actual: {l}" for l in class_labels],
                text_auto=True,
                title=name,
                color_continuous_scale=["#E8ECEF", "#A7C7E7", "#4f80a4"],
                labels={"color": "Count"},
            )
            fig_cm.update_coloraxes(showscale=False)
            fig_cm.update_layout(margin=dict(t=40, b=10, l=10, r=10))
            col.plotly_chart(_fig(fig_cm), use_container_width=True)

        # ── ROC curve comparison ───────────────────────────────────────────────
        st.markdown("#### ROC curve comparison")
        fig_roc = go.Figure()
        for (name, prob), color in zip(fitted_probs.items(), COLORS):
            if y_test.nunique() == 2:
                fpr, tpr, _ = roc_curve(y_test, prob)
                auc_val     = roc_auc_score(y_test, prob)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC = {auc_val:.3f})",
                    line=dict(color=color, width=2),
                ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random classifier",
            line=dict(dash="dash", color="#aaaaaa", width=1.5),
        ))
        fig_roc.update_layout(
            title="ROC Curve Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(orientation="h", yanchor="bottom", y=-0.5),
            hovermode="x unified",
        )
        st.plotly_chart(_fig(fig_roc), use_container_width=True)

        # ── Reality check ──────────────────────────────────────────────────────
        st.markdown("#### Reality check")

        baseline = max(y_test.mean(), 1 - y_test.mean())
        rc1, rc2 = st.columns(2)
        rc1.metric("Majority-class baseline (random split)", f"{baseline:.2%}")
        rc1.caption(
            "A trivial classifier that always predicts the majority class achieves "
            f"{baseline:.2%} accuracy. Model scores should beat this to be meaningful."
        )

        # Time-based split: no Year feature to avoid era leakage
        tb_train = prepared[prepared["Year"] <= 2014]
        tb_test  = prepared[prepared["Year"] >  2014]

        if len(tb_train) > 0 and len(tb_test) > 0 and tb_train["Increase"].nunique() == 2:
            tb_pre = ColumnTransformer([
                ("region",  OneHotEncoder(handle_unknown="ignore"), ["Region"]),
                ("numeric", StandardScaler(),                       ["Previous_Children"]),
            ])
            tb_scores = []
            for name, estimator in MODEL_DEFS.items():
                pipe = Pipeline([("pre", tb_pre), ("clf", estimator)])
                pipe.fit(tb_train[["Region", "Previous_Children"]], tb_train["Increase"])
                pred = pipe.predict(tb_test[["Region", "Previous_Children"]])
                tb_scores.append({
                    "Model":    name,
                    "Accuracy": accuracy_score(tb_test["Increase"], pred),
                    "F1-Score": f1_score(tb_test["Increase"], pred, zero_division=0),
                })
            tb_df = pd.DataFrame(tb_scores).sort_values("Accuracy", ascending=False).reset_index(drop=True)
            tb_baseline = max(tb_test["Increase"].mean(), 1 - tb_test["Increase"].mean())
            rc2.markdown(f"**Time-based split** (train ≤ 2014, test > 2014)  \n"
                         f"Baseline: **{tb_baseline:.2%}**")
            with rc2:
                _df(tb_df)
        else:
            rc2.info("Not enough data on both sides of 2014 for a time-based split.")

        st.warning(
            "⚠️ **High random-split scores may only reflect an era pattern.** "
            "If beneficiary counts consistently rise year-over-year, a model can appear "
            "accurate simply by learning 'it always increases' — not by understanding "
            "regional dynamics. Compare the random-split results with the time-based "
            "split and the majority-class baseline before drawing conclusions."
        )
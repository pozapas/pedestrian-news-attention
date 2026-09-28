"""common.py -- shared helpers for the TRB-P analysis scripts.

Loads the shared figure style + palette, provides results.json accumulation,
and event-data loading. Imported by p02..p05.
"""
from __future__ import annotations
import sys, io, json
from pathlib import Path
import numpy as np
import pandas as pd

# UTF-8 stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                  # TRB_P_ped_attention/
DATA_BUILD = ROOT / "data_build"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
STYLE_DIR = ROOT / "figure_style"            # shared repo-level kit
RESULTS = DATA_BUILD / "results.json"
for d in (FIGURES, TABLES, DATA_BUILD):
    d.mkdir(exist_ok=True)

# make palette importable
sys.path.insert(0, str(STYLE_DIR))
import palette  # noqa: E402
from palette import PED, AV, HIGHLIGHT, SECONDARY, TERTIARY, NEUTRAL, INK  # noqa


def use_style():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use(str(STYLE_DIR / "trb_news.mplstyle"))
    # Arial may be unavailable; matplotlib falls back to next in family. Silence.
    import warnings
    warnings.filterwarnings("ignore", message="findfont")
    return plt


def load_events() -> pd.DataFrame:
    return pd.read_parquet(DATA_BUILD / "events.parquet")


# ---- results.json accumulation ----
def load_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    return {}

def save_results(d: dict):
    RESULTS.write_text(json.dumps(d, indent=2, default=_jsonify), encoding="utf-8")

def update_results(section: str, payload: dict):
    d = load_results()
    d[section] = payload
    save_results(d)
    return d

def _jsonify(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


QA_PATH = DATA_BUILD / "qa.md"

def write_qa_section(key: str, body: str):
    """Idempotently write a QA section keyed by `key` into data_build/qa.md.
    Sections are delimited by <!--qa:KEY--> ... <!--/qa:KEY-->; re-running a step
    REPLACES its block (clean re-run reproducibility). Sections are re-emitted
    sorted by key, so numeric key prefixes (00-, 05-, 10-, ...) fix the order
    regardless of the order steps run in.
    """
    import re as _re
    text = QA_PATH.read_text(encoding="utf-8") if QA_PATH.exists() else ""
    sections = {}
    for m in _re.finditer(r"<!--qa:([^>]+)-->\n?(.*?)\n?<!--/qa:\1-->", text, _re.S):
        sections[m.group(1)] = m.group(2).strip()
    sections[key] = body.strip()
    out = "\n\n".join(f"<!--qa:{k}-->\n{sections[k]}\n<!--/qa:{k}-->"
                      for k in sorted(sections))
    QA_PATH.write_text(out + "\n", encoding="utf-8")


def resample_clusters(cluster_ids, rng):
    """Cluster bootstrap: resample unique clusters WITH replacement, return the
    concatenated row-index array. A cluster drawn twice contributes its rows
    twice. Shared by the ZTNB (Step 3) and hazard (Step 4) bootstraps so the
    cluster logic is correct once. `cluster_ids` is an array aligned to rows.
    """
    cluster_ids = np.asarray(cluster_ids)
    uniq = np.unique(cluster_ids)
    # precompute row indices per cluster once per call
    idx_by = {c: np.where(cluster_ids == c)[0] for c in uniq}
    drawn = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([idx_by[c] for c in drawn])


def pctile_ci(draws, lo=2.5, hi=97.5):
    """Percentile CI from bootstrap draws (ignoring NaNs)."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(d, lo)), float(np.percentile(d, hi)))


def panel_label(ax, letter, dx=-0.02, dy=1.02):
    """Bold lowercase panel label at top-left (Nature convention)."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")


# US state name lookup (for optional labels)
STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'DC': 'D.C.', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming',
}

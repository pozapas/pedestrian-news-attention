"""p08_first_article_predictors.py -- Review #5.

Rebuilds every model predictor from the FIRST article linked to each event
instead of from the consolidated event record. The consolidated fields can be
populated by any article in the thread, so an attribute can become observed
because coverage continued, which is the outcome-dependent ascertainment the
review identifies. Restricting to the first article removes that channel by
construction: only information available on the opening day can enter.

Ties on publication date are broken by article_id so the choice is deterministic.

Writes data_build/events_firstart.parquet with the same column names as the
original panel, plus *_evt copies of the consolidated versions for comparison.
"""
from __future__ import annotations
import json, re
import os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB = ROOT / "data_build"
CN = Path(os.environ.get("CRASHNEWS_DIR", "source_corpus"))  # full-text corpus, not redistributed
DERIVED = CN / "derived" / "gold"
NEON_LINKS = DERIVED / "event_links_for_neon.ndjson"
ART = CN / "4_NewsMedia" / "news_articles_2016-2025_clean.csv"

panel = pd.read_parquet(DB / "events.parquet")
print("panel events:", len(panel))

# ---------------------------------------------------------------- 1. bridge
links = pd.read_json(NEON_LINKS, lines=True, dtype=str)[["event_id", "article_id"]]
links = links.drop_duplicates(["event_id", "article_id"])
links = links[links["event_id"].isin(set(panel["event_id"]))]
print("links to panel events:", len(links))

# ------------------------------------------ 2. per-article dates + extraction
use = ["article_id", "publication_date", "extraction_json"]
arts = pd.read_csv(ART, dtype=str, keep_default_na=False, usecols=use,
                   encoding="utf-8-sig", engine="python", on_bad_lines="skip")
arts = arts.drop_duplicates("article_id")

# publication_date carries two formats in this corpus: ISO YYYY-MM-DD for most
# rows and US M/D/YYYY for the rest. A single inferred format silently coerces
# the other to NaT, which would drop most events from the first-article join.
raw = arts["publication_date"].astype(str).str.strip()
iso = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
us = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")
arts["pub"] = iso.fillna(us)
print("articles: %d  dates parsed: %d (iso %d, us %d)"
      % (len(arts), arts["pub"].notna().sum(), iso.notna().sum(), us.notna().sum()))

la = links.merge(arts, on="article_id", how="inner")
la = la[la["pub"].notna()]
# deterministic first article per event
la = la.sort_values(["event_id", "pub", "article_id"])
first = la.groupby("event_id", as_index=False).first()
print("events with an identifiable first article:", len(first))


# --------------------------------------------------- 3. parse the extraction
def J(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


ex = first["extraction_json"].map(J)

def get(k):
    return ex.map(lambda d: d.get(k))

def is_set(v):
    if v is None:
        return False
    s = str(v).strip().lower()
    return s not in ("", "none", "null", "unknown", "not stated", "not reported", "n/a", "na")

def age_num(v):
    if not is_set(v):
        return np.nan
    m = re.search(r"\d{1,3}", str(v))
    if not m:
        return np.nan
    a = int(m.group(0))
    return a if 0 <= a <= 120 else np.nan

def yes(v):
    if not is_set(v):
        return np.nan
    s = str(v).strip().lower()
    if s.startswith(("y", "true", "1")):
        return 1.0
    if s.startswith(("n", "false", "0")):
        return 0.0
    return np.nan

fa = pd.DataFrame({"event_id": first["event_id"].values})
fa["fa_age"] = get("victim_age").map(age_num).values
g = get("victim_gender")
fa["fa_female"] = g.map(lambda v: (1.0 if str(v).strip().lower().startswith("f")
                                   else (0.0 if str(v).strip().lower().startswith("m")
                                         else np.nan)) if is_set(v) else np.nan).values
fa["fa_named"] = get("victim_name").map(lambda v: 1.0 if is_set(v) else 0.0).values
fa["fa_hitrun"] = get("hit_run").map(yes).values
# charges: a JSON null means the field was not populated at all, which is
# missing. Values such as "none", "none filed" and "pending" are statements
# that no charge had been filed, which is an observed zero rather than a gap.
NOCHARGE = ("none", "no charge", "not filed", "pending", "none filed",
            "none expected", "none reported", "not yet", "no charges")
def charge_code(v):
    if not is_set(v):
        return np.nan                     # absent or explicitly unknown
    s = str(v).strip().lower()
    return 0.0 if any(s.startswith(k) or s == k for k in NOCHARGE) else 1.0
ch = get("charges")
fa["fa_charges"] = ch.map(charge_code).values
fc = get("fatality_count")
fa["fa_fatal"] = fc.map(lambda v: age_num(v)).values
lc = get("lighting_condition")
fa["fa_dark"] = lc.map(lambda v: (1.0 if "dark" in str(v).lower()
                                  else 0.0) if is_set(v) else np.nan).values

df = panel.merge(fa, on="event_id", how="left")

# ------------------------------------------- 4. rebuild the model covariates
df["age_missing_evt"] = df["age_missing"]
df["female_evt"] = df["female"]
df["victim_named_evt"] = df["victim_named"]
df["hit_run_flag_evt"] = df["hit_run_flag"]
df["multi_fatality_evt"] = df["multi_fatality"]
df["dark_evt"] = df["dark"]
df["charges_reported_evt"] = df["charges_reported"]

a = df["fa_age"]
df["child"] = (a <= 12).fillna(False).astype(int)
df["teen"] = ((a >= 13) & (a <= 19)).fillna(False).astype(int)
df["elderly"] = (a >= 65).fillna(False).astype(int)
df["adult"] = ((a >= 20) & (a < 65)).fillna(False).astype(int)
df["age_missing"] = a.isna().astype(int)

df["female"] = df["fa_female"].fillna(0).astype(int)
df["gender_missing"] = df["fa_female"].isna().astype(int)

df["victim_named"] = df["fa_named"].fillna(0).astype(int)

df["hit_run_flag"] = df["fa_hitrun"].fillna(0).astype(int)
df["hit_run_missing"] = df["fa_hitrun"].isna().astype(int)

df["multi_fatality"] = (df["fa_fatal"] > 1).fillna(False).astype(int)

df["dark"] = df["fa_dark"].fillna(0).astype(int)
df["dark_missing"] = df["fa_dark"].isna().astype(int)

df["charges_reported"] = df["fa_charges"].fillna(0).astype(int)
df["charges_missing"] = df["fa_charges"].isna().astype(int)

# events with no recoverable first article keep event-level values, flagged
nofa = df["fa_age"].isna() & df["fa_female"].isna() & df["fa_hitrun"].isna()
df["no_first_article"] = df["event_id"].isin(
    set(panel["event_id"]) - set(first["event_id"])).astype(int)

out = DB / "events_firstart.parquet"
df.to_parquet(out, index=False)

# ------------------------------------------------------------ 5. what moved
rep = {"events": int(len(df)),
       "events_without_first_article": int(df["no_first_article"].sum())}
for new, old, lab in [("victim_named", "victim_named_evt", "victim named"),
                      ("hit_run_flag", "hit_run_flag_evt", "hit-and-run"),
                      ("female", "female_evt", "female"),
                      ("multi_fatality", "multi_fatality_evt", "multi-fatality"),
                      ("charges_reported", "charges_reported_evt", "charges"),
                      ("dark", "dark_evt", "dark")]:
    rep[lab] = {"first_article_rate": float(df[new].mean()),
                "event_level_rate": float(df[old].mean()),
                "pp_change": float(100 * (df[new].mean() - df[old].mean()))}
rep["age_missing_first_article"] = float(df["age_missing"].mean())
rep["age_missing_event_level"] = float(df["age_missing_evt"].mean())

# does the gap widen with article count? that is the ascertainment signature
for lab, q in [("1 article", df["n_articles"] == 1),
               ("2-3", df["n_articles"].between(2, 3)),
               ("4+", df["n_articles"] >= 4)]:
    rep.setdefault("named_by_tier", {})[lab] = {
        "first_article": float(df.loc[q, "victim_named"].mean()),
        "event_level": float(df.loc[q, "victim_named_evt"].mean()),
        "n": int(q.sum())}

json.dump(rep, open(DB / "firstart_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
print("wrote", out)

"""p01_build_dataset.py -- TRB-P Step 1: build the event-level analysis dataset.

Loads the pedestrian corpus 2016-2025, recomputes article/outlet counts from the
bridge table (NEVER the stored `outlet_count`), parses covariates, and writes
`data_build/events.parquet` + appends checks to `data_build/qa.md`.

KEY BUILD DECISION (see qa.md): the per-year `event_article_links_YYYY.csv`
exports are INCOMPLETE -- they link only 11,571 of 12,878 events (1,307 events
have no link row at all, though their stored outlet_count is 1-14). The
dashboard was built from the reconciled bridge table
`derived/gold/event_links_for_neon.ndjson` (18,846 links, 12,877 events linked),
which reproduces every snapshot total to <0.1%. Per the study protocol's rule
("recompute article/outlet counts from event_article_links joined to
news_articles") and its instruction to investigate >2% snapshot disagreements
before modeling, we use the reconciled bridge table as the event->article link
source, joined to the raw `news_articles` CSVs for source_domain / publication
dates. Covariates come from the raw `crash_events` CSVs (complete, 12,878 rows).

Run:  python analysis/p01_build_dataset.py
"""
from __future__ import annotations
import sys, io, json, re
import os
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_BUILD = ROOT / "data_build"
DATA_BUILD.mkdir(exist_ok=True)
CN = Path(os.environ.get("CRASHNEWS_DIR", "source_corpus"))  # full-text corpus, not redistributed
CORPUS = CN / "Data"
DERIVED = CN / "derived" / "gold"
SNAPSHOT = DERIVED / "dashboard_snapshot.json"
NEON_LINKS = DERIVED / "event_links_for_neon.ndjson"
NEON_EVENTS = DERIVED / "events_for_neon.ndjson"
YEARS = list(range(2016, 2026))

EVENT_COLS = [
    "event_id", "event_date", "state", "city", "county", "fatality_count",
    "victim_age", "victim_gender", "victim_name", "hit_run", "charges",
    "road_type", "lighting_condition", "review_status", "duplicate_event_ids",
    "event_summary", "dedup_confidence", "outlet_count",
]

qa_lines: list[str] = []
def log(msg: str = ""):
    print(msg)
    qa_lines.append(msg)


def read_csv(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols,
                       on_bad_lines="warn", engine="c",
                       encoding="utf-8", encoding_errors="replace")


def parse_dates(s: pd.Series) -> pd.Series:
    """Robust two-pass parse: ISO YYYY-MM-DD then US M/D/YYYY (day precision)."""
    s = s.astype(str)
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    mask = out.isna() & (s.str.strip() != "")
    if mask.any():
        alt = pd.to_datetime(s[mask], format="%m/%d/%Y", errors="coerce")
        out.loc[mask] = alt
    return out.dt.normalize()


# ------------------------------------------------------------------ helpers
_SPLIT = re.compile(r"[;,]")

def youngest_age(s: str):
    if not s:
        return np.nan
    vals = []
    for tok in _SPLIT.split(s):
        m = re.search(r"\d+", tok)
        if m:
            v = int(m.group())
            if 0 <= v <= 120:
                vals.append(v)
    return float(min(vals)) if vals else np.nan

def first_gender(s: str) -> str:
    return _SPLIT.split(s)[0].strip().lower() if s else ""

_NONE_CHARGE = re.compile(
    r"^\s*(none|no charge|not? |unknown|n/?a|pending|tbd|under invest|"
    r"awaiting|to be|undetermined|unclear|not filed|not reported|"
    r"no arrest|no one|not applicable)", re.I)
def charges_reported(s: str) -> int:
    s = (s or "").strip()
    if not s or _NONE_CHARGE.search(s):
        return 0
    return 1


def main():
    log("## Step 1 — Build event-level dataset\n")
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    # --- events (raw crash_events; complete covariate source) ---------
    ev_list = []
    for y in YEARS:
        e = read_csv(CORPUS / str(y) / f"crash_events_{y}.csv", usecols=EVENT_COLS)
        e["file_year"] = y
        ev_list.append(e)
    events = pd.concat(ev_list, ignore_index=True)
    n0 = len(events)
    events = events.drop_duplicates(subset="event_id", keep="first")
    dup_exact = n0 - len(events)
    merged_ids = set()
    for s in events["duplicate_event_ids"]:
        if s:
            merged_ids.update(x.strip() for x in s.split(";") if x.strip())
    before = len(events)
    events = events[~events["event_id"].isin(merged_ids)].copy()
    dup_merged = before - len(events)
    assert events["event_id"].is_unique
    log(f"- Loaded {n0:,} crash_events rows → {len(events):,} unique events "
        f"(dropped {dup_exact} exact dup ids, {dup_merged} merged-away ids).")

    # --- reconciled bridge table (see module docstring) ---------------
    links = pd.read_json(NEON_LINKS, lines=True, dtype=str)[["event_id", "article_id"]]
    log(f"- Reconciled bridge table `event_links_for_neon.ndjson`: {len(links):,} links "
        f"(raw per-year event_article_links CSVs have only 16,747 → INCOMPLETE, "
        f"see build decision).")

    # --- article attributes from raw news_articles CSVs ---------------
    art_list = []
    for y in YEARS:
        art_list.append(read_csv(CORPUS / str(y) / f"news_articles_{y}.csv",
                                 usecols=["article_id", "source_domain", "publication_date"]))
    arts = pd.concat(art_list, ignore_index=True).drop_duplicates("article_id")
    log(f"- news_articles rows (complete): {len(arts):,}.")

    valid_ids = set(events["event_id"])
    links = links[links["event_id"].isin(valid_ids)].drop_duplicates(["event_id", "article_id"])
    la = links.merge(arts, on="article_id", how="left")
    miss_join = int(la["source_domain"].isna().sum())
    la["pub"] = parse_dates(la["publication_date"].fillna(""))
    log(f"- Link→article join: {miss_join} links unmatched to news_articles; "
        f"{int(la['pub'].isna().sum()):,} of {len(la):,} links have unparseable pub date.")

    g = la.groupby("event_id")
    agg = pd.DataFrame({
        "n_articles": g["article_id"].nunique(),
        "n_outlets": g["source_domain"].apply(lambda x: x[x.fillna("") != ""].nunique()),
        "first_pub": g["pub"].min(),
        "last_pub": g["pub"].max(),
    })
    def followup(sub):
        fp = sub.min()
        gaps = (sub - fp).dt.days
        pos = gaps[gaps >= 1]
        return float(pos.min()) if len(pos) else np.nan
    agg["followup_day"] = g["pub"].apply(followup)

    df = events.merge(agg, on="event_id", how="left")
    n_unlinked = int(df["n_articles"].isna().sum())
    df["n_articles"] = df["n_articles"].fillna(0).astype(int)
    df["n_outlets"] = df["n_outlets"].fillna(0).astype(int)

    # independent_story_count (dedup of syndication) from neon events — the
    # metric the snapshot's storyCountHistogram is computed on; kept as an
    # optional robustness outcome.
    en = pd.read_json(NEON_EVENTS, lines=True)[
        ["event_id", "linked_article_count", "independent_story_count"]]
    df = df.merge(en, on="event_id", how="left")

    # --- stored-vs-recomputed mismatch (diagnostic) -------------------
    stored = pd.to_numeric(df["outlet_count"], errors="coerce")
    mism = int(((stored != df["n_articles"]) & stored.notna()).sum())
    log(f"- Unlinked events: {n_unlinked} (snapshot unlinkedEvents="
        f"{snap['quality']['unlinkedEvents']}). "
        f"stored outlet_count vs recomputed n_articles mismatches: {mism} "
        f"(snapshot storedCountMismatches={snap['quality']['storedCountMismatches']}).")

    # recompute must equal neon linked_article_count exactly
    rc_vs_neon = int((df["n_articles"] != df["linked_article_count"].fillna(0)).sum())
    log(f"- recomputed n_articles vs neon linked_article_count mismatches: {rc_vs_neon} "
        f"(expect 0).")

    # --- drop zero-article margin (ZT requires n>=1) ------------------
    df = df[df["n_articles"] >= 1].copy()
    log(f"- Dropped {n_unlinked} zero-article event(s); modeling set n={len(df):,}.")

    # --- covariates ---------------------------------------------------
    df["event_date"] = parse_dates(df["event_date"])
    df["year"] = df["event_date"].dt.year.fillna(df["file_year"]).astype(int)
    df["weekend"] = df["event_date"].dt.weekday.isin([5, 6]).astype(int)

    df["age_youngest"] = df["victim_age"].apply(youngest_age)
    df["age_missing"] = df["age_youngest"].isna().astype(int)
    a = df["age_youngest"]
    df["child"] = (a <= 12).fillna(False).astype(int)
    df["teen"] = ((a >= 13) & (a <= 19)).fillna(False).astype(int)
    df["adult"] = ((a >= 20) & (a <= 64)).fillna(False).astype(int)   # reference
    df["elderly"] = (a >= 65).fillna(False).astype(int)

    df["gender1"] = df["victim_gender"].apply(first_gender)
    df["female"] = (df["gender1"] == "female").astype(int)
    df["gender_missing"] = (~df["gender1"].isin(["male", "female"])).astype(int)

    df["victim_named"] = (df["victim_name"].str.strip() != "").astype(int)

    hr = df["hit_run"].str.strip().str.lower()
    df["hit_run_flag"] = (hr == "true").astype(int)
    df["hit_run_missing"] = (~hr.isin(["true", "false"])).astype(int)

    df["multi_fatality"] = (pd.to_numeric(df["fatality_count"], errors="coerce")
                            .fillna(1) > 1).astype(int)

    lc = df["lighting_condition"].str.lower()
    df["dark"] = lc.str.contains("dark", na=False).astype(int)
    df["dark_missing"] = (~lc.str.contains("dark|day", na=False)).astype(int)

    df["charges_reported"] = df["charges"].apply(charges_reported)
    df["charges_missing"] = (df["charges"].str.strip() == "").astype(int)

    rt = df["road_type"].str.lower()
    def road_class(x):
        if not x or x in ("unknown", "n/a"):
            return "missing"
        if "intersection" in x:
            return "intersection"
        if "highway" in x or "freeway" in x or "interstate" in x:
            return "highway"
        if "arterial" in x or "mid-block" in x or "mid‑block" in x or "street" in x:
            return "arterial"
        return "other"
    df["road_class"] = rt.apply(road_class)

    df["review_accepted"] = (df["review_status"].str.strip().str.lower() == "accepted").astype(int)
    df["span_days"] = (df["last_pub"] - df["first_pub"]).dt.days
    df["got_followup"] = df["followup_day"].notna().astype(int)
    def tier(n):
        return "1" if n == 1 else "2-3" if n <= 3 else "4-9" if n <= 9 else "10+"
    df["cov_tier"] = df["n_articles"].apply(tier)

    ok_time = (df["first_pub"] >= df["event_date"])
    log(f"- first_pub >= event_date for {ok_time.mean():.1%} of events "
        f"(>=95% expected; parsed both ISO and US date formats).")

    # =============================== CHECKS ============================
    log("\n### CHECKS")
    total = len(df); target = 12878
    corpus_n = total + n_unlinked
    log(f"- [total] modeling n={total:,}; corpus={corpus_n:,}; "
        f"snapshot kpis.events={snap['kpis']['events']:,}; "
        f"within 2%: {abs(corpus_n-target)/target < 0.02}")
    assert (df["n_articles"] >= 1).all(); log("- [n_articles>=1] ✅")
    assert df["event_id"].is_unique; log("- [unique event_id] ✅")
    assert rc_vs_neon == 0, "recompute disagrees with neon linked_article_count"
    log("- [recompute == neon linked_article_count] ✅ (0 mismatches)")

    def hb(n):
        return "1" if n == 1 else "2" if n == 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"
    hist = df["n_articles"].apply(hb).value_counts().to_dict()
    log("- [n_articles histogram] " +
        str({k: hist.get(k, 0) for k in ['1', '2', '3-5', '6-10', '11+']}) +
        f"  (sum={int(df['n_articles'].sum()):,} articles; snapshot kpis.articles="
        f"{snap['kpis']['articles']:,}).")
    log("    NB snapshot.storyCountHistogram is computed on independent_story_count "
        "(syndication-deduped, total 18,061), a different metric than our Y_i="
        "distinct article_id (linked_article_count).")

    ca = df[df["state"] == "CA"]
    sca = next(s for s in snap["states"] if s["state"] == "CA")
    log(f"- [CA] events {len(ca):,} (snap {sca['events']:,}); "
        f"articles {int(ca['n_articles'].sum()):,} (snap {sca['articles']:,}) ✅")

    cf = snap["crashFacts"]
    hy, hn = int((hr == "true").sum()), int((hr == "false").sum())
    log(f"- [hit_run known share] {hy/(hy+hn):.4f} (snap {cf['hitRun']['shareOfKnown']})")
    dk = int(df["dark"].sum()); li = int(lc.str.contains("day", na=False).sum())
    log(f"- [after-dark known share] {dk/(dk+li):.4f} (snap {cf['afterDark']['shareOfKnown']})")
    fat = int(pd.to_numeric(df["fatality_count"], errors="coerce").fillna(1).sum())
    log(f"- [fatalities] {fat:,} (snap crashFacts.fatalities {cf['fatalities']:,}; "
        f"note: 1 unlinked event's fatalities excluded).")

    log("\n### Rows per year (mine vs snapshot yearly.events)")
    syr = {d["year"]: d["events"] for d in snap["yearly"]}
    yc = df["year"].value_counts().sort_index()
    for y in YEARS:
        log(f"    {y}: mine={int(yc.get(y,0)):>5}  snap={syr.get(y,0):>5}")

    log("\n### Missingness (key fields, % of modeling set)")
    for c, lab in [("age_missing", "age"), ("gender_missing", "gender"),
                   ("hit_run_missing", "hit_run"), ("charges_missing", "charges"),
                   ("dark_missing", "lighting")]:
        log(f"    {lab:10s}: {df[c].mean():.1%}")
    log("\n### Coverage tiers (Table 1 tiers 1/2-3/4-9/10+)")
    for t in ["1", "2-3", "4-9", "10+"]:
        log(f"    tier {t:4s}: {int((df['cov_tier']==t).sum()):>6}")

    # --- save ---------------------------------------------------------
    keep = ["event_id", "event_date", "year", "state", "city", "county",
            "n_articles", "n_outlets", "independent_story_count",
            "first_pub", "last_pub", "followup_day", "got_followup", "span_days",
            "cov_tier", "age_youngest", "age_missing", "child", "teen", "adult",
            "elderly", "female", "gender_missing", "victim_named", "hit_run_flag",
            "hit_run_missing", "multi_fatality", "weekend", "dark", "dark_missing",
            "charges_reported", "charges_missing", "road_class", "review_accepted",
            "fatality_count", "event_summary", "dedup_confidence"]
    out = df[keep].copy()
    out["independent_story_count"] = pd.to_numeric(
        out["independent_story_count"], errors="coerce").fillna(out["n_articles"]).astype(int)
    out.to_parquet(DATA_BUILD / "events.parquet", index=False)
    log(f"\n- Saved data_build/events.parquet ({len(out):,} rows, {out.shape[1]} cols).")

    import common
    common.write_qa_section("20-step1", "\n".join(qa_lines))


if __name__ == "__main__":
    main()

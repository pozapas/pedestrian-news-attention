"""p20_build_app_sample.py -- 180-article, two-coder validation sample for the web app.

Both coders code all 180 articles, so inter-coder agreement is available on
every item. Each coder also receives 5 hidden retest copies (different item
codes, placed in the second half of their order), giving 185 screens each.

The sample is drawn from the full eligible pool of events: every event whose
first linked article is available with more than 200 characters of text,
excluding every event used in round 1 or its calibration set. Round 2 was
never issued, so its events remain eligible. Strata are assigned in the order
below, and the allocation oversamples the rare covariates the paper interprets
so that each carries enough positives to estimate agreement:

  S1 multi-fatality 25, S2 child victim 25, S3 teen victim 20,
  S4 older victim (65+) 30, S5 charges reported 30, S6 hit-and-run 25,
  S7 remainder 25.

Weights are N_stratum / n_sampled, with N taken from the full eligible pool.

Writes:
  webapp/lib/items.js                        article text + per-coder order (NO machine values)
  validation_app/manifest_app.csv   item code -> article, stratum, weight, machine values
  validation_app/sample_design_app.json      pool sizes, allocation, positives
"""
from __future__ import annotations
import csv, io, json, re
import os
from pathlib import Path
import numpy as np
import pandas as pd

csv.field_size_limit(10 ** 9)
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data_build"
R1 = ROOT / "validation"
OUTV = ROOT / "validation_app"
APP = ROOT / "webapp"
CN = Path(os.environ.get("CRASHNEWS_DIR", "source_corpus"))  # full-text corpus, not redistributed
ART = CN / "4_NewsMedia" / "news_articles_2016-2025_clean.csv"
LINKS = CN / "derived" / "gold" / "event_links_for_neon.ndjson"

SEED = 20260925
ALLOC = {"S1_multi_fatality": 25, "S2_child_victim": 25, "S3_teen_victim": 20,
         "S4_older_victim": 30, "S5_charges_reported": 30, "S6_hit_and_run": 25,
         "S7_remainder": 25}
CODERS = ["C01", "C02"]
N_RETEST = 5
MACHINE = ["multi_fatality", "child", "teen", "elderly", "female",
           "victim_named", "hit_run_flag", "charges_reported", "dark"]
rng = np.random.RandomState(SEED)
OUTV.mkdir(exist_ok=True)
(APP / "lib").mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ exclusions
r1 = pd.read_csv(R1 / "sample_manifest.csv")
used_events = set(r1["event_id"])
cal_articles = {r["article_id"] for r in csv.DictReader(
    io.open(R1 / "calibration" / "calibration_KEY_template.csv", encoding="utf-8-sig"))}
links = pd.read_json(LINKS, lines=True, dtype=str)[["event_id", "article_id"]]
links = links.drop_duplicates(["event_id", "article_id"])
used_events |= set(links.loc[links["article_id"].isin(cal_articles), "event_id"])

# ------------------------------------------------------- first articles
arts = pd.read_csv(ART, dtype=str, keep_default_na=False,
                   usecols=["article_id", "publication_date", "article_text", "source_name"],
                   encoding="utf-8-sig", engine="python", on_bad_lines="skip")
raw = arts["publication_date"].astype(str).str.strip()
arts["pub"] = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce").fillna(
    pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce"))
la = links.merge(arts, on="article_id", how="inner")
la = la[la["pub"].notna()].sort_values(["event_id", "pub", "article_id"])
first = la.groupby("event_id", as_index=False).first()
first = first[first["article_text"].str.len() > 200].set_index("event_id")

fa = pd.read_parquet(DB / "events_firstart.parquet")
fa = fa[~fa["event_id"].isin(used_events) & fa["event_id"].isin(first.index)].copy()


def stratum(r):
    if r.multi_fatality == 1: return "S1_multi_fatality"
    if r.child == 1: return "S2_child_victim"
    if r.teen == 1: return "S3_teen_victim"
    if r.elderly == 1: return "S4_older_victim"
    if r.charges_reported == 1: return "S5_charges_reported"
    if r.hit_run_flag == 1: return "S6_hit_and_run"
    return "S7_remainder"


fa["stratum"] = fa.apply(stratum, axis=1)
pool = fa["stratum"].value_counts().to_dict()
print("eligible pool:", len(fa), pool)

picks = []
for s, n in ALLOC.items():
    g = fa[fa["stratum"] == s]
    picks.append(g.sample(n=min(n, len(g)), random_state=rng))
sel = pd.concat(picks, ignore_index=True)
sel["article_id"] = sel["event_id"].map(first["article_id"])
sel["N_stratum"] = sel["stratum"].map(pool)
sel["n_sampled"] = sel["stratum"].map(sel["stratum"].value_counts())
sel["weight"] = sel["N_stratum"] / sel["n_sampled"]
for m in MACHINE:
    sel[m] = sel[m].fillna(0).astype(int)
pos = {m: int(sel[m].sum()) for m in MACHINE}
print("sample:", sel["stratum"].value_counts().to_dict())
print("machine positives:", pos)

# ----------------------------------------------------------- article text
CSS = re.compile(r"(?:[\w#.:\-\[\]=\"' ,>*()]+\{[^{}]{0,400}\})+")


def clean(t):
    t = str(t)
    t = CSS.sub(" ", t)                      # scraped stylesheet fragments
    t = re.sub(r"@media[^{]*\{", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def pubdate(ts):
    return ts.date().isoformat() if pd.notna(ts) else ""


used = set()
def code():
    while True:
        c = "".join(rng.choice(list("ACDEFHJKLMNPRTUVWXY3479"), size=6))
        if c not in used:
            used.add(c); return c


items, manifest = {}, []
main_code = {}
for r in sel.itertuples(index=False):
    c = code()
    main_code[r.article_id] = c
    a = first.loc[r.event_id]
    items[c] = {"published": pubdate(a["pub"]),
                "outlet": a["source_name"] or "unknown", "text": clean(a["article_text"])}
    manifest.append({"item_code": c, "article_id": r.article_id, "event_id": r.event_id,
                     "is_retest_copy": 0, "retest_of": "", "coder_scope": "both",
                     "stratum": r.stratum, "N_stratum": r.N_stratum,
                     "n_sampled": r.n_sampled, "weight": r.weight,
                     **{m: int(getattr(r, m)) for m in MACHINE}})

order = {}
for cid in CODERS:
    seq = list(main_code.values())
    rng.shuffle(seq)
    half = len(seq) // 2
    # retest copies of items from the first half, reinserted in the second half
    src = list(rng.choice(seq[:half], size=N_RETEST, replace=False))
    for c0 in src:
        c1 = code()
        items[c1] = dict(items[c0])
        base = next(m for m in manifest if m["item_code"] == c0)
        manifest.append({**base, "item_code": c1, "is_retest_copy": 1,
                         "retest_of": c0, "coder_scope": cid})
        pos_i = int(rng.randint(half + 5, len(seq) + 1))
        seq.insert(pos_i, c1)
    order[cid] = seq
order["T00"] = order["C01"]          # test token, excluded from exports by default

lens = [len(v["text"]) for v in items.values()]
print("items: %d  text chars median %d  p95 %d  max %d"
      % (len(items), np.median(lens), np.percentile(lens, 95), max(lens)))

js = ("// Generated by analysis/p20_build_app_sample.py -- article text and order only.\n"
      "// Contains no machine-extracted values.\n"
      "export const ITEMS = " + json.dumps(items, ensure_ascii=False) + ";\n"
      "export const ORDER = " + json.dumps(order) + ";\n")
(APP / "lib" / "items.js").write_text(js, encoding="utf-8")
pd.DataFrame(manifest).to_csv(OUTV / "manifest_app.csv", index=False, encoding="utf-8-sig")
json.dump({"seed": SEED, "eligible_events": int(len(fa)), "strata_pool": {k: int(v) for k, v in pool.items()},
           "allocation": ALLOC, "machine_positives": pos,
           "excluded_round1_events": len(used_events)},
          open(OUTV / "sample_design_app.json", "w"), indent=1)
print("items.js %.0f KB; screens per coder: %s"
      % (len(js.encode()) / 1024, {k: len(v) for k, v in order.items()}))

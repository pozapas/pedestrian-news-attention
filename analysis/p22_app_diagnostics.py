"""p22_app_diagnostics.py -- provenance and quality diagnostics for the app coding.

Reads validation_app/returns/coding_export.csv (written by p21 --fetch) and
reports work sessions, pacing, focus losses, retest pairs, and the coder-by-coder
cross-tabulations for the fields with the lowest agreement.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "validation_app"
df = pd.read_csv(V / "returns" / "coding_export.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
df = df[df["coder_id"].isin(["C01", "C02"])].copy()
man = pd.read_csv(V / "manifest_app.csv")
MACH = ["multi_fatality", "child", "teen", "elderly", "female", "victim_named",
        "hit_run_flag", "charges_reported", "dark"]
man = man.rename(columns={m: "m_" + m for m in MACH})
df = df.merge(man, on="item_code", how="left")
for c in ["seconds_on_item", "first_seconds_on_item", "copy_events", "focus_losses", "revisions"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
df["served"] = pd.to_datetime(df["served_at"], utc=True)
df["submitted"] = pd.to_datetime(df["submitted_at"], utc=True)

print("=== sessions (gap > 20 min starts a new session), times UTC ===")
for c, g in df.sort_values("submitted").groupby("coder_id"):
    t = g["submitted"].sort_values().reset_index(drop=True)
    new = t.diff().dt.total_seconds().fillna(1e9) > 1200
    sid = new.cumsum()
    for s, tt in t.groupby(sid):
        print("  %s session %d: %s -> %s  items %3d  span %5.0f min"
              % (c, s, tt.iloc[0].strftime("%m-%d %H:%M"), tt.iloc[-1].strftime("%m-%d %H:%M"),
                 len(tt), (tt.iloc[-1] - tt.iloc[0]).total_seconds() / 60))

print("\n=== pacing and activity ===")
for c, g in df.groupby("coder_id"):
    s = g["first_seconds_on_item"]
    fl = g["focus_losses"]
    print("  %s seconds p10/p25/median/p75/p90: %s" % (c, "/".join(str(int(x)) for x in s.quantile([.1, .25, .5, .75, .9]))))
    print("     focus losses per item: 0=%d 1=%d 2=%d 3+=%d   copies on %d items, revisions %d"
          % ((fl == 0).sum(), (fl == 1).sum(), (fl == 2).sum(), (fl >= 3).sum(),
             (g["copy_events"] > 0).sum(), (g["revisions"] > 0).sum()))
    print("     median seconds with no focus loss %d, with focus loss %d"
          % (s[fl == 0].median(), s[fl > 0].median()))
    ch = g["text_len"] if "text_len" in g else None

# reading speed against article length
import json, re
items = json.loads(re.search(r"export const ITEMS = (\{.*?\});\nexport const ORDER",
                   (ROOT / "webapp" / "lib" / "items.js").read_text(encoding="utf-8"), re.S).group(1))
df["chars"] = df["item_code"].map(lambda k: len(items.get(k, {}).get("text", "")))
print("\n=== time against article length (Spearman) ===")
for c, g in df.groupby("coder_id"):
    print("  %s rho(seconds, chars) = %.2f   words per minute at median = %.0f"
          % (c, g["first_seconds_on_item"].corr(g["chars"], method="spearman"),
             (g["chars"].median() / 5.5) / (g["first_seconds_on_item"].median() / 60)))

FIELDS = ["is_pedestrian_fatal_crash", "n_pedestrians_killed", "victim_age", "victim_gender",
          "victim_named", "driver_fled", "charges_mentioned", "lighting"]
print("\n=== retest pairs (original | copy), differing fields only ===")
for c, g in df.groupby("coder_id"):
    for _, r in g[g["is_retest_copy"] == 1].iterrows():
        o = g[(g["article_id"] == r["article_id"]) & (g["is_retest_copy"] == 0)].iloc[0]
        diff = {f: (o[f], r[f]) for f in FIELDS if o[f] != r[f]}
        print("  %s %s idx %s->%s: %s" % (c, r["article_id"], o["index"], r["index"], diff or "identical"))

w = {c: df[(df["coder_id"] == c) & (df["is_retest_copy"] == 0)].set_index("article_id") for c in ["C01", "C02"]}
both = w["C01"].index.intersection(w["C02"].index)
yes = [a for a in both if w["C01"].loc[a, "is_pedestrian_fatal_crash"] == "yes" == w["C02"].loc[a, "is_pedestrian_fatal_crash"]]
for f in ["driver_fled", "charges_mentioned", "lighting", "victim_gender"]:
    print("\n=== %s: rows C01, columns C02 ===" % f)
    print(pd.crosstab(w["C01"].loc[yes, f], w["C02"].loc[yes, f], margins=True).to_string())
print("\n=== is_pedestrian_fatal_crash: rows C01, columns C02 ===")
print(pd.crosstab(w["C01"].loc[both, "is_pedestrian_fatal_crash"], w["C02"].loc[both, "is_pedestrian_fatal_crash"]).to_string())

print("\n=== machine flag against each coder, charges and hit-and-run ===")
for f, m, pos in [("charges_mentioned", "charges_reported", "filed"), ("driver_fled", "hit_run_flag", "yes"),
                  ("lighting", "dark", "dark")]:
    for c in ["C01", "C02"]:
        g = w[c].loc[yes]
        print("  %s %s:" % (f, c), pd.crosstab(g["m_" + m], g[f]).to_dict())

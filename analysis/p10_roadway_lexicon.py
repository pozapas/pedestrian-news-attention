"""p10_roadway_lexicon.py -- Review #18.

Specifies the roadway-element content measure that previously appeared only as
a Discussion assertion. Reports the codebook, the usable-text rule, the
excluded-article analysis, and a narrow/broad sensitivity band in place of the
human validation that a lexical measure of this kind does not carry.

Counting rule: an article contributes at most once to each theme regardless of
how many times a term occurs, and once to ANY if it matches at least one theme.

Writes data_build/roadway_lexicon.json.
"""
from __future__ import annotations
import csv, io, json, re
import os
from pathlib import Path
import numpy as np
import pandas as pd

csv.field_size_limit(10 ** 9)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB = ROOT / "data_build"
CN = Path(os.environ.get("CRASHNEWS_DIR", "source_corpus"))  # full-text corpus, not redistributed
ART = CN / "4_NewsMedia" / "news_articles_2016-2025_clean.csv"

MIN_CHARS = 200          # "usable text" threshold, stated rather than implicit

# Narrow codebook: the element must be named explicitly.
NARROW = {
    "Crosswalk":            r"\bcross-?walks?\b",
    "Speed limit":          r"\bspeed limit\b|\bposted speed\b",
    "Traffic signal":       r"\btraffic (?:signal|light)s?\b|\bstop ?light\b",
    "Street lighting":      r"\bstreet ?light\w*\b|\bstreet ?lamp\w*\b",
    "Stop or yield control": r"\bstop sign\b|\byield sign\b",
    "Roadway design":       r"\broad(?:way)? design\b|\broad diet\b|\btraffic calming\b",
    "Sidewalk":             r"\bsidewalks?\b",
    "Median or refuge":     r"\bmedian (?:island|strip|barrier)\b|\brefuge island\b",
}

# Broad codebook: adds oblique references and condition language.
BROAD = dict(NARROW)
BROAD.update({
    "Crosswalk":            r"\bcross-?walks?\b|\bzebra cross\w*|\bmarked crossing\b|\bunmarked crossing\b",
    "Speed limit":          r"\bspeed limit\b|\bposted speed\b|\bmph zone\b|\bspeeding zone\b",
    "Traffic signal":       r"\btraffic (?:signal|light)s?\b|\bstop ?light\b|\bsignali[sz]ed\b|\bpedestrian signal\b",
    "Street lighting":      r"\bstreet ?light\w*\b|\bstreet ?lamp\w*\b|\bpoorly lit\b|\bunlit\b|\bno lighting\b|\bdark stretch\b",
    "Roadway design":       r"\broad(?:way)? design\b|\broad diet\b|\btraffic calming\b|\blane width\b|\bdesign of the (?:road|street)\b|\bengineering\b",
    "Median or refuge":     r"\bmedian\b|\brefuge island\b",
})


def scan(codebook):
    rx = {k: re.compile(v, re.I) for k, v in codebook.items()}
    hits = {k: 0 for k in rx}
    n_any = 0
    n_use = 0
    n_short = 0
    lens = []
    with io.open(ART, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            t = row.get("article_text") or ""
            lens.append(len(t.strip()))
            if len(t.strip()) < MIN_CHARS:
                n_short += 1
                continue
            n_use += 1
            any_hit = False
            for k, r in rx.items():
                if r.search(t):
                    hits[k] += 1          # once per article per theme
                    any_hit = True
            if any_hit:
                n_any += 1
    return hits, n_any, n_use, n_short, lens


hits_n, any_n, use_n, short_n, lens = scan(NARROW)
hits_b, any_b, use_b, short_b, _ = scan(BROAD)

out = {
    "min_chars_usable": MIN_CHARS,
    "articles_total": use_n + short_n,
    "articles_usable": use_n,
    "articles_below_threshold": short_n,
    "pct_below_threshold": 100.0 * short_n / (use_n + short_n),
    "median_text_length": float(np.median(lens)),
    "narrow": {"themes": {k: 100.0 * v / use_n for k, v in hits_n.items()},
               "any_pct": 100.0 * any_n / use_n,
               "none_pct": 100.0 * (use_n - any_n) / use_n},
    "broad": {"themes": {k: 100.0 * v / use_b for k, v in hits_b.items()},
              "any_pct": 100.0 * any_b / use_b,
              "none_pct": 100.0 * (use_b - any_b) / use_b},
    "codebook_narrow": NARROW,
    "codebook_broad": BROAD,
}
json.dump(out, open(DB / "roadway_lexicon.json", "w"), indent=1)

print("articles %d | usable %d | below %d chars: %d (%.1f%%) | median length %.0f"
      % (out["articles_total"], use_n, MIN_CHARS, short_n,
         out["pct_below_threshold"], out["median_text_length"]))
print()
print("%-24s %9s %9s" % ("theme", "narrow", "broad"))
for k in NARROW:
    print("  %-22s %8.1f%% %8.1f%%" % (k, out["narrow"]["themes"][k],
                                       out["broad"]["themes"][k]))
print("  %-22s %8.1f%% %8.1f%%" % ("ANY element", out["narrow"]["any_pct"],
                                   out["broad"]["any_pct"]))
print("  %-22s %8.1f%% %8.1f%%" % ("NO element", out["narrow"]["none_pct"],
                                   out["broad"]["none_pct"]))
print("wrote", DB / "roadway_lexicon.json")

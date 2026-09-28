"""p21_score_app.py -- score the two-coder web-app validation.

  python analysis/p21_score_app.py --fetch
      download the current export from the app (admin key in the APP_ADMIN_KEY environment variable)
      and score it
  python analysis/p21_score_app.py
      score validation_app/returns/coding_export.csv as it is

Reports, in order:
  pacing       per-coder seconds per item (first attempt), items under 30 s,
               active days, coder statement
  inter-coder  Cohen's kappa and raw agreement for every field, on the 180
               articles both coders saw (retest copies excluded)
  retest       per-coder agreement between an article and its hidden copy
  machine      machine covariates against the human reference, on the paper's
               own definitions (reported = 1, not stated = 0), weighted by the
               stratum weights, separately against each coder and against the
               consensus

The consensus is the shared answer when the two coders agree, including
answers that differ only in ways no model covariate uses (for example "no"
against "not stated" on driver_fled). Where a disagreement changes a
covariate, the adjudicator's answer from returns/ADJUDICATE_ME.xlsx (written by
p23_make_adjudication.py, column YOUR ANSWER) is used; until then that article
is left out of the consensus comparison for that covariate.
"""
from __future__ import annotations
import argparse, io, json, os, sys, urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation_common import FIELDS, indicators, weighted_binary, norm  # noqa: E402
from p23_make_adjudication import bkey, read_adjudication  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = Path(os.environ.get("APP_DIR", ROOT / "validation_app"))
RET = V / "returns"
URL = "https://ped-news-coding.vercel.app/api/export"
COVS = ["victim_named", "hit_run_flag", "multi_fatality", "female",
        "child", "teen", "elderly", "dark", "charges_reported"]
CODERS = ["C01", "C02"]


def fetch():
    key = os.environ["APP_ADMIN_KEY"]
    RET.mkdir(exist_ok=True)
    for fmt, name in (("", "coding_export.csv"), ("&format=summary", "summary.json")):
        with urllib.request.urlopen(f"{URL}?admin={key}{fmt}") as r:
            (RET / name).write_bytes(r.read())
    print("downloaded export to", RET)


def cohen(a, b):
    a, b = list(a), list(b)
    if not a:
        return float("nan"), float("nan")
    po = np.mean([x == y for x, y in zip(a, b)])
    ca, cb, n = Counter(a), Counter(b), len(a)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (float((po - pe) / (1 - pe)) if pe < 1 else float("nan")), float(po)


B_REPS = 2000


def sboot(stat, strata, seed=20260926):
    """Percentile 95% interval of stat(index array), resampling articles with
    replacement within each sampling stratum, as the design was drawn."""
    rng = np.random.RandomState(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    vals = []
    for _ in range(B_REPS):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < B_REPS // 2:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def fmt_ci(v, ci, pct=False):
    k = 100 if pct else 1
    f = "%5.1f" if pct else "%.2f"
    if not np.isfinite(v):
        return "   n/a" + " " * 14
    return (f % (k * v)) + " [" + (f % (k * ci[0])).strip() + "-" + (f % (k * ci[1])).strip() + "]"


def main():
    p = RET / "coding_export.csv"
    if not p.exists():
        print("No export yet. Run with --fetch.")
        return
    df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df[df["coder_id"].isin(CODERS)]
    if df.empty:
        print("The export has no coder answers yet.")
        return
    for f in FIELDS:
        df[f] = df[f].map(norm)
    df["secs"] = pd.to_numeric(df["first_seconds_on_item"], errors="coerce")
    man = pd.read_csv(V / "manifest_app.csv")
    df = df.merge(man[["item_code", "article_id", "is_retest_copy"]], on="item_code", how="left")
    out = {}

    # ---------------------------------------------------------------- pacing
    print("=== pacing ===")
    summ = json.load(open(RET / "summary.json")) if (RET / "summary.json").exists() else {}
    pace = {}
    for c, g in df.groupby("coder_id"):
        s = g["secs"].dropna()
        days = pd.to_datetime(g["submitted_at"]).dt.date.nunique()
        pace[c] = {"answered": int(len(g)), "median_s": float(s.median()),
                   "p10_s": float(s.quantile(.1)), "under_30s": int((s < 30).sum()),
                   "active_days": int(days),
                   "statement": bool(summ.get(c, {}).get("statement_submitted", False))}
        print("  %s answered %3d  median %4.0f s  p10 %4.0f s  <30 s: %3d  days %d  statement %s"
              % (c, pace[c]["answered"], pace[c]["median_s"], pace[c]["p10_s"],
                 pace[c]["under_30s"], days, pace[c]["statement"]))
    out["pacing"] = pace

    orig = df[df["is_retest_copy"] == 0]
    wide = {c: orig[orig["coder_id"] == c].set_index("article_id") for c in CODERS}
    both = sorted(set(wide["C01"].index) & set(wide["C02"].index))

    # ----------------------------------------------------------- inter-coder
    print("\n=== inter-coder agreement, %d articles coded by both (95%% CI, stratified bootstrap) ==="
          % len(both))
    irr = {}
    stratum_of = man[man["is_retest_copy"] == 0].drop_duplicates("article_id").set_index("article_id")["stratum"]
    for f in FIELDS:
        idx = both if f == "is_pedestrian_fatal_crash" else [
            a for a in both if wide["C01"].loc[a, "is_pedestrian_fatal_crash"] == "yes"
            and wide["C02"].loc[a, "is_pedestrian_fatal_crash"] == "yes"]
        a1 = wide["C01"].loc[idx, f].to_numpy()
        a2 = wide["C02"].loc[idx, f].to_numpy()
        k, po = cohen(a1, a2)
        ci = sboot(lambda ii: cohen(a1[ii], a2[ii])[0], stratum_of.loc[idx].to_numpy())
        irr[f] = {"kappa": k, "kappa_ci": ci, "agreement": po, "n": len(idx)}
        print("  %-26s n=%3d  agreement %5.1f%%  kappa %s" % (f, len(idx), 100 * po, fmt_ci(k, ci)))
    out["inter_coder"] = irr

    # ---------------------------------------------------------------- retest
    print("\n=== test-retest ===")
    rt = {}
    for c, g in df.groupby("coder_id"):
        pairs = []
        for _, r in g[g["is_retest_copy"] == 1].iterrows():
            o = g[(g["article_id"] == r["article_id"]) & (g["is_retest_copy"] == 0)]
            if len(o):
                pairs.append(np.mean([o.iloc[0][f] == r[f] for f in FIELDS]))
        rt[c] = {"pairs": len(pairs), "agreement": float(np.mean(pairs)) if pairs else float("nan")}
        print("  %s pairs=%d  field agreement %.1f%%" % (c, len(pairs), 100 * rt[c]["agreement"]))
    out["test_retest"] = rt

    # ------------------------------------------------------------- consensus
    # the adjudicator's answers, from the sheet written by p23_make_adjudication.py
    adj, bad = read_adjudication(RET / "ADJUDICATE_ME.xlsx")
    for b in bad:
        print("WARNING: adjudication %s is not an allowed answer; ignored" % b)
    if adj or bad:
        print("\nadjudicated answers read: %d" % len(adj))
    cons, todo = {}, []
    for a in both:
        h = {}
        f1 = wide["C01"].loc[a, "is_pedestrian_fatal_crash"]
        f2 = wide["C02"].loc[a, "is_pedestrian_fatal_crash"]
        for f in FIELDS:
            v1, v2 = wide["C01"].loc[a, f], wide["C02"].loc[a, f]
            if f != "is_pedestrian_fatal_crash" and f1 != f2:
                # only the coder who answered yes coded the details
                v1 = v2 = v1 if f1 == "yes" else v2
            if v1 == v2 or bkey(f, v1) == bkey(f, v2):
                # identical, or different only in ways no model covariate uses
                h[f] = v1
            elif (a, f) in adj:
                h[f] = adj[(a, f)]
            else:
                h[f] = None
                todo.append({"article_id": a, "field": f, "C01": v1, "C02": v2,
                             "item_code_C01": wide["C01"].loc[a, "item_code"],
                             "notes_C01": wide["C01"].loc[a, "coder_notes"],
                             "notes_C02": wide["C02"].loc[a, "coder_notes"], "value": ""})
        cons[a] = h
    print("unresolved disagreements that change a covariate: %d%s"
          % (len(todo), " (answer them in returns/ADJUDICATE_ME.xlsx)" if todo else ""))

    # --------------------------------------------------------------- machine
    ev = man[man["is_retest_copy"] == 0].drop_duplicates("article_id").set_index("article_id")
    needs = {"victim_named": ["victim_named"], "hit_run_flag": ["driver_fled"],
             "multi_fatality": ["n_pedestrians_killed"], "female": ["victim_gender"],
             "child": ["victim_age"], "teen": ["victim_age"], "elderly": ["victim_age"],
             "dark": ["lighting"], "charges_reported": ["charges_mentioned"]}
    comp = {}
    refs = {c: {a: wide[c].loc[a, FIELDS].to_dict() for a in wide[c].index} for c in CODERS}
    refs["consensus"] = cons
    for name, ref in refs.items():
        print("\n=== machine against %s, paper covariate definitions, weighted ===" % name)
        if name == "consensus" and todo:
            print("  (PROVISIONAL: %d disagreements are unresolved and excluded, which favours"
                  " easy articles; adjudicate before reporting)" % len(todo))
        print("  %-17s %4s %4s  %-20s %-20s %-20s %s" % ("covariate", "n", "pos", "sensitivity %",
                                                        "precision %", "kappa", "spec %"))
        comp[name] = {}
        for cov in COVS:
            keep = [a for a, h in ref.items()
                    if h.get("is_pedestrian_fatal_crash") == "yes"
                    and all(h.get(f) is not None for f in needs[cov])]
            if not keep:
                continue
            y = np.array([indicators(ref[a])[cov] for a in keep])
            x = np.array([int(ev.loc[a, cov]) for a in keep])
            w = np.array([float(ev.loc[a, "weight"]) for a in keep])
            st = ev.loc[keep, "stratum"].to_numpy()
            s = weighted_binary(y, x, w)
            for m in ("sensitivity", "precision", "kappa"):
                s[m + "_ci"] = sboot(lambda ii: weighted_binary(y[ii], x[ii], w[ii])[m], st)
            comp[name][cov] = s
            print("  %-17s %4d %4d  %-20s %-20s %-20s %5.1f"
                  % (cov, s["n_items"], s["n_pos_human"],
                     fmt_ci(s["sensitivity"], s["sensitivity_ci"], True),
                     fmt_ci(s["precision"], s["precision_ci"], True),
                     fmt_ci(s["kappa"], s["kappa_ci"]), 100 * s["specificity"]))
    out["machine_vs_human"] = comp

    outp = Path(os.environ.get("APP_SCORES", ROOT / "data_build" / "validation_app_scores.json"))
    json.dump(out, open(outp, "w"), indent=1, default=float)
    print("\nwrote", outp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    main()

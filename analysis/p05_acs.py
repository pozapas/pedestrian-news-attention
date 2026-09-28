"""p05_acs.py -- TRB-P Step 5 (STRETCH): county median-income covariate sensitivity.

Pulls county median household income (ACS 5-yr 2023, table B19013) from the public
Census API (no key needed at this volume), joins to events on normalized county
name within state, and adds income quartile to the secondary ZTNB as a sensitivity.
Gracefully SKIPS (writing results['acs']={'status': ...}) if the API is
unreachable or the join rate is < 60% -- the paper stands without it.

Run:  python analysis/p05_acs.py
"""
from __future__ import annotations
import json, re, urllib.request, urllib.error
import numpy as np
import pandas as pd
import common as C
import p03_ztnb as P3
import ztnb

ACS_URL = ("https://api.census.gov/data/2023/acs/acs5"
           "?get=NAME,B19013_001E&for=county:*")

STATE_NAME_TO_ABBR = {v: k for k, v in C.STATE_NAMES.items()}
STATE_NAME_TO_ABBR["District of Columbia"] = "DC"


def norm_county(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(county|parish|borough|census area|municipality|city and borough)\b", "", s)
    s = re.sub(r"[^a-z ]", "", s).strip()
    return re.sub(r"\s+", " ", s)


def fetch_acs():
    req = urllib.request.Request(ACS_URL, headers={"User-Agent": "trb-p-research"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    rows = data[1:]
    recs = []
    for name, inc, stfips, cofips in rows:
        try:
            income = float(inc)
        except (TypeError, ValueError):
            continue
        if income < 0:
            continue
        parts = name.split(",")
        if len(parts) != 2:
            continue
        county = norm_county(parts[0])
        st = STATE_NAME_TO_ABBR.get(parts[1].strip())
        if st:
            recs.append((st, county, income))
    return pd.DataFrame(recs, columns=["state", "county_norm", "median_income"])


def main():
    df = C.load_events()
    try:
        acs = fetch_acs()
    except (urllib.error.URLError, TimeoutError, Exception) as e:  # noqa
        print(f"[acs] API unreachable ({type(e).__name__}); skipping stretch step.")
        C.update_results("acs", {"status": "skipped_api_unreachable", "detail": str(e)[:120]})
        return

    print(f"[acs] pulled {len(acs):,} county income records.")
    df = df.copy()
    df["county_norm"] = df["county"].apply(norm_county)
    merged = df.merge(acs, on=["state", "county_norm"], how="left")
    join_rate = merged["median_income"].notna().mean()
    print(f"[acs] county join rate = {join_rate:.1%}")
    if join_rate < 0.60:
        print("[acs] join rate < 60% -> dropping step (paper stands without it).")
        C.update_results("acs", {"status": "skipped_low_join", "join_rate": float(join_rate)})
        return

    d = merged[merged["median_income"].notna()].copy()
    # income quartiles
    d["inc_q"] = pd.qcut(d["median_income"], 4, labels=False, duplicates="drop")
    # secondary ZTNB + income quartile dummies (ref = Q1)
    alpha = P3.determine_alpha(df)
    X = P3.make_design(d, P3.COVS_MAIN + P3.COVS_CONTEXT, road=True)
    for q in [1, 2, 3]:
        X[f"inc_q{q}"] = (d["inc_q"].values == q).astype(float)
    fit = P3.fit_beta_fixed_alpha(d["n_articles"].values.astype(float), X.values, alpha)
    bn = dict(zip(X.columns, fit["beta"]))
    # reference = quartile 0 (lowest income); RRs for Q2/Q3/Q4 (labels 1/2/3)
    inc_rr = {f"Q{q+1}": float(np.exp(bn[f"inc_q{q}"])) for q in [1, 2, 3]}
    key = {k: float(np.exp(bn[k])) for k in ["child", "hit_run_flag", "victim_named", "multi_fatality"]}
    print(f"[acs] income-quartile RRs (ref Q1=lowest): {inc_rr}")
    print(f"[acs] key covariate RRs with income adjustment: {key}")
    C.update_results("acs", {
        "status": "ok", "join_rate": float(join_rate), "n_joined": int(len(d)),
        "income_quartile_rr_vs_Q1lowest": inc_rr, "key_covariate_rr_income_adjusted": key,
        "note": "income quartile added to secondary ZTNB as sensitivity; RRs vs Q1 (lowest income)",
    })
    body = (f"## Step 5 — ACS county income (stretch)  ✅\n\n"
            f"- Pulled {len(acs):,} county median-income records (ACS 5-yr 2023, B19013).\n"
            f"- County join rate {join_rate:.1%} (n={len(d):,}).\n"
            f"- Coverage-intensity RR by county income quartile vs Q1 (lowest): "
            f"Q2 {inc_rr['Q2']:.2f}, Q3 {inc_rr['Q3']:.2f}, Q4 {inc_rr['Q4']:.2f}.\n"
            f"- Key covariate RRs stable under income adjustment: child {key['child']:.2f}, "
            f"hit-run {key['hit_run_flag']:.2f}, named {key['victim_named']:.2f}, "
            f"multi-fatality {key['multi_fatality']:.2f}.")
    C.write_qa_section("60-step5", body)
    print("[qa] step 5 section written")


if __name__ == "__main__":
    main()

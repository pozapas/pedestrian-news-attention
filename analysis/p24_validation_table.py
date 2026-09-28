"""p24_validation_table.py -- manuscript numbers and Table 5 for the human validation.

Inputs: validation_app/returns/coding_export.csv, ADJUDICATE_ME.xlsx (filled),
the app manifest, and data_build/validation_app_scores.json from p21.

Writes:
  outputs/tables/table5.tex                 validation table (auto-generated)
  data_build/validation_paper_numbers.json  every number the text quotes
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation_common import FIELDS, indicators, norm  # noqa: E402
from p21_score_app import cohen, sboot  # noqa: E402
from p23_make_adjudication import bkey, read_adjudication  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "validation_app"
RET = V / "returns"
PAPER = ROOT / "outputs" / "tables"

ROWS = [("Victim attributes", [("child", "Child ($\\le$12)"), ("teen", "Teen (13--19)"),
                               ("elderly", "Age $\\ge$65"), ("female", "Female victim"),
                               ("victim_named", "Victim named")]),
        ("Crash attributes", [("hit_run_flag", "Hit-and-run"), ("multi_fatality", "Multi-fatality")]),
        ("Context covariates (secondary model)", [("dark", "Dark conditions"),
                                                  ("charges_reported", "Charges reported")])]

df = pd.read_csv(RET / "coding_export.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
df = df[df["coder_id"].isin(["C01", "C02"])]
for f in FIELDS:
    df[f] = df[f].map(norm)
man = pd.read_csv(V / "manifest_app.csv")
design = json.load(open(V / "sample_design_app.json"))
df = df.merge(man[["item_code", "article_id", "is_retest_copy", "stratum", "weight"]], on="item_code")
orig = df[df["is_retest_copy"] == 0]
w = {c: orig[orig["coder_id"] == c].set_index("article_id") for c in ["C01", "C02"]}
arts = sorted(set(w["C01"].index) & set(w["C02"].index))
ev = man[man["is_retest_copy"] == 0].drop_duplicates("article_id").set_index("article_id")
adj, bad = read_adjudication(RET / "ADJUDICATE_ME.xlsx")
assert not bad, bad

# consensus, as in p21
cons = {}
for a in arts:
    f1 = w["C01"].loc[a, "is_pedestrian_fatal_crash"]; f2 = w["C02"].loc[a, "is_pedestrian_fatal_crash"]
    h = {}
    for f in FIELDS:
        v1, v2 = w["C01"].loc[a, f], w["C02"].loc[a, f]
        if f != "is_pedestrian_fatal_crash" and f1 != f2:
            v1 = v2 = v1 if f1 == "yes" else v2
        h[f] = v1 if (v1 == v2 or bkey(f, v1) == bkey(f, v2)) else adj.get((a, f))
    cons[a] = h
assert all(v is not None for h in cons.values() for v in h.values())

n = {}
n["articles"] = len(arts)
n["eligible_events"] = design["eligible_events"]
n["strata"] = len(design["allocation"])
n["adjudicated_decisions"] = len(adj)
n["adjudicated_articles"] = len({a for a, _ in adj})
not_ped = [a for a in arts if cons[a]["is_pedestrian_fatal_crash"] == "no"]
n["consensus_not_ped_fatal"] = len(not_ped)
wt = ev.loc[arts, "weight"]
n["weighted_share_ped_fatal"] = float(wt[[a not in not_ped for a in arts]].sum() / wt.sum())
n["confirmed_ped_fatal"] = len(arts) - len(not_ped)
notes = orig[orig["coder_notes"].str.strip() != ""]
n["articles_flagged_unreadable"] = int(notes["article_id"].nunique())
n["q1_intercoder_agreement"] = float(np.mean([w["C01"].loc[a, "is_pedestrian_fatal_crash"] ==
                                              w["C02"].loc[a, "is_pedestrian_fatal_crash"] for a in arts]))

both_yes = [a for a in arts if w["C01"].loc[a, "is_pedestrian_fatal_crash"] == "yes"
            and w["C02"].loc[a, "is_pedestrian_fatal_crash"] == "yes"]
n["both_yes"] = len(both_yes)
st = ev.loc[both_yes, "stratum"].to_numpy()
ind = {c: pd.DataFrame([indicators(w[c].loc[a, FIELDS].to_dict()) for a in both_yes], index=both_yes)
       for c in w}
scores = json.load(open(ROOT / "data_build" / "validation_app_scores.json"))["machine_vs_human"]["consensus"]
cov = {}
for _, rows in ROWS:
    for k, _lab in rows:
        a1, a2 = ind["C01"][k].to_numpy(), ind["C02"][k].to_numpy()
        kap, po = cohen(a1, a2)
        ci = sboot(lambda ii: cohen(a1[ii], a2[ii])[0], st)
        s = scores[k]
        cov[k] = {"intercoder_kappa": kap, "intercoder_kappa_ci": ci, "intercoder_agreement": po,
                  "human_pos": s["n_pos_human"], "n": s["n_items"],
                  "sens": s["sensitivity"], "sens_ci": s["sensitivity_ci"],
                  "prec": s["precision"], "prec_ci": s["precision_ci"],
                  "kappa": s["kappa"], "kappa_ci": s["kappa_ci"], "spec": s["specificity"]}
n["covariates"] = cov

# retest on the model covariates
rt = []
for c, g in df.groupby("coder_id"):
    for _, r in g[g["is_retest_copy"] == 1].iterrows():
        o = g[(g["article_id"] == r["article_id"]) & (g["is_retest_copy"] == 0)].iloc[0]
        io, ir = indicators(o[FIELDS].to_dict()), indicators(r[FIELDS].to_dict())
        same_q1 = o["is_pedestrian_fatal_crash"] == r["is_pedestrian_fatal_crash"]
        rt += [same_q1 and io[k] == ir[k] for _, rows in ROWS for k, _l in rows]
n["retest_pairs"] = int(len(df[df["is_retest_copy"] == 1]))
n["retest_covariate_agreement"] = float(np.mean(rt))

# ---------------------------------------------------------------- table
def pct(x): return "%.1f" % (100 * x)
def ci(v, c, p=True):
    f = pct if p else (lambda x: "%.2f" % x)
    return "%s (%s--%s)" % (f(v), f(c[0]), f(c[1]))

L = ["% Auto-generated by p24_validation_table.py -- do not edit by hand.",
     "\\begin{tabular}{l r c r r r}", "\\toprule",
     " & & Coders & \\multicolumn{3}{c}{Extraction against adjudicated consensus} \\\\",
     "\\cmidrule(lr){4-6}",
     "Covariate & Positives & $\\kappa$ & Sensitivity, \\% & Precision, \\% & $\\kappa$ (95\\% CI) \\\\",
     "\\midrule"]
for gi, (grp, rows) in enumerate(ROWS):
    if gi:
        L.append("\\addlinespace[2pt]")
    L.append("\\multicolumn{6}{l}{\\emph{%s}}\\\\" % grp)
    for k, lab in rows:
        c = cov[k]
        L.append("\\quad %s & %d & %.2f & \\databarL{%.1f}{100}{%s} & \\databarL{%.1f}{100}{%s} & %s \\\\"
                 % (lab, c["human_pos"], c["intercoder_kappa"], 100 * c["sens"], pct(c["sens"]),
                    100 * c["prec"], pct(c["prec"]), ci(c["kappa"], c["kappa_ci"], False)))
L += ["\\bottomrule", "\\end{tabular}"]
(PAPER / "table5.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
json.dump(n, open(ROOT / "data_build" / "validation_paper_numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in n.items() if k != "covariates"}, indent=1, default=float))
for k, c in cov.items():
    print("%-17s pos %3d  coder k %.2f %s  sens %s  prec %s  k %s" % (
        k, c["human_pos"], c["intercoder_kappa"], [round(x, 2) for x in c["intercoder_kappa_ci"]],
        ci(c["sens"], c["sens_ci"]), ci(c["prec"], c["prec_ci"]), ci(c["kappa"], c["kappa_ci"], False)))

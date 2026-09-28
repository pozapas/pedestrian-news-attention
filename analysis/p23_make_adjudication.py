"""p23_make_adjudication.py -- the adjudication workbook for the two-coder app study.

Lists only the coder disagreements that change a model covariate, so a split
between "no" and "not stated" on driver_fled, which leaves hit_run_flag at 0
either way, is not listed. Machine answers are not shown, so the adjudicator
stays blind to the system being validated.

Writes validation_app/returns/ADJUDICATE_ME.xlsx. The adjudicator picks an
answer from the dropdown in the yellow column and saves the same file;
p21_score_app.py reads it through read_adjudication(). An existing workbook is
never overwritten.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation_common import norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "validation_app"
OUT = V / "returns" / "ADJUDICATE_ME.xlsx"
SHEET = "Adjudicate"
HEADER_ROW = 6

QUESTION = {
    "is_pedestrian_fatal_crash": ("Was a pedestrian killed in this crash?", ["yes", "no"]),
    "n_pedestrians_killed": ("How many pedestrians died?", None),
    "victim_age": ("Age of the pedestrian who died (youngest if several)", None),
    "victim_gender": ("Gender of the pedestrian who died", ["male", "female", "mixed", "not_stated"]),
    "victim_named": ("Does the article name the pedestrian who died?", ["yes", "no"]),
    "driver_fled": ("Did the driver leave the scene?", ["yes", "no", "not_stated"]),
    "charges_mentioned": ("What does the article say about criminal charges?",
                          ["filed", "none_or_pending", "not_stated"]),
    "lighting": ("Was it light or dark at the time of the crash?",
                 ["daylight", "dark", "dawn_or_dusk", "not_stated"]),
}


def _num(v):
    return int(v) if str(v).isdigit() else None


def bkey(field, v):
    """What a field value means for the model covariates built from it."""
    v = norm(v)
    if field == "n_pedestrians_killed":
        n = _num(v); return n is not None and n > 1
    if field == "victim_age":
        a = _num(v)
        if a is None: return "none"
        return "child" if a <= 12 else "teen" if a <= 19 else "older" if a >= 65 else "adult"
    if field == "victim_gender": return v == "female"
    if field == "driver_fled": return v == "yes"
    if field == "charges_mentioned": return v == "filed"
    if field == "lighting": return v == "dark"
    return v


def allowed(field, c1, c2):
    opts = QUESTION[field][1]
    if opts is not None:
        return opts
    # numeric questions: offer both coders' numbers and not_stated; any number may be typed
    nums = sorted({x for x in (c1, c2) if str(x).isdigit()}, key=int)
    return nums + ["not_stated"]


def build_rows():
    df = pd.read_csv(V / "returns" / "coding_export.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    man = pd.read_csv(V / "manifest_app.csv", usecols=["item_code", "article_id", "is_retest_copy"])
    df = df.merge(man, on="item_code")
    df = df[df["coder_id"].isin(["C01", "C02"]) & (df["is_retest_copy"] == 0)]
    w = {c: df[df["coder_id"] == c].set_index("article_id") for c in ["C01", "C02"]}
    items = json.loads(re.search(r"export const ITEMS = (\{.*?\});\nexport const ORDER",
                                 (ROOT / "webapp" / "lib" / "items.js").read_text(encoding="utf-8"), re.S).group(1))
    rows = []
    for a in sorted(set(w["C01"].index) & set(w["C02"].index)):
        r1, r2 = w["C01"].loc[a], w["C02"].loc[a]
        f1, f2 = norm(r1["is_pedestrian_fatal_crash"]), norm(r2["is_pedestrian_fatal_crash"])
        if f1 != f2:
            fields = ["is_pedestrian_fatal_crash"]
        elif f1 == "yes":
            fields = [f for f in QUESTION if f != "is_pedestrian_fatal_crash"
                      and bkey(f, r1[f]) != bkey(f, r2[f])]
        else:
            fields = []
        it = items[r1["item_code"]]
        for f in fields:
            rows.append({"article_id": a, "field": f, "question": QUESTION[f][0],
                         "coder_1": r1[f] or "(said not a pedestrian death)",
                         "coder_2": r2[f] or "(said not a pedestrian death)",
                         "allowed": allowed(f, r1[f], r2[f]),
                         "notes": " / ".join(x for x in (r1["coder_notes"], r2["coder_notes"]) if x),
                         "source": "%s, %s" % (it["outlet"], it["published"]),
                         "text": it["text"]})
    return rows


def write_workbook(rows):
    from openpyxl import Workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    F = "Arial"
    fill = lambda c: PatternFill("solid", start_color=c, end_color=c)
    thin = Side(style="thin", color="BFBFBF")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    top = Alignment(vertical="top", wrap_text=True)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    wb = Workbook()
    ws = wb.active
    ws.title = SHEET
    n = len(rows)
    first, last = HEADER_ROW + 1, HEADER_ROW + n

    # ---- legend
    ws["A1"] = "Adjudication of coder disagreements"
    ws["A1"].font = Font(name=F, size=14, bold=True)
    ws["A2"] = ("For each row, read the article on the right, then pick the correct answer "
                "from the dropdown in the yellow YOUR ANSWER column. The row turns green when answered.")
    ws["A3"] = ("Code only what the article says; choose not_stated when it does not say. "
                "Example: Coder 1 = filed, Coder 2 = not_stated, article says \"charged with DUI\" -> choose filed.")
    ws["A4"] = ("For number questions you may also type a number. "
                "When done, save this file under the same name (keep it .xlsx).")
    for r in (2, 3, 4):
        ws.cell(r, 1).font = Font(name=F, size=10, color="404040")
    ws["H1"] = "Answered"
    ws["H1"].font = Font(name=F, size=11, bold=True)
    ws["H1"].alignment = Alignment(horizontal="right")
    ws["I1"] = '=COUNTA(F%d:F%d)&" of %d"' % (first, last, n)
    ws["I1"].font = Font(name=F, size=12, bold=True, color="1F6B2E")

    # ---- header
    heads = ["#", "Article", "Question", "Coder 1 said", "Coder 2 said", "YOUR ANSWER",
             "Allowed answers", "Coder notes", "Article (outlet, date, full text)", "field"]
    widths = [5, 14, 34, 18, 18, 20, 26, 22, 95, 4]
    for j, (h, wd) in enumerate(zip(heads, widths), start=1):
        c = ws.cell(HEADER_ROW, j, h)
        c.font = Font(name=F, bold=True, color="FFFFFF")
        c.fill = fill("B8860B" if h == "YOUR ANSWER" else "1F3864")
        c.alignment = center
        c.border = box
        ws.column_dimensions[c.column_letter].width = wd
    ws.row_dimensions[HEADER_ROW].height = 30

    # ---- rows, grouped by article with alternating bands
    band = ["FFFFFF", "EEF3FA"]
    groups, start = [], 0
    for i in range(1, n + 1):
        if i == n or rows[i]["article_id"] != rows[start]["article_id"]:
            groups.append((start, i)); start = i
    for g, (s, e) in enumerate(groups):
        bg = fill(band[g % 2])
        for i in range(s, e):
            r = first + i
            row = rows[i]
            vals = [i + 1, row["article_id"], row["question"], row["coder_1"], row["coder_2"], None,
                    " | ".join(row["allowed"]), row["notes"], None, row["field"]]
            for j, v in enumerate(vals, start=1):
                c = ws.cell(r, j, v)
                c.font = Font(name=F, size=10)
                c.alignment = top if j in (3, 7, 8, 9) else center
                c.border = box
                c.fill = bg
            ws.cell(r, 3).font = Font(name=F, size=10, bold=True)
            ws.cell(r, 4).font = Font(name=F, size=10, bold=True, color="1F4E9A")   # coder 1 blue
            ws.cell(r, 5).font = Font(name=F, size=10, bold=True, color="A0522D")   # coder 2 brown
            ans = ws.cell(r, 6)
            ans.fill = fill("FFF2CC")
            ans.font = Font(name=F, size=11, bold=True)
            dv = DataValidation(type="list", formula1='"%s"' % ",".join(row["allowed"]), allow_blank=True)
            if QUESTION[row["field"]][1] is None:
                dv.showErrorMessage = False        # numbers may be typed
            else:
                dv.error, dv.errorTitle = "Please pick one of the listed answers.", "Not an allowed answer"
            dv.prompt, dv.promptTitle = "Pick: " + " / ".join(row["allowed"]), "Your answer"
            ws.add_data_validation(dv)
            dv.add(ans)
        # article text once per article, merged down its rows
        t = ws.cell(first + s, 9, "%s\n\n%s" % (row["source"], rows[s]["text"]))
        t.font = Font(name=F, size=10)
        t.alignment = top
        t.fill = bg
        if e - s > 1:
            ws.merge_cells(start_row=first + s, start_column=9, end_row=first + e - 1, end_column=9)
            for col in (2,):
                ws.merge_cells(start_row=first + s, start_column=col, end_row=first + e - 1, end_column=col)
        # height: enough lines for the article, spread over its rows (Excel caps a row at 409 pt)
        text = rows[s]["text"]
        lines = sum(max(1, -(-len(p) // 105)) + 1 for p in text.split("\n\n")) + 2
        per_row = min(409, max(60, lines * 13.5 / (e - s)))
        for i in range(s, e):
            ws.row_dimensions[first + i].height = per_row

    # answered rows turn green; question-1 rows get a marker colour
    green = FormulaRule(formula=['LEN($F%d)>0' % first], fill=fill("C6EFCE"), font=Font(name=F, color="006100", bold=True))
    ws.conditional_formatting.add("F%d:F%d" % (first, last), green)
    ws.conditional_formatting.add("A%d:A%d" % (first, last),
                                  FormulaRule(formula=['LEN($F%d)>0' % first], fill=fill("C6EFCE")))

    ws.column_dimensions["J"].hidden = True
    ws.freeze_panes = ws.cell(first, 3)
    ws.auto_filter.ref = "A%d:H%d" % (HEADER_ROW, last)
    ws.sheet_view.zoomScale = 110
    ws.page_setup.orientation = "landscape"
    from openpyxl.workbook.properties import CalcProperties
    wb.calculation = CalcProperties(fullCalcOnLoad=True)   # the progress counter computes on open
    wb.save(OUT)


def read_adjudication(path=None):
    """{(article_id, field): answer} from the filled workbook, plus a list of problems."""
    from openpyxl import load_workbook
    path = Path(path or OUT)
    if not path.exists():
        csv_path = path.with_name("adjudication.csv")   # released form, no article text
        if not csv_path.exists():
            return {}, []
        t = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        return {(r.article_id, r.field): norm(r.adjudicated) for r in t.itertuples()}, []
    ws = load_workbook(path, data_only=True)[SHEET]
    out, bad = {}, []
    article = None
    for r in range(HEADER_ROW + 1, ws.max_row + 1):
        field = ws.cell(r, 10).value
        if not field:
            continue
        article = ws.cell(r, 2).value or article           # merged cells hold the value in the first row
        v = ws.cell(r, 6).value
        if v is None or str(v).strip() == "":
            continue
        v = norm(str(v).strip())
        if isinstance(ws.cell(r, 6).value, float) and float(ws.cell(r, 6).value).is_integer():
            v = str(int(ws.cell(r, 6).value))
        opts = QUESTION[field][1]
        ok = (v.isdigit() or v == "not_stated") if opts is None else v in opts
        if ok:
            out[(article, field)] = v
        else:
            bad.append("row %s (%s): %r" % (ws.cell(r, 1).value, field, ws.cell(r, 6).value))
    return out, bad


def main():
    if OUT.exists():
        print("%s already exists; not overwriting it." % OUT)
        return
    rows = build_rows()
    write_workbook(rows)
    fields = pd.Series([r["field"] for r in rows]).value_counts()
    print("wrote %s: %d rows from %d articles" % (OUT, len(rows), len({r["article_id"] for r in rows})))
    print(fields.to_string())


if __name__ == "__main__":
    main()

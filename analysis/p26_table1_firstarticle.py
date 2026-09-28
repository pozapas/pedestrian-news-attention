"""p26_table1_firstarticle.py -- Table 1 under the single specification.

Rebuilds the coverage-tier table with every attribute read from the first
linked article, which is how the models read them, using the table builder in
p02_descriptives.py unchanged. Median victim age uses the first-article age.

Writes tables/table1.tex and copies it to outputs/tables/table1.tex.
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import p02_descriptives as P  # noqa: E402

fa = pd.read_parquet(C.DATA_BUILD / "events_firstart.parquet")
fa["age_youngest"] = pd.to_numeric(fa["fa_age"], errors="coerce")
rows = P.build_table1(fa)
src = C.TABLES / "table1.tex"
txt = src.read_text(encoding="utf-8").replace("by p02_descriptives.py", "by p26_table1_firstarticle.py (first-article attributes)")
src.write_text(txt, encoding="utf-8")
shutil.copy(src, C.ROOT / "outputs" / "tables" / "table1.tex")
for k, v in rows.items():
    print("%-24s %s" % (k, {c: (round(x, 3) if x is not None else None) for c, x in v.items()}))

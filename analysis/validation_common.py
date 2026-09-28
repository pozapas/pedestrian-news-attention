"""validation_common.py -- field list and scoring helpers shared by p21 to p24."""
from __future__ import annotations
import re
import numpy as np


FIELDS = ["is_pedestrian_fatal_crash", "n_pedestrians_killed", "victim_age",
          "victim_gender", "victim_named", "driver_fled", "charges_mentioned",
          "lighting"]


def norm(x):
    x = ("" if x is None else str(x)).strip().lower()
    return x if x else "blank"


def to_int(v):
    m = re.search(r"\d+", v or "")
    return int(m.group(0)) if m else None


def indicators(h):
    """Human answers -> the paper's covariate definitions (reported = 1)."""
    age = to_int(h["victim_age"]) if h["victim_age"] not in ("not_stated", "blank") else None
    n = to_int(h["n_pedestrians_killed"])
    return {
        "victim_named": int(h["victim_named"] == "yes"),
        "hit_run_flag": int(h["driver_fled"] == "yes"),
        "multi_fatality": int(n is not None and n > 1),
        "female": int(h["victim_gender"] == "female"),
        "child": int(age is not None and age <= 12),
        "teen": int(age is not None and 13 <= age <= 19),
        "elderly": int(age is not None and age >= 65),
        "dark": int(h["lighting"] == "dark"),
        "charges_reported": int(h["charges_mentioned"] == "filed"),
    }


def weighted_binary(y, x, w):
    y, x, w = map(np.asarray, (y, x, w))
    tp = w[(y == 1) & (x == 1)].sum(); fn = w[(y == 1) & (x == 0)].sum()
    fp = w[(y == 0) & (x == 1)].sum(); tn = w[(y == 0) & (x == 0)].sum()
    T = tp + fn + fp + tn
    po = (tp + tn) / T
    pe = ((tp + fn) / T) * ((tp + fp) / T) + ((tn + fp) / T) * ((tn + fn) / T)
    return {"sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan"),
            "n_items": int(len(y)), "n_pos_human": int((y == 1).sum())}

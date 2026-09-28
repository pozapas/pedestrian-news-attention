"""Shared Okabe-Ito palette with fixed role mapping for the TRB 2027 Crash-News papers.

Specs from 00_TRB2027_README.md section 2. Colorblind-safe. Every figure script
imports PALETTE / ROLE and uses the role names, never raw hex, so the two papers
stay visually identical.
"""
from __future__ import annotations

# Fixed role -> hex mapping (README section 2 table).
PALETTE = {
    "human": "#0072B2",       # deep blue  - pedestrian corpus, human-driven baselines
    "av": "#D55E00",          # vermillion - AV corpus, AV effects
    "highlight": "#E69F00",   # amber      - flagged events, callouts
    "secondary": "#009E73",   # green      - third series
    "tertiary": "#CC79A7",    # purple     - rare fourth series
    "neutral": "#999999",     # grey       - reference lines
    "neutral_light": "#B0B0B0",
    "ink": "#3B3B3B",         # text, axes
}

# Convenience aliases matching how the paper describes roles.
ROLE = PALETTE
PED = PALETTE["human"]        # pedestrian / human role = deep blue
AV = PALETTE["av"]
HIGHLIGHT = PALETTE["highlight"]
SECONDARY = PALETTE["secondary"]
TERTIARY = PALETTE["tertiary"]
NEUTRAL = PALETTE["neutral"]
INK = PALETTE["ink"]

# Sequential ramp for heat shading (figures AND table cells): #F7FBFF -> #0072B2.
SEQ_RAMP = ["#F7FBFF", "#0072B2"]


def seq_cmap(name: str = "trb_blues"):
    """Return a matplotlib LinearSegmentedColormap for the sequential ramp."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQ_RAMP)


def ramp_hex(t: float) -> str:
    """Map t in [0,1] onto the sequential ramp, return a hex string.

    Used by the table generators to write \\cellcolor values into .tex.
    """
    import numpy as np
    from matplotlib.colors import to_rgb
    t = float(min(1.0, max(0.0, t)))
    c0 = np.array(to_rgb(SEQ_RAMP[0]))
    c1 = np.array(to_rgb(SEQ_RAMP[1]))
    c = c0 + t * (c1 - c0)
    return "#{:02X}{:02X}{:02X}".format(
        int(round(c[0] * 255)), int(round(c[1] * 255)), int(round(c[2] * 255))
    )


def rgb01(hex_str: str):
    """Return an (r,g,b) tuple in 0..1 for LaTeX \\definecolor{}{rgb}{...}."""
    from matplotlib.colors import to_rgb
    return to_rgb(hex_str)

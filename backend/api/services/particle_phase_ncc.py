"""Phase comparison per particle: NCC statistics over the best-matching patterns.

The single-pixel phase test answers "which phase renders this ONE pattern
best". A particle (an intermetallic, a precipitate) is judged on several
patterns, and what goes into a thesis is a number with a spread. For every
particle and every candidate phase this module takes the per-pattern render
NCC (measured vs simulated Kikuchi pattern, after the per-phase orientation
re-index/refinement of the compare engine) and reports:

* ``mean`` and ``std`` (sample standard deviation, n-1) over the ``top_n``
  patterns with the highest NCC for that phase -- by default the ten best,
* ``max`` -- the single best NCC,
* the assigned phase (highest mean), the next-best phase and the gap
  ``delta`` between the two means.

A gap below :data:`INDISTINGUISHABLE_DELTA` is flagged: the two phases cannot
be told apart by NCC (isostructural pairs such as Al6Fe / MnAl6 differ by
~0.003). The flag is a statement about the measurement, not a tie-break --
the higher mean is still listed first.

Pure functions only; the route that collects the scores lives in
``backend/api/routes/indexing.py`` (``/particle-phase-ncc``). The LaTeX output
reproduces the two tables of the phase assignment (booktabs + tabularx,
German decimal comma).
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

#: Patterns per phase that enter the mean ("the ten best").
TOP_N = 10
#: Below this NCC gap the assigned and the next-best phase are reported as not
#: distinguishable by NCC.
INDISTINGUISHABLE_DELTA = 0.01


def _finite(values: Iterable) -> np.ndarray:
    a = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return a[np.isfinite(a)]


def phase_stats(scores: Iterable, top_n: int = TOP_N) -> Dict:
    """Statistics of one phase on one particle.

    ``mean``/``std`` over the ``top_n`` highest finite scores (all of them when
    there are fewer); ``std`` is None for a single value. ``n`` is how many
    entered the mean, ``n_total`` how many patterns had a finite score."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    s = np.sort(_finite(scores))[::-1]
    if s.size == 0:
        return {"mean": None, "std": None, "max": None, "n": 0, "n_total": 0}
    top = s[:top_n]
    return {
        "mean": float(top.mean()),
        "std": float(top.std(ddof=1)) if top.size > 1 else None,
        "max": float(s[0]),
        "n": int(top.size),
        "n_total": int(s.size),
    }


def summarize_particle(name: str, scores_by_phase: Mapping[str, Iterable],
                       top_n: int = TOP_N,
                       indistinguishable_delta: float = INDISTINGUISHABLE_DELTA) -> Dict:
    """Rank the candidate phases of one particle by their top-``top_n`` mean.

    Returns ``{"name", "phases": [{"phase_key", mean, std, max, n, n_total}...]
    (best first, unscored last), "assigned", "next_best", "delta",
    "indistinguishable", "n_patterns"}``."""
    rows = []
    for key, scores in scores_by_phase.items():
        st = phase_stats(scores, top_n)
        rows.append({"phase_key": str(key), **st})
    rows.sort(key=lambda r: (r["mean"] is None, -(r["mean"] or 0.0)))
    scored = [r for r in rows if r["mean"] is not None]
    assigned = scored[0] if scored else None
    nxt = scored[1] if len(scored) > 1 else None
    delta = (assigned["mean"] - nxt["mean"]) if (assigned and nxt) else None
    n_patterns = max((r["n_total"] for r in rows), default=0)
    return {
        "name": str(name),
        "phases": rows,
        "assigned": assigned["phase_key"] if assigned else None,
        "next_best": nxt["phase_key"] if nxt else None,
        "delta": delta,
        "indistinguishable": bool(delta is not None and delta < indistinguishable_delta),
        "n_patterns": int(n_patterns),
    }


def summarize(particles: Sequence[Mapping], phase_info: Optional[Mapping[str, Mapping]] = None,
              top_n: int = TOP_N,
              indistinguishable_delta: float = INDISTINGUISHABLE_DELTA) -> Dict:
    """Whole evaluation: one summary per particle plus the candidate matrix.

    ``particles``: ``[{"name": str, "scores": {phase_key: [ncc, ...]}}, ...]``.
    ``phase_info``: ``{phase_key: {"label": str, "space_group": str}}`` used for
    display and the LaTeX tables (the key is used when a phase has none).

    ``matrix`` is ``{phase_key: {particle_name: mean}}`` in library order of
    first appearance; ``best_per_particle`` names the highest mean per particle
    (the bold entry of the candidate table)."""
    info = {str(k): dict(v) for k, v in (phase_info or {}).items()}
    out_particles = [
        summarize_particle(p["name"], p.get("scores") or {}, top_n, indistinguishable_delta)
        for p in particles
    ]
    phase_order: List[str] = []
    for p in particles:
        for k in (p.get("scores") or {}):
            if str(k) not in phase_order:
                phase_order.append(str(k))
    matrix: Dict[str, Dict[str, Optional[float]]] = {k: {} for k in phase_order}
    for sp in out_particles:
        for r in sp["phases"]:
            matrix[r["phase_key"]][sp["name"]] = r["mean"]
    for k in phase_order:
        info.setdefault(k, {})
        info[k].setdefault("label", k)
        info[k].setdefault("space_group", "")
    return {
        "top_n": int(top_n),
        "indistinguishable_delta": float(indistinguishable_delta),
        "particles": out_particles,
        "phase_order": phase_order,
        "phase_info": info,
        "matrix": matrix,
        "best_per_particle": {sp["name"]: sp["assigned"] for sp in out_particles},
    }


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------

def num_de(x: Optional[float], digits: int = 3) -> str:
    """``0{,}541`` -- German decimal comma, braced so LaTeX adds no space.
    ``--`` for a missing value (callers put it outside math mode, see :func:`_m`)."""
    if x is None or not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}".replace(".", "{,}")


def _m(x: Optional[float], digits: int = 3, bold: bool = False) -> str:
    """A number in math mode; a missing one as a text dash (``$--$`` would set
    two minus signs)."""
    txt = num_de(x, digits)
    if txt == "--":
        return "--"
    return f"$\\mathbf{{{txt}}}$" if bold else f"${txt}$"


def _tex_escape(text: str) -> str:
    return (str(text).replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("%", r"\%").replace("#", r"\#").replace("_", r"\_"))


_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d+(?:[.,]\d+)?)")
_GREEK = re.compile(r"(alpha|beta|gamma|delta|epsilon|eta|theta|kappa|lambda|mu|sigma|tau|phi|chi|omega)(\d*)",
                    re.IGNORECASE)
_FORMULA_TOKEN = re.compile(r"(?:[A-Z][a-z]?(?:\d+(?:[.,]\d+)?)?|[()\[\]])+(?:\d+(?:[.,]\d+)?)?")


def formula_tex(formula: str) -> str:
    """``Al13Fe4`` -> ``Al$_{13}$Fe$_4$``; ``Mn0.5Fe0.5`` -> ``Mn$_{0,5}$Fe$_{0,5}$``.
    Text that already contains ``$`` is passed through unchanged."""
    s = str(formula)
    if "$" in s:
        return s

    def sub(m):
        el, n = m.group(1), m.group(2).replace(".", ",")
        return f"{el}$_{n}$" if len(n) == 1 else f"{el}$_{{{n}}}$"
    # Only a token that is a formula as a whole gets subscripts: a Pearson
    # symbol ("cF4") or a note ("Variante 2") stays as written.
    parts = re.split(r"(\s+|,|-|\(|\))", s)
    out = []
    for part in parts:
        g = _GREEK.fullmatch(part or "")
        if g:
            sym = f"\\{g.group(1).lower()}"
            out.append(f"${sym}_{{{g.group(2)}}}$" if g.group(2) else f"${sym}$")
        elif part and _FORMULA_TOKEN.fullmatch(part) and _ELEMENT.search(part):
            out.append(_ELEMENT.sub(sub, part))
        else:
            out.append(_tex_escape(part))
    return "".join(out)


_SCREW = re.compile(r"(?<=[PIFCABR])([2346])([1-5])(?=/|[a-z]|$|\s)")


def space_group_tex(sg: str) -> str:
    """Hermann-Mauguin symbol in math mode: ``Fm-3m`` -> ``$Fm\\bar{3}m$``,
    ``P63/mmc`` -> ``$P6_3/mmc$``. Empty in, empty out."""
    s = str(sg or "").strip().replace(" ", "")
    if not s:
        return ""
    if s.startswith("$"):
        return s
    if "_" not in s:
        s = _SCREW.sub(lambda m: f"{m.group(1)}_{m.group(2)}", s)
    s = re.sub(r"-(\d)", r"\\bar{\1}", s)
    return f"${s}$"


def _label_tex(info: Mapping, key: str) -> str:
    entry = info.get(key) or {}
    return entry.get("label_tex") or formula_tex(entry.get("label") or key)


def latex_assignment_table(summary: Mapping, *, label: str = "tab:ncc_phasen",
                           caption: Optional[str] = None) -> str:
    """Table 1: per particle the assigned phase, NCC mean +- s, NCC_max, the
    next-best phase and the gap. Indistinguishable pairs get a footnote."""
    info = summary["phase_info"]
    top_n = summary["top_n"]
    names = ", ".join(p["name"] for p in summary["particles"])
    cap = caption or (
        f"Phasenzuordnung der Partikel {names}. Angegeben ist der normierte "
        "Kreuzkorrelationskoeffizient (NCC) zwischen gemessenem und simuliertem "
        "Kikuchi-Muster nach Orientierungsverfeinerung, gemittelt über die "
        f"{_count_word(top_n)} Muster mit der höchsten Übereinstimmung je Partikel.")
    rows, notes = [], []
    for sp in summary["particles"]:
        a, b = sp["assigned"], sp["next_best"]
        if a is None:
            rows.append(f"    {_tex_escape(sp['name'])} & -- & -- & -- & -- & -- \\\\")
            continue
        ra = next(r for r in sp["phases"] if r["phase_key"] == a)
        sg = space_group_tex(info.get(a, {}).get("space_group", ""))
        phase_cell = _label_tex(info, a)
        if sp["indistinguishable"] and b is not None:
            mark = chr(ord("a") + len(notes))
            phase_cell += f"\\textsuperscript{{{mark}}}"
            notes.append(
                f"\\textsuperscript{{{mark}}} {_label_tex(info, a)} und {_label_tex(info, b)} "
                f"sind anhand der NCC nicht unterscheidbar ($\\Delta$NCC $<$ "
                f"{num_de(summary['indistinguishable_delta'], 2)}); angegeben sind die Werte "
                f"für {_label_tex(info, a)}.")
        if sg:
            phase_cell += f" ({sg})"
        mean_std = (f"${num_de(ra['mean'])}\\pm{num_de(ra['std'])}$" if ra["std"] is not None
                    else _m(ra["mean"]))
        if b is not None:
            rb = next(r for r in sp["phases"] if r["phase_key"] == b)
            nxt = f"{_label_tex(info, b)} ({_m(rb['mean'])})"
            delta = _m(sp["delta"])
        else:
            nxt, delta = "--", "--"
        rows.append(f"    {_tex_escape(sp['name'])} & {phase_cell}\n"
                    f"          & {mean_std} & {_m(ra['max'])} & {nxt} & {delta} \\\\")
    body = "\n".join(rows)
    foot = ""
    if notes:
        foot = ("\n\n  \\vspace{0.5ex}\n  \\begin{minipage}{\\linewidth}\\footnotesize\n    "
                + " \\\\\n    ".join(notes) + "\n  \\end{minipage}")
    return (
        "% Benötigte Pakete: booktabs, tabularx, amsmath\n"
        "\\begin{table}[htbp]\n  \\centering\n"
        f"  \\caption{{{cap}}}\n  \\label{{{label}}}\n  \\small\n"
        "  \\renewcommand{\\arraystretch}{1.2}\n"
        "  \\begin{tabularx}{\\linewidth}{@{}l >{\\raggedright\\arraybackslash}X c c "
        ">{\\raggedright\\arraybackslash}X c@{}}\n    \\toprule\n"
        "    Partikel & Zugeordnete Phase (Raumgruppe) & NCC ($\\bar{x}\\pm s$) & "
        "NCC$_\\text{max}$ & Nächstbeste Phase (NCC) & $\\Delta$NCC \\\\\n    \\midrule\n"
        f"{body}\n    \\bottomrule\n  \\end{{tabularx}}{foot}\n\\end{{table}}\n")


def latex_candidate_table(summary: Mapping, *, label: str = "tab:ncc_kandidaten",
                          caption: Optional[str] = None) -> str:
    """Table 2: mean NCC of every candidate phase on every particle; the
    highest value per particle in bold."""
    info = summary["phase_info"]
    parts = [p["name"] for p in summary["particles"]]
    best = summary["best_per_particle"]
    names = ", ".join(parts)
    cap = caption or (
        f"Mittlere NCC ausgewählter Kandidatphasen für die Partikel {names} "
        f"(Mittelwert über jeweils {_count_word(summary['top_n'])} Muster nach "
        "Orientierungsverfeinerung). Der jeweils höchste Wert je Partikel ist fett hervorgehoben.")
    lines = []
    for key in summary["phase_order"]:
        cells = []
        for pn in parts:
            v = summary["matrix"].get(key, {}).get(pn)
            cells.append(_m(v, bold=(best.get(pn) == key and v is not None)))
        sg = space_group_tex(info.get(key, {}).get("space_group", ""))
        lines.append(f"    {_label_tex(info, key)} & {sg} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(_tex_escape(p) for p in parts)
    return (
        "% Benötigte Pakete: booktabs, amsmath\n"
        "\\begin{table}[htbp]\n  \\centering\n"
        f"  \\caption{{{cap}}}\n  \\label{{{label}}}\n  \\small\n"
        f"  \\begin{{tabular}}{{@{{}}l l {' '.join('c' for _ in parts)}@{{}}}}\n    \\toprule\n"
        f"    Phase & Raumgruppe & {head} \\\\\n    \\midrule\n"
        + "\n".join(lines) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")


def csv_table(summary: Mapping, sep: str = ";") -> str:
    """Long-format CSV (one row per particle and phase), decimal comma, for
    Excel with German locale."""
    info = summary["phase_info"]
    head = ["Partikel", "Phase", "Raumgruppe", "Rang", "NCC_Mittel", "NCC_s", "NCC_max",
            "n_Mittel", "n_Muster", "zugeordnet", "Delta_NCC", "nicht_unterscheidbar"]
    out = [sep.join(head)]

    def f(x):
        return "" if x is None else f"{x:.4f}".replace(".", ",")
    for sp in summary["particles"]:
        for rank, r in enumerate(sp["phases"], start=1):
            k = r["phase_key"]
            is_a = k == sp["assigned"]
            out.append(sep.join([
                sp["name"], info.get(k, {}).get("label", k), info.get(k, {}).get("space_group", ""),
                str(rank) if r["mean"] is not None else "", f(r["mean"]), f(r["std"]), f(r["max"]),
                str(r["n"]), str(r["n_total"]), "ja" if is_a else "",
                f(sp["delta"]) if is_a else "",
                ("ja" if sp["indistinguishable"] else "nein") if is_a else "",
            ]))
    return "\n".join(out) + "\n"


_WORDS = {1: "einem", 2: "zwei", 3: "drei", 4: "vier", 5: "fünf", 6: "sechs", 7: "sieben",
          8: "acht", 9: "neun", 10: "zehn", 11: "elf", 12: "zwölf"}


def _count_word(n: int) -> str:
    return _WORDS.get(int(n), str(int(n)))


def with_latex(summary: Dict) -> Dict:
    """``summary`` plus ``latex`` (both tables) and ``csv``."""
    out = dict(summary)
    out["latex"] = {
        "zuordnung": latex_assignment_table(summary),
        "kandidaten": latex_candidate_table(summary),
    }
    out["csv"] = csv_table(summary)
    return out


def disc_pixels(center_index: int, radius_px: int, n_rows: int, n_cols: int) -> List[int]:
    """Flat indices of the disc of ``radius_px`` (index-grid distance) around
    ``center_index``, clipped to the scan. ``radius_px`` 0 is the pixel alone."""
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("scan size unknown")
    if not 0 <= int(center_index) < n_rows * n_cols:
        raise ValueError(f"pixel {center_index} is outside the {n_rows} x {n_cols} scan")
    r0, c0 = divmod(int(center_index), n_cols)
    rad = max(0, int(radius_px))
    out = []
    for r in range(max(0, r0 - rad), min(n_rows, r0 + rad + 1)):
        for c in range(max(0, c0 - rad), min(n_cols, c0 + rad + 1)):
            if (r - r0) ** 2 + (c - c0) ** 2 <= rad * rad:
                out.append(r * n_cols + c)
    return out

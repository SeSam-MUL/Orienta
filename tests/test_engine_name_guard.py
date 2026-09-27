"""The built-in simulation engine is called "Orienta Engine" wherever a user can read it.

Until 2026-09-25 the engine had three names at once: "Ours" in the docs, the test
protocols and the backend log lines ("OUR GPU MC"), "GPU (native)" on the toggle
button — which read "GPU (nativ) — CPU (18 Kerne)" on a Mac without a GPU — and
"GPU pipeline" on the batch badge. This guard pins the one name.

It scans two things:

* every string value in the four locale trees (``frontend/src/locales/<lang>/*.json``);
* every string literal (including f-string parts and docstrings, which FastAPI
  publishes under ``/docs``) in the backend modules whose text reaches the
  frontend: the engine runner, the simulation route and the Aztec comparison
  figure.

Internal identifiers stay as they are on purpose: ``engine: "ours"`` in the API,
the provenance sidecars and the locale *keys* (``oursLabel`` …) are contracts
that saved masters and tests depend on. The guard therefore matches the words
``Ours`` / ``OUR`` only, never ``ours`` in an identifier.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCALES = ROOT / "frontend" / "src" / "locales"
LANGS = ("en", "de", "ja", "zh")
ENGINE_NAME = "Orienta Engine"

# "Ours"/"OUR" as whole words. "ours" (lower case) is the API identifier and is
# allowed; "YOUR" and "colours" do not match because of the word boundaries.
RETIRED_WORD = re.compile(r"\b(Ours|OUR)\b")
# The other two names the engine used to go by on screen.
RETIRED_LABELS = ("GPU (native)", "GPU (nativ)", "GPU（ネイティブ）", "GPU（原生）")

BACKEND_FILES = (
    "backend/api/services/gpu_sim_runner.py",
    "backend/api/routes/simulation.py",
    "backend/api/routes/analysis.py",
)


def _leaf_strings(node, prefix=""):
    """Yield (dotted_key, value) for every string leaf of a locale JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaf_strings(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _leaf_strings(value, f"{prefix}[{i}]")
    elif isinstance(node, str):
        yield prefix, node


def _string_literals(py_path: Path):
    """Every str constant in a module, with its line — f-string parts included."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def retired_hits(text: str) -> list[str]:
    """The retired names found in one user-visible string (empty = clean)."""
    hits = [m.group(0) for m in RETIRED_WORD.finditer(text)]
    hits += [label for label in RETIRED_LABELS if label in text]
    return hits


def test_scanner_bites_on_the_old_strings():
    # The guard is only worth something if it would have fired before the rename.
    assert retired_hits("[gpu_sim] MC source (b): OUR GPU MC (cupy CUDA kernel)") == ["OUR"]
    assert retired_hits('Leave it on "Ours" for the built-in pipeline') == ["Ours"]
    assert retired_hits("✨ GPU (nativ) — {{hardware}}") == ["GPU (nativ)"]
    # …and stays quiet on the identifier and on ordinary words.
    assert retired_hits('engine: "ours"') == []
    assert retired_hits("YOUR colours, our engine") == []


@pytest.mark.parametrize("lang", LANGS)
def test_locale_values_do_not_call_the_engine_ours(lang):
    offenders = []
    for path in sorted((LOCALES / lang).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in _leaf_strings(data):
            hits = retired_hits(value)
            if hits:
                offenders.append(f"{path.name}:{key} -> {hits}: {value!r}")
    assert not offenders, "retired engine name in user-visible text:\n" + "\n".join(offenders)


@pytest.mark.parametrize("lang", LANGS)
def test_engine_toggle_and_title_name_the_orienta_engine(lang):
    """The proper name is the same in every language (not translated)."""
    sim = json.loads((LOCALES / lang / "simulation.json").read_text(encoding="utf-8"))
    for key in ("oursLabel", "oursLabelWithHardware", "oursTooltip"):
        assert ENGINE_NAME in sim["engine"][key], f"{lang}: engine.{key} = {sim['engine'][key]!r}"
    for key in ("titleOursWithHardware", "titleOursPlain"):
        assert ENGINE_NAME in sim["header"][key], f"{lang}: header.{key} = {sim['header'][key]!r}"
    assert ENGINE_NAME in sim["missingDialog"]["pipelineOurs"]
    for key in ("startTooltipOurs", "simulateAllMissingTooltipOurs", "batchSelectTooltipOurs"):
        assert ENGINE_NAME in sim["actions"][key], f"{lang}: actions.{key} = {sim['actions'][key]!r}"
    assert ENGINE_NAME in sim["params"]["oursElectronNote"]


@pytest.mark.parametrize("rel", BACKEND_FILES)
def test_backend_strings_do_not_call_the_engine_ours(rel):
    offenders = []
    for lineno, text in _string_literals(ROOT / rel):
        hits = retired_hits(text)
        if hits:
            offenders.append(f"{rel}:{lineno} -> {hits}: {text.strip()[:80]!r}")
    assert not offenders, "retired engine name in a backend string:\n" + "\n".join(offenders)


def test_api_identifier_is_untouched():
    """Saved masters carry ``engine: "ours"`` in their provenance sidecar; the
    rename must not have reached the contract."""
    runner = (ROOT / "backend/api/services/gpu_sim_runner.py").read_text(encoding="utf-8")
    provenance = (ROOT / "backend/api/services/sht_provenance.py").read_text(encoding="utf-8")
    assert '"engine": "ours"' in runner
    assert 'return "ours" if' in provenance

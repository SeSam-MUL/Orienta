# bc-gmm — the reference Orienta add-on

Decomposes a scan's pattern-quality (band contrast) distribution into Gaussian
components, and reports each component's mean, spread and area fraction, the
histogram it was fitted to, and a per-pixel map of which component each pixel
belongs to.

This is Orienta's own `analysis/bc_analysis.py` rebuilt against the add-on
contract. Both ship: the core keeps its version, and this one exists to prove
the contract can carry a real analysis. If a change to the contract breaks this
add-on, the contract changed — that is what it is for.

It does one thing the core version does not: **the number of components is a
parameter.** The core fixes it at three, because its result type has hard-coded
`bc_low`/`bc_mid`/`bc_high` fields, and its own docstring names that as the
generalisation to make.

## Installing it

Copy this folder into `~/.orienta/addons/`, restart the backend, and enable it:

    POST /api/addons/bc-gmm/enabled
    {"enabled": true}

**That call is the whole of it.** API 0 ships no add-on settings screen, so any
message that says "enable it" means this request and not a place to click.
Orienta puts the folder on `sys.path` itself when it runs the add-on — there is
nothing to `pip install`. It runs in Orienta's process with your permissions,
which is true of every add-on and is why you are asked once before it runs.

If you **move** an add-on that has already run in this backend session, restart
the backend. Orienta refuses a module whose file is not inside the folder the
manifest came from, and a Python module that has already been imported keeps
the location it was imported from for the life of the process — so the copy at
the new path is refused, loudly, rather than the old code being run under the
new manifest.

## Using it

It needs a result with a pattern-quality map. Run it from the add-on panel, or:

    POST /api/addons/bc-gmm/run
    {"analysis_key": "addon.bc_gmm", "result_id": "<id>",
     "params": {"n_components": 3, "bin_width": 10}}

`bin_width` is in the units of the quality map — 10 suits native band contrast
(0–255). Pass `0` to take the width from the data by the Freedman–Diaconis
rule, which is what a computed 0–1 quality map needs.

**Component ranks.** Components are reported lowest mean first: rank 0, rank 1,
… For a three-component fit on *native band contrast* those are the classical
deformed / recovered / recrystallised populations. That reading is in this
paragraph and not in the data, because it is not true for any other component
count and not true for any other quality metric — and an add-on that labelled
its output "recrystallised" would be asserting it whatever it was given.

## Writing the manifest

A handful of things are easy to get wrong, so they are spelled out here.

**Parameters.** Only what `params_schema.properties` declares is passed to your
function; anything else the caller sends is refused and reported back in
`ignored_params`. An analysis with no `params_schema` takes no parameters.

**Nothing validates the values.** `minimum`, `maximum`, `enum` and the types
are documentation for a settings form that is not built yet. Today the runner
filters parameters by **name** and passes the values through untouched, so an
`n_components` of `0` or `"three"` reaches your function exactly as sent.
Check your own inputs and raise with a sentence — `analysis.py` does, in
`_positive_int` — or your user gets scikit-learn's traceback instead. And check
against *your own* wording when you test it: `pytest.raises(match="n_components")`
passes with the guard deleted, because scikit-learn's own refusal names the
parameter too.

`default` is the exception — it **is** read, by the runner, and for that reason
it lives in two places: the schema's is what a form will show, and your
signature's is what an empty `params` actually runs with. **The signature's is
what gets recorded**, because that is the value your analysis was given. Keep
them equal; a test in `tests/addons/test_example_addon.py` pins that for this
add-on. If they disagree, Orienta records what ran and logs the disagreement,
and a default for a parameter your function does not take is left out of the
trail altogether — a methods paragraph must not assert a value no line of the
run ever saw.

**What the sentence may name.** A slot in `sentence` may name a **declared
parameter** or a **scalar output key**. `{n_components}` is the first,
`{n_px}` is the second — the runner merges your scalar outputs into the
recorded parameters so the methods paragraph can state what the analysis
*found*, not only what it was asked to do. An input wins over an output of the
same name. A slot that names neither renders as `[name not recorded]`, honestly
and visibly.

**A `default` is recorded for you, so state one.** The runner records what the
caller *sent*, and a parameter the user left alone is not sent — so a sentence
slot naming it used to render `[n_components not recorded]` on the most
ordinary call there is, the button with nothing typed into it. That was a
defect in the runner, found on this add-on and fixed there: the runner now
merges your declared `default`s — as your signature has them — into the
recorded parameters, under anything the caller actually sent. What it cannot do is invent one, so a parameter your
sentence names and your schema gives no `default` still renders as not
recorded, honestly. Give every such parameter a `default`.

A parameter declared with `format = "path"` is passed to you and **never
recorded**, its `default` included: the recorded parameters go into the methods
paragraph and into every exported `.h5`, and a local path there publishes the
operator's username and directory layout.

**One asymmetry to know about**, because it will otherwise look like a bug in
your add-on. A *declared* string parameter that happens to look like a path —
say a `sample_id` a user fills in as `C:/runs/2026-09/A`, on a field you did
not mark `format = "path"` — is caught by Orienta's provenance guard, and that
guard behaves differently in the two places you will meet it. Under pytest it
**raises**, so the run dies and your test goes red; in a normal Orienta it
**warns** and records the value. Same add-on, same input, opposite outcomes.
If you mean a path, declare `format = "path"` and it is passed to you and
never recorded; if you mean an identifier, do not let it be one.

**Citations.** This add-on's `citations = ["orienta"]` is the id of Orienta's
own bibliography entry, because this add-on ships inside Orienta and has no DOI
of its own. **Yours is different:** put your DOI in `doi`, and the works you
want cited in `citations` as `doi:10.xxxx/yyyy`. An id the core bibliography
already knows is used as-is; anything else becomes an entry built from the DOI,
and a reference Orienta cannot turn into either is shown in the list marked
unresolved rather than quietly dropped.

**`requires_orienta`.** This add-on declares none. Orienta derives its running
version from the git release tag, and a source checkout has none — so on a
checkout *any* floor refuses to run, by design, since a build that cannot say
which release it is must not be assumed to meet one. State a floor when your
add-on genuinely needs one and your users install tagged releases; leave it
empty when you would rather run everywhere.

## What the context can and cannot tell you

Every optional field on `AddonContext` may be `None`, and `None` means *not
available*, never *zero* and never *one*. Three worth knowing before you build
on them:

`step_size_um` is resolved from the result and, failing that, from the loaded
scan. On a result whose scan has since been unloaded or switched, Orienta
cannot say what the step was — so it says `None` rather than defaulting to 1,
because a wrong step silently mis-scales every length you compute from it. If
your analysis reports µm, check the field and refuse with a sentence when it is
missing, the way this one refuses without a quality map.

`eds_at_pct` is `{}` on a real run today. The field and its shape are part of
the contract; the wiring to Orienta's per-pixel EDS quantification is not built
yet.

`report(message, fraction)` is a no-op on a real run today. The route installs
no progress sink, so every call is discarded. Call it anyway — it is the
contract, and the day a sink is wired in your add-on needs no change — but do
not build anything that depends on the message arriving.

## What this rebuild could not express, and what it did about it

Two things, both shown rather than quietly left out. An add-on you write will
meet them too.

**Grain-average band contrast is missing.** The core module's third function,
`grain_average_bc`, reads `grains.grain_ids` and `grains.n_grains`, and
`AddonContext` carries no grains — neither a grain map nor a grain count. It
cannot be reconstructed from what the context *does* carry either: grouping
pixels into grains means comparing neighbouring orientations under the
crystal's point group, and the symmetry operators live in `orix`, which this
add-on deliberately does not import. Ignoring symmetry and taking raw
axis-angle differences would invent boundaries inside single grains. So the
function is absent and named here, rather than approximated. If you need
grains, that gap in the context is the thing to ask for.

**The methods paragraph cannot name a measured word.** Only *numbers* reach it:
the runner merges `ScalarOutput`s into the recorded parameters, and a
`ScalarOutput`'s value is a float. So this analysis cannot state in its
sentence whether it fitted native band contrast or Orienta's computed pattern
quality — two different measurements that produce different numbers from the
same scan. `quality_source` is reported in the `fit_summary` table instead,
where the panel shows it and the export carries it, but it does not reach the
prose a user pastes into a manuscript. A string parameter *declared* in the
manifest would be recorded, but that would mean asking the user to type a fact
the software already knows, which is worse than the gap.

## Requirements

**numpy and scikit-learn.** Not scipy either, in the end: the one place it
would have been used is the normal CDF for the fitted-probability column, and
`math.erf` from the standard library does that exactly. No `orix`, no
`kikuchipy`, no `diffsims`, no `torch` — those are what Orienta's own start-up
already pays for, and the first add-on's dependency surface should pin as
little as possible. `tests/addons/test_example_addon.py` measures that claim
against this source rather than trusting this paragraph.

**And one module of Orienta itself**, which is a known limit of API 0 rather
than a choice this example made:

    from backend.api.services.addons.outputs import MapOutput, ScalarOutput

`validate_outputs` checks your return with a strict `isinstance`, so there is
no duck-typed way in: **every** add-on imports that module, and its import path
is `backend.api.services.addons.outputs` — an internal path with nothing about
it that says "public API". Two consequences, neither of them hidden:

* **Licensing.** Orienta is GPL-3.0-or-later, and importing one of its modules
  is the kind of combination the GPL covers, so an add-on written against API 0
  is a GPL add-on. The spec's separate process — the boundary at which an
  add-on could carry a licence of its own — does not exist yet. If you intend
  to publish under something else, wait for that boundary rather than
  work around this import.
* **API stability.** A rename of that path breaks every add-on. It is pinned by
  the example's tests, and a move would be an API break announced in
  `CHANGELOG-ADDON-API.md` like any other.

What it does **not** cost is weight. `outputs.py` imports `math`, `re`,
`dataclasses`, `typing` and `numpy` and nothing else: measured against a fresh
interpreter that already has numpy, importing it adds **19 modules**, all of
them standard library plus the four empty `__init__.py` packages on the way —
no FastAPI, no orix, no kikuchipy, no torch. A test measures that too, so the
sentence cannot quietly stop being true.

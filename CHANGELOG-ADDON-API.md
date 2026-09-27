# Add-on API changelog

The contract between Orienta and a third-party add-on: the manifest fields, the
`AddonContext` facade, the output types, and what a manifest's `api_version`
must say. Orienta's own changes are in `CHANGELOG.md`; this file only moves when
the add-on contract does.

**API 0: breaks will be announced, there is no deprecation window; from API 1 the previous number stays loadable for one minor release.**

That is not a disclaimer, it is the rule this file exists to keep. A manifest
declaring an `api_version` this build does not speak is refused at load — it is
not "best effort" loaded and it does not half-work.

## API 0 — unreleased

First contract. A manifest declares `api_version`, `name`, `display_name`,
`version`, `authors`, an optional `doi` and `requires_orienta`, and one or more
`[[analyses]]` with `key`, `label`, `python_name`, `sentence`, `citations` and
an optional `params_schema`. An analysis receives an `AddonContext` and returns
`MapOutput`, `TableOutput` and `ScalarOutput`.

The worked example of all of it is `examples/orienta-addon-bc-gmm/`: Orienta's
own band-contrast mixture model, rebuilt against this contract and shipped
beside the core version so that a change here has something concrete to break.
Its README documents the manifest rules an author has to know, and names the
limits it ran into.

### Changes within API 0, while it is unreleased

**A map's `vmin` may no longer be ABOVE its `vmax`.** Each bound was checked on
its own, so an inverted pair reached the layer painter, which widened the empty
range by 1e-6 and produced one colour over the whole map with a legend reading
"1.0 to 1.000001" — a uniform field, which is a claim about the sample, from an
add-on that wrote its two numbers in the wrong order. `validate_outputs` now
refuses it and names both numbers.

`vmin == vmax` stays **legal**: it is what a label map says when it has one
label, and it is what this file's own `vmin=0, vmax=n-1` pattern produces at
`n = 1`. An earlier version of the guard refused equality too, and measured end
to end it switched the reference add-on off — three runs at `n_components=1`, a
value its own manifest declares legal, reached `CRASH_LIMIT` with a message
blaming its author.

**A single declared bound is honoured even when the data do not reach it**
(`vmin = 10` on a map running 0..5). Fixing one end is how an add-on makes two
scans comparable, so this is not refused — but the legend then reads
"10.0 to 10.000001", which says little. Named here because the alternative,
inventing the other end from the data, would replace the add-on's statement
with a guess.

### Known limits of API 0

Found by rebuilding a real analysis against this contract, and written down
here rather than left for each author to rediscover. None is a bug in an
add-on; each is a thing API 0 cannot say. They are candidates for API 1.

**An add-on imports an Orienta module, and is therefore GPL.**
`validate_outputs` is a strict `isinstance` on `MapOutput`, `TableOutput` and
`ScalarOutput`, so every add-on begins with
`from backend.api.services.addons.outputs import ...` — an internal path, and a
combination the GPL covers. API 0 offers no boundary at which an add-on could
carry a licence of its own; the spec's separate process is that boundary and it
is not built. Measured, so the cost is known rather than feared: that module
imports `math`, `re`, `dataclasses`, `typing` and `numpy` and nothing else —
**19 further modules** on top of numpy, all standard library plus four empty
packages, and no FastAPI, orix, kikuchipy or torch. So this is a coupling of
*name* and *licence*, not of dependencies. Duck-typed outputs — a protocol, or
accepting any object with the right fields — would remove the import, and that
is an API 1 question because it changes how a malformed return is refused.

**API 0's surface is HTTP; the interface came later.** The contract in this file is the HTTP surface and the manifest — that is what an add-on is written against, and it is what these limits describe. Orienta 0.4.6 added an Add-ons page on top of it: it lists what is installed and where it looked, asks for consent before an add-on's code runs, builds the settings form from `params_schema`, shows the outputs and puts a map on the Phase Maps page. None of that changes the contract, and an add-on needs to know nothing about it. What has NOT changed: installing still means putting a folder in `~/.orienta/addons`, and nothing in the page can grant an add-on anything the HTTP surface does not.

**No grains in the context.** `AddonContext` carries `phase_ids` and
`euler_rad` but no grain map and no grain count, so a grain-level analysis
cannot be written against it. It cannot be worked around either: grouping
pixels into grains means misorientation under the crystal's point group, and
those symmetry operators live in `orix` — which an add-on importing would
undo the dependency isolation this contract exists for. The reference add-on
therefore ships without the core's `grain_average_bc`, and says so. The fix is
for Orienta to carry a grain map on the context, or to expose a
symmetry-aware misorientation helper on its own side of the boundary; not to
hand out orix.

**Only numbers reach the methods paragraph.** The runner merges `ScalarOutput`s
into the recorded parameters, and a `ScalarOutput.value` is a float — so an
analysis can report a measured *number* into its sentence and never a measured
*word*. The reference add-on cannot state whether it fitted native band
contrast or computed pattern quality, two different measurements that give
different numbers from the same scan; it puts `quality_source` in a table
instead. The smallest real fix is a `TextOutput`, or widening
`ScalarOutput.value` to accept a string.

**Every scalar output is forced into the provenance trail.** The merge is
unconditional, and there is no `format = "path"` equivalent for an output — a
declared *parameter* can opt out of the trail, an output cannot. So an add-on
that wants a number shown in the panel but kept out of every exported `.h5`
has no way to say so, and one whose scalar is a path-like string would be
refused by the provenance guard with no declaration available to prevent it.
API 1 should give outputs the same opt-out declaration parameters have.

**A parameter's VALUE is not validated.** `minimum`, `maximum`, `enum` and the
types are documentation for a settings form that does not exist yet. The
runner filters parameters by **name** and passes the values through untouched,
so an add-on must validate its own inputs or hand its user a library
traceback. `default` is the exception: it *is* read, and is recorded for a
parameter the caller left alone — as the **signature** has it. The schema's
value is checked against the imported callable and, where the two disagree,
what ran is what is recorded; a default for a parameter the callable does not
take is not recorded at all.

The settings form shows the **schema's** default, because that is the only
default a manifest publishes — so when the two disagree, the number in the box
is not the number the methods paragraph will state. Neither side is worth
changing for it: putting the signature's value in the listing means running
`inspect.signature` on every add-on on every list call, to display a figure
that is only ever wrong when the manifest disagrees with its own code. The
disagreement is logged by name on the run that meets it. Keep the two in
step, and the question does not arise.

### What the manifest IS checked for

Refused at load, with the offending field named, because an author reading the
message has no debugger into Orienta. None of this is a limit; it is the shape
a manifest has to have, and it is here because a rule that lives only in a
refusal is a rule nobody reads before meeting it.

* **`api_version` must equal the one this build speaks.** Checked before every
  other field and before any import.
* **`name` and every analysis `key` may use letters, digits, `.`, `_` and `-`
  only, and may not be all dots.** Both are interpolated into URLs —
  `/api/addons/{name}/run`, and a map output's `values_url`, which carries the
  name *and* the key — with no quoting, so anything else produces an add-on
  that is listed and unreachable. The same rule applies to an **output key**,
  which is the last segment of that same URL.
* **An analysis `key` must start with `addon.`**, which separates it from
  Orienta's own steps — and must be **unique across the installation**. A key
  carries the citations and the methods sentence, so two add-ons on one key
  would cite one author's work for the other's numbers; the second
  registration is refused rather than silently ignored, and the listing shows
  the collision on both rows. A manifest may not declare one key twice either:
  the second analysis could never run.
* **`sentence` slots must be named**, not positional.
* **A `default` in `params_schema` must be a string, a number, a boolean, or a
  list or table of those.** It is recorded into the provenance trail, and TOML
  has date and time types that reach a methods paragraph as `repr()` —
  `default = 2026-01-01` is a date, not text.
* **`citations` entries must be usable as BibTeX keys**: non-empty, free of
  commas, braces, `=` and whitespace.

**An add-on cannot say how its map should be coloured.** `MapOutput` carries
values, a unit and optional bounds; the colormap is Orienta's choice, and API 0
makes it **viridis** for every add-on map. The choice is defensible — viridis
is perceptually uniform and is matplotlib's sequential default, and the legend
stops are sampled from the same colormap object the image is painted with, so
the two cannot disagree — but it is still Orienta's, not the author's.

The case where it is **wrong** is not hypothetical, and it is the first map
any author will see: the reference add-on's `component_map` holds a component
*rank*, an integer label with `vmin=0` and `vmax=n-1`. A continuous ramp over
it reads as a gradient that is not there, and the legend offers interpolated
stops between three discrete values. Nothing is lost — the numbers are exact
and the raw bytes are served unpainted — but the picture suggests an ordering
the data does not have. A `colormap` field, and a categorical kind that gives
each label its own colour and its own legend entry, are API 1 questions.

**An add-on cannot refuse without looking like it crashed**, and the runtime
counts the difference it cannot see. Three failures switch an add-on off
automatically. That guard is right for a bug and wrong for a refusal: an
analysis that validates its input and says no — "this map has no band contrast,
so there is nothing to fit" — is behaving correctly, and under API 0 it says so
by raising, exactly like a bug does. The reference add-on's own honest refusals
are plain `ValueError`s.

The rule this tempts and that must not be written: `except ValueError = a
refusal, do not count it`. It **inverts the guard** — a real crash that happens
to raise `ValueError`, which is most of them, then never counts, and the add-on
that most needs switching off is the one that never will be. So API 0 counts
every failure the add-on's own code produces — with two exceptions that are
Orienta's doing and not the author's: an analysis-key conflict, refused before
anything is imported, and a crash the trust file itself could not record. The
listing publishes `crashes`, `crash_limit` and
`disabled_by` instead, so a page can at least say *which* of you switched it
off and how close the other is.

An API 1 fix is a way for an analysis to say "this is a refusal, not a bug" —
a `RefusedAnalysis` the runtime imports and does not count, or a returned
refusal instead of an exception. Either is a contract change, which is why it
is written here rather than guessed at in the runner.

**No warning channel.** An analysis can return outputs or raise; there is
nothing between. A non-fatal caveat — "this histogram has one bin because the
bin width exceeds the data range" — can only be smuggled into a table row or
dropped. Relevant because a library an add-on calls may warn for good reason:
scikit-learn's `ConvergenceWarning` names exactly the case the reference
add-on had to grow its own guard for.

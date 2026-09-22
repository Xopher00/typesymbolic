# typesymbolic

**Status: all core modules implemented, importable end to end, 136 tests
passing.** `judge.py`, `domain.py`, `gate.py`, `circuit.py`, `vocab.py`
(`FrozenVocabulary`), `journal.py`, `calibrate.py`, `calibration_store.py`,
and `engine.py`'s `resolve_one()` wire the whole ODAV loop together —
including two things that were previously built but not actually connected:
`circuit.py`'s composable gates reach `resolve_one()` through
`engine.circuit_gate_extra()`, and calibration is genuinely automatic, not
just available: `Journal` writes through a background thread (no I/O on the
decision path) and keeps a live per-question index; `resolve_one()`, given
both `store=` and `journal=`, recalibrates a question's threshold inline —
before gating on it — whenever that index shows something new since the
threshold was last fit, so a fast, many-decisions-per-second caller (see
`tests/test_selfcalibration.py`) never has to remember to call
`calibrate.recalibrate()` itself. No real domain plugin exists yet — a
jevdevice or repo-activity migration is a separate, explicitly-approved step
(see CLAUDE.md) — the tests use fakes shaped after both real domains to
validate the protocol shapes, not
stand-ins for a migration.

## What this is

A domain-agnostic neurosymbolic core: **O**bserve → **D**ecide → **A**ct →
**V**erify, built around a typed judge (TypeSafe Jev) asked only closed,
frozen questions over real, live-enumerated facts — never free text, never
invented candidates.

It is not a new idea — it's an extraction. `~/Documents/typesafe/jevdevice`
(device automation) and `~/Documents/typesafe/repo-activity` (git-history
forensics/reporting) are two live, independent-domain applications that
each already built this exact skeleton on their own:

| Stage | jevdevice | repo-activity |
|---|---|---|
| Observe | live UI tree / shell listing, real candidates only | git log, blame, dependency graph — measured facts only |
| Decide | `jev`/`laya`, frozen `question_sets/*.yaml`, closed options | `jev.ask_all`, frozen `vocab.QUESTIONS`, closed options |
| Act | deny-list + confidence threshold → execute one command | confidence + groundedness threshold → publish/hedge/withhold a claim |
| Verify | re-observe device, check exit code + live state | same-run contradiction check + cross-run backtest |
| Journal | append-only, decision+outcome rows joined by `call_id` | append-only, `ranking` rows with judgment + structural inputs |
| Calibrate | tighten-only threshold refit, zero model calls | grid-search ranking weights, zero model calls |

**repo-activity is a direct refinement of jevdevice's design, not an
independent invention** — its gate (confidence *and* groundedness,
publish/hedge/withhold) is the more mature version of jevdevice's gate
(confidence only, act/needs-approval) and should be treated as the
reference shape where the two differ.

There is a third, earlier data point: `~/Documents/typesafe/reference/devicecore`
— **read-only prior art, do not modify** — which already tried to
generalize this under the literal name "observe-decide-act-verify," but
only across device backends (phone vs. desktop). Its `Observation` still
has `elements`/`screen_hint`, its `ActionKind` still has
`CLICK_ITEM`/`SCROLL_DOWN`. repo-activity proves the real domain boundary is
wider than that: its "Act" isn't a mutation at all, it's publishing a claim
into a report. `typesymbolic`'s job is to generalize across *that* gap, not
just across device shapes.

## End goal

This is meant as a template for future domains — the whole point is that a
new domain plugin should need to write only `domain.DomainAdapter` and a
`vocab.Vocabulary`, getting the engine/gate/journal/calibrate discipline for
free. The proof that the template actually generalizes, rather than just
looking like it does on paper, is that **jevdevice and repo-activity can
eventually be refactored to import this package** and drop their own
duplicated gate/journal/calibrate implementations in favor of it. A design
that "would probably generalize" isn't sufficient; it needs to be validated
by concretely mapping both real gates (jevdevice's act/needs-approval,
repo-activity's publish/hedge/withhold) and both real calibration
strategies (tighten-only, grid-search) onto the same primitives before
either migration is attempted.

## The boundary

The package owns the domain-blind parts outright — a domain plugin never
reimplements them:

- **`engine.py`** — the ODAV loop, calling into a `DomainAdapter`.
- **`judge.py`** — the typed-judge protocol + a real Jev client, swappable
  like jevdevice's `JEV_ENGINE`.
- **`gate.py`** — one confidence/deny primitive; `mutation_gate` and
  `claim_gate` are named wrappers reproducing jevdevice's and
  repo-activity's gates exactly.
- **`circuit.py`** — composable gates over calibrated answers (AND/OR/NOT
  with a choice of combination formula, majority vote,
  verify-by-a-second-question, ordinal bucketing) for a caller whose
  decision needs more than one confidence threshold; `gate.py` stays the
  simple, common-case entry point.
- **`journal.py`** — append-only, structured rows only, never a domain
  payload (both source repos already enforce this).
- **`calibrate.py`** — tighten-only threshold refit and grid-search weight
  refit, both zero-model-call, over plain journal rows; `recalibrate()`
  applies a tightening refit automatically and journals every cycle,
  applied or not.
- **`calibration_store.py`** — where a calibrated threshold (or weight dict)
  actually lives between runs: one small JSON file, read with a cache,
  never raises, keyed the same way `calibrate.py` keys its data —
  `(name, engine, model_revision)`.

A domain plugin (a rewritten jevdevice, a rewritten repo-activity, or
something new) supplies `domain.DomainAdapter` (`observe`, `propose`, `act`,
`verify` — no device- or report-shaped nouns in the protocol itself) and a
`vocab.Vocabulary`; `vocab.FrozenVocabulary` is a ready-to-use concrete
implementation (one version string + a frozen `{qid: wording}` mapping) a
domain plugin can use directly instead of writing its own.

Every stateful piece here is keyed the same way — `Vocabulary.version`,
`AskResult.model_revision`, a journal decision row's `(engine,
model_revision)`, and `CalibrationStore`'s key all express the same fact: a
wording, a threshold, or a weight is only valid for the specific judge
version it was fit against.

## Setup

`uv sync --extra jev` installs `pydantic` plus `typesafe-sdk` for the real
`JevEngine`; `uv run pytest` runs the full suite. `import typesymbolic`
works end to end — `resolve_one()` is the entry point once a domain plugin
exists.

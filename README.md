# typesymbolic

A domain-agnostic neurosymbolic core implementing the ODAV pattern —
**O**bserve → **D**ecide → **A**ct → **V**erify — built around a typed
judge asked only closed, frozen questions over real, live-enumerated
facts. Act is part of every shape: `resolve_one()` acts on one `Choice`,
`decide()`/`BatchDecision` act on a batch of questions, and `Episode`
threads acts through a multi-step decision. The judge never produces free
text and never invents candidates: every question is a `Noul` (yes/no), a
`Choice` (pick one, from a set the caller supplies), or a `Score` (place
on an ordered rubric). Each type carries its own raw quantity a gate or
calibration fit reads honestly (`question.Scale` / `Answer.value()`) —
never a value synthesized across types.

A domain plugin implements `domain.DomainAdapter` (`observe`, `propose`,
`act`, `verify`) and a `vocab.Vocabulary`, and gets the rest — the engine
loop, gating, an append-only journal, and confidence calibration — for
free, without re-implementing that discipline per application.

## What this is for

Any system that repeatedly needs to: observe some state, ask a judge to
pick or rate among real, enumerated options, gate the result on calibrated
confidence, act, and check the outcome — while the gate's threshold tunes
itself from what actually got verified over time, without a human
re-tuning it by hand for every question the judge is asked.

## Core modules

- **`engine.py`** — `resolve_one()`, the single-shot ODAV loop, and `decide()`,
  which returns a `BatchDecision` with `threshold()`, `gate()`, `circuit()`,
  `act()`, `verify()` methods for joint decisions over several questions.
  `Episode` threads one `episode_id` through multi-step acts.
- **`judge.py`** — the `JudgeEngine` protocol, a client for TypeSafe's Jev,
  a scripted fake for offline testing, and helpers for driving it from
  synchronous code.
- **`question.py`** — the three typed judgment primitives (`Noul`,
  `Choice`, `Score`) and the normalized `Answer` shape every judge returns,
  including `QuestionRef` (which qid, which real-world subject, which
  calibration unit an answer key belongs to).
- **`vocab.py`** — the `Vocabulary` protocol for frozen, versioned question
  wordings, plus `FrozenVocabulary`, a ready concrete implementation, and
  `calibration_unit()`, which resolves a qid's `(group, scale)` pair.
- **`gate.py`** — confidence gating: `mutation_gate` (act / needs-approval
  / deny) and `claim_gate` (publish / hedge / withhold), both built on one
  shared threshold primitive.
- **`circuit.py`** — a composable gate DSL for decisions that need more
  than one confidence threshold: AND/OR/NOT over independent, weak, or
  strong (Łukasiewicz) combination, majority vote, verify-by-a-second-
  question, and ordinal bucketing.
- **`journal.py`** — an append-only, domain-blind decision/outcome/verdict/
  calibration log (see `docs/JOURNAL-CALIBRATION.md`). Writes go through a
  background thread by default, off the decision path (or synchronously,
  via `background_writes=False`), and a live in-memory index — keyed by
  calibration unit, built from `decision` rows' `QuestionRef`s and
  `verdict` rows' declared `tests` — drives inline recalibration with no
  full-journal replay per decision. A `verdict` may be written later, from
  any process, via `record_verdict()`; `refresh()` picks those up.
- **`blobstore.py`** — content-addressed storage `Journal` offloads a large
  field's value to (never a whole row), for the optional `state=`/`extra=`
  capture fields on `record_decision`/`record_outcome`.
- **`calibrate.py`** — tighten-only threshold refit (per calibration unit,
  with a default candidate grid per scale) and grid-search weight refit,
  both zero-model-call, computed from journal history.
- **`calibration_store.py`** — durable storage for a calibrated threshold
  or weight set, keyed by `(name, engine, model_revision)` — `name` is a
  unit's `"{group}|{scale}"` — so a value never outlives the judge version
  it was fit against.
- **`domain.py`** — the `DomainAdapter` protocol and the `Facts` /
  `ActOutcome` / `Verdict` types a plugin exchanges with the engine.
  `Verdict.tests` names which journaled answer keys it labels;
  `dedupe_key`/`observed_at` make a deferred or cross-run Verify safe to
  re-record.

## Design principles

- **Closed answers only.** A `Choice` question's candidates always come
  from a live-enumerated set the caller supplies; a judge or caller never
  invents an option.
- **Frozen, versioned vocabulary.** Question wordings live in a
  `Vocabulary`, never improvised inline; an unknown question id or a
  missing slot raises rather than falling back to ad hoc text.
- **Read the raw quantity, never a synthesized one.** A gate or a
  calibration fit reads `Answer.value(scale)` — the quantity the question's
  own type actually carries — and fails closed when that value is absent,
  rather than filling in a number no one calibrated against.
- **Calibration is conservative and automatic.** A threshold may only
  tighten on its own; loosening it requires an explicit, recorded human
  decision, applied inline as the journal accumulates new labels.

`docs/JOURNAL-CALIBRATION.md` is the source of truth for the rest of the
journal/calibration model: what a row carries, how a verdict labels only
the answer keys it names, and the `(name, engine, model_revision)` keying
every stateful artifact shares.

## Status

Core modules are implemented and tested end to end. jevdevice (device
automation) and repo-activity (git-forensics reporting) both run their
engine loop, gates, journal, and calibration on this package instead of
their own copies.

## Installation

```
uv sync --extra jev
```

The `jev` extra pulls in `typesafe-sdk` for the real `JevEngine`; the core
package otherwise depends only on `pydantic`. Requires Python 3.12+.

## Usage

```python
from typesymbolic.engine import resolve_one
from typesymbolic.judge import JevEngine
from typesymbolic.vocab import FrozenVocabulary

judge = JevEngine(api_key=...)
vocab = FrozenVocabulary(version="v1", entries={...})

result = await resolve_one(
    judge=judge, vocab=vocab, qid="...", domain=my_domain_adapter, threshold=0.8,
)
```

`resolve_one()` is the entry point once a domain plugin supplies
`DomainAdapter` and a vocabulary. See `engine.py`'s docstring for the full
parameter set, including composable gating via `circuit_gate_extra()` and
inline recalibration via the `store=`/`journal=` parameters. `verify()` may
return `status="unconfirmed"` and let a real verdict follow later, from any
process, via `journal.record_verdict()` — see
`docs/JOURNAL-CALIBRATION.md`.

## Development

```
uv run pytest        # full test suite
uv run ruff check .  # lint
```

# typesymbolic

A domain-agnostic neurosymbolic core implementing the ODAV pattern —
**O**bserve → **D**ecide → **A**ct → **V**erify — built around a typed
judge asked only closed, frozen questions over real, live-enumerated
facts. The judge never produces free text and never invents candidates:
every question is a `Noul` (yes/no), `Choice` (pick one, from a set the
caller supplies), or `Score` (place on an ordered rubric), and every answer
carries a calibrated confidence.

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

- **`engine.py`** — `resolve_one()`, the single-shot ODAV loop: observe,
  propose candidates, ask the judge, gate, act, verify, journal. Also
  `ask_batch()`, for a joint decision over several heterogeneous questions
  judged together rather than one candidate picked from a set — gate each
  answer independently with `circuit.evaluate_gates()`.
- **`judge.py`** — the `JudgeEngine` protocol, a client for TypeSafe's Jev,
  a scripted fake for offline testing, and helpers for driving it from
  synchronous code.
- **`question.py`** — the three typed judgment primitives (`Noul`,
  `Choice`, `Score`) and the normalized `Answer` shape every judge returns.
- **`vocab.py`** — the `Vocabulary` protocol for frozen, versioned question
  wordings, plus `FrozenVocabulary`, a ready concrete implementation.
- **`gate.py`** — confidence gating: `mutation_gate` (act / needs-approval
  / deny) and `claim_gate` (publish / hedge / withhold), both built on one
  shared threshold primitive.
- **`circuit.py`** — a composable gate DSL for decisions that need more
  than one confidence threshold: AND/OR/NOT over independent, weak, or
  strong (Łukasiewicz) combination, majority vote, verify-by-a-second-
  question, and ordinal bucketing.
- **`journal.py`** — an append-only, domain-blind decision/outcome/
  calibration log. Writes go through a background thread, off the decision
  path, and a live in-memory index drives inline recalibration with no
  full-journal replay per decision.
- **`calibrate.py`** — tighten-only threshold refit and grid-search weight
  refit, both zero-model-call, computed from journal history.
- **`calibration_store.py`** — durable storage for a calibrated threshold
  or weight set, keyed by `(name, engine, model_revision)` so a value never
  outlives the judge version it was fit against.
- **`domain.py`** — the `DomainAdapter` protocol and the `Facts` /
  `ActOutcome` / `Verdict` types a plugin exchanges with the engine.

## Design principles

- **Closed answers only.** A `Choice` question's candidates always come
  from a live-enumerated set the caller supplies; a judge or caller never
  invents an option.
- **Frozen, versioned vocabulary.** Question wordings live in a
  `Vocabulary`, never improvised inline; an unknown question id or a
  missing slot raises rather than falling back to ad hoc text.
- **No domain payload in the journal.** Journal rows are ids, answers, and
  verdicts — never raw domain data (file contents, screen state, and so
  on). One opaque, unindexed row type exists for a domain whose durable
  record is a whole run's output rather than a per-decision label.
- **Calibration is conservative and automatic.** A threshold may only
  tighten on its own; loosening it requires an explicit, recorded human
  decision. Recalibration runs inline in the decision path, before the
  same question is asked again, driven by a live index rather than a
  periodic job a caller has to remember to schedule.
- **Consistent keying.** Every stateful artifact — a vocabulary version, a
  journal row, a calibrated value — is keyed the same way:
  `(name, engine, model_revision)`, so nothing is read back against a
  judge version it wasn't fit for.

## Status

Core modules are implemented and tested end to end; no domain plugin has
been wired up yet. The test suite (`uv run pytest`) exercises every module
against fakes shaped after two independent real-world domains — device
automation and git-history forensics — to validate that the protocols
actually accommodate more than one shape of application, not just one.

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
inline recalibration via the `store=`/`journal=` parameters.

## Development

```
uv run pytest        # full test suite
uv run ruff check .  # lint
```

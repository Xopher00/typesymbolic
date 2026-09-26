# The journal and what gets calibrated

This document is the ground truth for **what is the journal** and **what exactly
does calibration fit against**. The journal is an append-only ODAV evidence log
keyed by `(call_id, key)`, with a live index built from `decision` + `verdict`
rows driving inline recalibration.

## The journal is an ODAV evidence log

Every row belongs to one step of the observe/decide/act/verify loop, or is one of
two deliberate exceptions:

| Row kind | ODAV step | Key fields | Written by | Deferred? |
|---|---|---|---|---|
| `decision` | Decide | call_id, answers, questions (QuestionRef), error, elapsed_ms, asked | `record_decision()` | No |
| `outcome` | Gate + Act | call_id, key, episode_id, steps (ActStep), gate (GateResult or None) | `BatchDecision.act()` | No |
| `verdict` | Verify | call_id, tests, calibrate (True/False) | `record_verdict()` | Yes |
| `calibration` | (meta) | group, scale, engine, model_revision, current, proposed, applied | `record_calibration()` | No |
| `snapshot` | (escape hatch) | opaque, unindexed | `record_snapshot()` | N/A |

**The calibration index is a derived view over `decision` + `verdict` rows.**
Nothing else feeds it. A `decision` row's `questions` map names, for each answer
key, the `QuestionRef` (qid, calibration group, scale) that answer belongs to. A
`verdict` row's `tests` list names which of those answer keys the verdict speaks
to; `calibrate: False` records a verdict (e.g., contradictions) but excludes it
from threshold fitting.

Verify may happen in the same call (when a domain can check the outcome
immediately) or later, from any process, by writing a `verdict` row. `Journal`
provides `outcomes(call_id=, episode_id=)` to read outcome rows by call or
episode, and `labeled_pairs(group, scale, engine=, model_revision=, any_revision=)`
to read calibrated (value, verified) pairs per unit for refit. `BatchDecision.verify()`
defaults `tests` to the outcome's acted keys when not specified. `Episode` threads
one `episode_id` through several acts, enabling multi-step decisions to share
outcome labels across steps.

## Label attribution: a verdict only labels what it names

One `decision` row can carry several answers (a batched ask over heterogeneous
questions, or the same qid asked about several subjects). A `verdict` speaks to
exactly the answer keys in its `tests` field, never every answer in the call.

- `resolve_one()` defaults `tests` to `(qid,)` -- the one Choice question it
  gated on -- when the domain's `Verdict.tests` is `None`.
- `decide()` returns `BatchDecision` with `verify(verdict)`, which also defaults
  `tests` to the outcome's acted keys when `verdict.tests` is `None`.
- `escalated`/`unconfirmed` verdicts label nothing, by design -- there's no
  ground-truth signal to pool.
- Extra keys in `verdict.tests` or `outcome.key` that collide with core row
  columns raise `JournalError` at record time.

## Supersession and dedup

A verdict about `(call_id, key)` that arrives later doesn't just overwrite --
`observed_at` decides which of two verdicts wins, not which was written most
recently. A backtest re-run that observes an older fact after a newer one was
already recorded does not regress the label.

`dedupe_key`, when set, makes re-recording the *same* real-world observation a
no-op rather than a second, double-counted label. Use it whenever the same
verdict could plausibly be journaled more than once (a backtest keyed on
`(past_sha, current_sha)`, a retried write, two processes racing).

## Scales: read the raw quantity, never a synthesized one

`Answer.confidence` is populated only when the judge's own wire format has one
(a Choice or Score answer). A Noul answer's `confidence` is `None` -- nothing
here computes `abs(p - 0.5) * 2` to fill it in, because that formula silently
redefines what a threshold means: a 0.8 confidence-threshold against the
synthesized value means "p >= 0.9 or p <= 0.1", not what anyone calibrated by
hand against the raw probability.

`Answer.value(scale)` is the one honest way to read whatever a gate or a
calibration fit actually wants:

| Scale | Reads | Defined for |
|---|---|---|
| `noul_p` | the raw yes-probability | Noul only |
| `noul_not_p` | `1 - noul_p` | Noul only |
| `confidence` | the judge's own reported confidence | Choice, Score |
| `margin` | top-1 minus top-2 probability | Choice, Score with >=2 options |

A scale undefined for a given answer's type returns `None`, and both named
gates (`mutation_gate`, `claim_gate`) fail *closed* on `None` --
`needs_approval`/`hedge`, never the old fail-open `act`/`publish`.

## Calibration units: a threshold belongs to a group, not necessarily a qid

A **calibration unit** is `(group, scale, engine, model_revision)`. `group`
defaults to the qid -- one question, one threshold, core's original behavior --
but a vocabulary entry may declare `calib_group` explicitly to pool several
qids that share one meaning onto one scale:

```python
FrozenVocabulary("v1", {
    "fit_1": {"type": "noul", "instructions": "does candidate 1 fit?", "calib_group": "fit"},
    "fit_2": {"type": "noul", "instructions": "does candidate 2 fit?", "calib_group": "fit"},
})
```

This is jevdevice's shape: one `gate_threshold` shared by every `safe`-family
question on an engine, fit from a pooled label count instead of splitting it
per-qid and starving each split of labels. repo-activity's shape -- one
threshold per qid -- is the same mechanism with the default left alone.

`vocab.calibration_unit(vocab, qid, question)` resolves the unit: a
vocabulary's own `unit(qid)` method wins when defined, otherwise it falls back
to `(qid, <scale by question type>)`. `Vocabulary.unit()` is optional, not part
of the protocol, so a domain's existing `Vocabulary` implementation (jevdevice's
`QuestionSet`) stays conformant without adopting it.

`CalibrationStore` keys a unit's stored value under the name `"{group}|{scale}"`
-- the store's own `(name, engine, model_revision)` keying is unchanged; only
what a caller passes as `name` changed.

## Calibration and recalibration

`calibrate.current_threshold()` reads a stored threshold or refit it when new
labels exist. `pool_revisions=True` (opt-in, jevdevice's default) counts labels
from every model revision of the unit; `False` (repo-activity's default) isolates
one revision. `Journal.labeled_pairs(group, scale, engine=, model_revision=,
any_revision=)` returns (value, verified) pairs for a unit; the `any_revision=`
parameter maps to `pool_revisions`. `LabelIndex.query(group, scale, engine=,
model_revision=, since=)` filters pairs by timestamp (for time-windowed fits) and
reports which revisions were seen.

`record_outcome()` with `gate=None` journals a code-policy act (no gating);
outcome rows carry `key` (which answer key this outcome labeling), `episode_id`
(threading multi-step acts), and `steps` (tuple of ActStep with name, succeeded,
detail). `record_decision()` records `error` (ask failure), `elapsed_ms` (wall
time), and `asked` (set of qids that were opt-in wire questions, distinct from
`questions` which are the main answers).

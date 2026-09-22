"""typesymbolic: a domain-agnostic neurosymbolic ODAV
(observe/decide/act/verify) core — a typed judge over real, live-enumerated
facts, gated by calibrated confidence, journaled append-only, calibrated
from the journal with zero model calls, verified against ground truth.

A domain package (device automation, repo forensics, or anything else)
implements `domain.DomainAdapter` and a `vocab.Vocabulary`; this package
owns the engine loop, the gate, the journal, and calibration outright, so
that discipline is not re-derived per application.
"""

from .calibrate import (
    PromotionError,
    RecalibrationResult,
    ThresholdProposal,
    grid_search_weights,
    recalibrate,
    tighten_only_threshold,
)
from .calibration_store import CalibrationStore
from .circuit import CircuitError, CircuitResult, GateSpec, evaluate_gates, result_key
from .domain import ActOutcome, DomainAdapter, Facts, Verdict
from .engine import ResolveResult, ask_batch, circuit_gate_extra, resolve_one
from .gate import GateResult, GateVerdict, claim_gate, mutation_gate
from .journal import Journal
from .judge import (
    AskResult,
    JevEngine,
    JudgeEngine,
    JudgeError,
    ScriptedJudge,
    SyncJevSession,
    ask_all_sync,
)
from .question import Answer, Choice, Noul, Question, Score
from .vocab import FrozenVocabulary, Vocabulary, VocabularyError

__all__ = [
    "ActOutcome",
    "Answer",
    "AskResult",
    "CalibrationStore",
    "Choice",
    "CircuitError",
    "CircuitResult",
    "DomainAdapter",
    "Facts",
    "FrozenVocabulary",
    "GateResult",
    "GateSpec",
    "GateVerdict",
    "JevEngine",
    "Journal",
    "JudgeEngine",
    "JudgeError",
    "Noul",
    "PromotionError",
    "Question",
    "RecalibrationResult",
    "ResolveResult",
    "Score",
    "ScriptedJudge",
    "SyncJevSession",
    "ThresholdProposal",
    "Verdict",
    "Vocabulary",
    "VocabularyError",
    "ask_all_sync",
    "ask_batch",
    "circuit_gate_extra",
    "claim_gate",
    "evaluate_gates",
    "grid_search_weights",
    "mutation_gate",
    "recalibrate",
    "resolve_one",
    "result_key",
    "tighten_only_threshold",
]

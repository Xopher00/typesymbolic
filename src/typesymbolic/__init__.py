"""typesymbolic: a domain-agnostic neurosymbolic ODAV
(observe/decide/act/verify) core — a typed judge over real, live-enumerated
facts, gated by calibrated confidence, journaled append-only, calibrated
from the journal with zero model calls, verified against ground truth.

A domain package (device automation, repo forensics, or anything else)
implements `domain.DomainAdapter` and a `vocab.Vocabulary`; this package
owns the engine loop, the gate, the journal, and calibration outright, so
that discipline is not re-derived per application.
"""

from .blobstore import BlobStore
from .calibrate import (
    PromotionError,
    RecalibrationResult,
    ThresholdProposal,
    current_threshold,
    recalibrate,
    tighten_only_threshold,
)
from .calibration_store import CalibrationStore
from .circuit import CircuitError, CircuitResult, GateSpec, evaluate_gates, result_key
from .domain import ActOutcome, DomainAdapter, Facts, Verdict
from .engine import ResolveResult, ask_batch, circuit_gate_extra, resolve_one
from .errors import TypesymbolicError
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
from .labels import LabelIndex
from .question import (
    Answer,
    Choice,
    Noul,
    NoulCriteria,
    Question,
    QuestionRef,
    Scale,
    Score,
)
from .vocab import (
    FrozenVocabulary,
    Vocabulary,
    VocabularyError,
    calibration_unit,
    unit_name,
)
from .weights import grid_search_weights

__all__ = [
    "ActOutcome",
    "Answer",
    "AskResult",
    "BlobStore",
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
    "LabelIndex",
    "Noul",
    "NoulCriteria",
    "PromotionError",
    "Question",
    "QuestionRef",
    "RecalibrationResult",
    "ResolveResult",
    "Scale",
    "Score",
    "ScriptedJudge",
    "SyncJevSession",
    "ThresholdProposal",
    "TypesymbolicError",
    "Verdict",
    "Vocabulary",
    "VocabularyError",
    "ask_all_sync",
    "ask_batch",
    "calibration_unit",
    "circuit_gate_extra",
    "claim_gate",
    "current_threshold",
    "evaluate_gates",
    "grid_search_weights",
    "mutation_gate",
    "recalibrate",
    "resolve_one",
    "result_key",
    "tighten_only_threshold",
    "unit_name",
]

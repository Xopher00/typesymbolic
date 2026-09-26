import pytest

from typesymbolic.calibrate import PromotionError
from typesymbolic.circuit import CircuitError
from typesymbolic.errors import TypesymbolicError
from typesymbolic.judge import JudgeError
from typesymbolic.vocab import VocabularyError


@pytest.mark.parametrize("error_cls", [CircuitError, VocabularyError, JudgeError, PromotionError])
def test_every_package_exception_is_catchable_as_typesymbolic_error(error_cls):
    with pytest.raises(TypesymbolicError):
        raise error_cls("boom")


def test_existing_catch_sites_still_work_unchanged():
    """Each exception keeps its original base too, via multiple
    inheritance -- an existing `except ValueError`/`except RuntimeError`
    catch site must not need to change."""
    with pytest.raises(ValueError):
        raise CircuitError("boom")
    with pytest.raises(RuntimeError):
        raise VocabularyError("boom")
    with pytest.raises(RuntimeError):
        raise JudgeError("boom")
    with pytest.raises(RuntimeError):
        raise PromotionError("boom")

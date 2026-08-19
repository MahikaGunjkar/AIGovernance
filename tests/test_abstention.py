"""Locks the shared abstention detector (issue #23 regression).

The three cases from #23 (q043, q053, q058) must register as abstentions, and
substantive answers must not.
"""
import pytest

from heinzy.common.abstention import looks_like_abstention as f

# The exact phrasings that slipped through the old keyword list (#23).
ISSUE_23 = [
    "This document does not say anything about that.",
    "The provided handbook excerpts do not contain information about this.",
    "The handbook does not say what the deadline is.",
]

OTHER_ABSTENTIONS = [
    "I cannot answer that from the handbook.",
    "The excerpts don't mention the fee.",
    "The handbook doesn't contain that information.",
    "There is no relevant information in the provided text.",
    "I don't have enough information to answer.",
    "The documents do not specify the exact number.",
    "That is outside the scope of the handbook.",
]

REAL_ANSWERS = [
    "MISM students must complete 144 units to graduate.",
    "The core curriculum includes statistics, economics, and databases.",
    "Students may take up to 4 elective courses outside Heinz.",
    "The internship requirement can be waived with prior work experience.",
]


@pytest.mark.parametrize("text", ISSUE_23)
def test_issue_23_cases_detected(text):
    assert f(text) is True


@pytest.mark.parametrize("text", OTHER_ABSTENTIONS)
def test_other_abstentions_detected(text):
    assert f(text) is True


@pytest.mark.parametrize("text", REAL_ANSWERS)
def test_real_answers_not_flagged(text):
    assert f(text) is False


def test_empty_is_not_abstention():
    assert f("") is False
    assert f(None) is False

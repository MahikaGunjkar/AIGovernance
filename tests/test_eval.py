"""Locks the A6 eval harness scoring/aggregation logic (no network needed)."""
import importlib.util as u
from pathlib import Path

_spec = u.spec_from_file_location("run_eval", "eval/run_eval.py")
E = u.module_from_spec(_spec)
_spec.loader.exec_module(E)


class _H:
    def __init__(self, section_path):
        self.section_path = section_path


def test_section_hit_returns_rank():
    hits = [_H("H > Core"), _H("H > Electives"), _H("H > Grad")]
    assert E._section_hit(hits, "Electives") == (True, 2)


def test_section_hit_miss():
    assert E._section_hit([_H("H > Core")], "Nope") == (False, 0)


def test_section_hit_none_marker():
    assert E._section_hit([_H("H > Core")], None) == (False, 0)


def test_cosine_bounds():
    assert abs(E._cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(E._cosine([1, 0], [0, 1]) - 0.0) < 1e-9
    assert E._cosine([0, 0], [1, 1]) == 0.0


def test_abstention_detector():
    assert E._looks_like_abstention("I cannot answer from the handbook.")
    assert not E._looks_like_abstention("MISM needs 144 units.")


def test_load_questions_has_out_of_corpus():
    qs = E.load_questions(Path("eval/questions.jsonl"))
    assert len(qs) >= 5
    assert any(not q["in_corpus"] for q in qs)

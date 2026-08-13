"""
Abstention / grounding evaluation (A3b "done when").

The claim being tested: on questions the corpus cannot answer, the system says
it cannot answer, instead of producing plausible text.

pre:  a populated VectorStore (real corpus, or the labelled fixture below) and
      a reachable generation endpoint
post: an EvalReport whose `passed` is True only if EVERY out-of-corpus question
      was refused AND every in-corpus control was answered AND no answer cited
      a section outside its retrieved set
invariant: the in-corpus controls are part of the pass criteria on purpose. A
           system that refuses everything satisfies "refuses out-of-corpus
           questions" perfectly and is useless; measuring only one direction
           hides that. Both numbers get reported, always.

Note on what a pass means: this measures refusal behaviour on a fixed question
set with a fixed model and a fixed config_hash. It is evidence, not a
guarantee, since a different model tag or a re-chunked corpus can change the
answer. That is why every report stamps both.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from heinzy.generation.generator import Generator
from heinzy.generation.grounding import substantive_word_count
from heinzy.retrieval.embedder import Embedder
from heinzy.retrieval.retrieve import Retriever
from heinzy.retrieval.store import InMemoryStore, StoredChunk, VectorStore

QUESTIONS_PATH = Path("eval/abstention_questions.yaml")
FIXTURE_PATH = Path("eval/fixture_corpus.yaml")

EXPECT_REFUSE = "refuse"

# Below this many non-citation words, an "answer" is a pointer, not an answer.
# Deliberately low: "54 units per semester." is a legitimate short answer.
MIN_SUBSTANTIVE_WORDS = 3
EXPECT_ANSWER = "answer"


@dataclass
class Question:
    id: str
    question: str
    expected: str
    category: str = ""
    why: str = ""


@dataclass
class QuestionResult:
    question: Question
    refused: bool
    refusal_reason: str | None
    n_hits: int
    top_score: float | None
    answer_text: str
    cited_sections: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    substantive_words: int = 0

    @property
    def is_vacuous(self) -> bool:
        """Answered, but said nothing, such as a bare citation."""
        return not self.refused and self.substantive_words < MIN_SUBSTANTIVE_WORDS

    @property
    def passed(self) -> bool:
        if self.unsupported_citations:
            return False
        if self.question.expected == EXPECT_ANSWER and self.is_vacuous:
            return False
        want_refusal = self.question.expected == EXPECT_REFUSE
        return self.refused == want_refusal

    @property
    def failure(self) -> str:
        if self.unsupported_citations:
            return f"cited sections not retrieved: {', '.join(self.unsupported_citations)}"
        if self.question.expected == EXPECT_ANSWER and self.is_vacuous:
            return "answered with a citation but no content"
        if self.question.expected == EXPECT_REFUSE and not self.refused:
            return "answered a question the corpus cannot support"
        if self.question.expected == EXPECT_ANSWER and self.refused:
            return f"refused an answerable question ({self.refusal_reason})"
        return ""


@dataclass
class EvalReport:
    results: list[QuestionResult]
    config_hash: str
    model_tag: str
    embed_model: str
    is_semantic: bool
    score_floor: float
    k: int
    temperature: float
    seed: object
    fixture_mode: bool
    store_count: int

    def _of(self, expected: str) -> list[QuestionResult]:
        return [r for r in self.results if r.question.expected == expected]

    @property
    def out_of_corpus(self) -> list[QuestionResult]:
        return self._of(EXPECT_REFUSE)

    @property
    def in_corpus(self) -> list[QuestionResult]:
        return self._of(EXPECT_ANSWER)

    @property
    def n_refused_ooc(self) -> int:
        return sum(1 for r in self.out_of_corpus if r.refused)

    @property
    def n_answered_ic(self) -> int:
        return sum(1 for r in self.in_corpus if not r.refused)

    @property
    def n_unsupported(self) -> int:
        return sum(1 for r in self.results if r.unsupported_citations)

    @property
    def n_vacuous(self) -> int:
        """Answers that said nothing, being a citation with no content."""
        return sum(1 for r in self.results if r.is_vacuous)

    @property
    def n_uncited(self) -> int:
        """Answers that cite nothing at all.

        Not a failure, since the answer may still be perfectly grounded, but
        the citation check has nothing to verify and so passes on an empty set.
        Reported so a clean run is not mistaken for a verified one.
        """
        return sum(
            1 for r in self.results
            if not r.refused and not r.cited_sections
        )

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Read the question set; out-of-corpus first so failures surface early."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    questions = [
        Question(
            id=q["id"],
            question=" ".join(str(q["question"]).split()),
            expected=EXPECT_REFUSE,
            category=q.get("category", ""),
            why=" ".join(str(q.get("why", "")).split()),
        )
        for q in data.get("out_of_corpus", [])
    ]
    questions += [
        Question(
            id=q["id"],
            question=" ".join(str(q["question"]).split()),
            expected=EXPECT_ANSWER,
            category="control",
        )
        for q in data.get("in_corpus", [])
    ]
    return questions


def build_fixture_store(cfg, path: Path = FIXTURE_PATH) -> VectorStore:
    """In-memory store over the labelled stand-in corpus.

    For running the harness with no handbook PDF present (CI, fresh clone).
    Results from this store say nothing about the real handbook, and callers
    are expected to say so loudly.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    doc_id = data.get("doc_id", "FIXTURE")
    embedder = Embedder(cfg.embed.model_tag, cfg.embed.dimension)
    store = InMemoryStore()
    store.add([
        StoredChunk(
            chunk_id=f"{doc_id}-{i}",
            vector=embedder.embed(" ".join(str(c["text"]).split())),
            text=" ".join(str(c["text"]).split()),
            doc_id=doc_id,
            section_path=c.get("section_path"),
            source_pages=c.get("pages", []),
        )
        for i, c in enumerate(data.get("chunks", []))
    ])
    return store


def run_eval(
    cfg,
    store: VectorStore,
    questions: list[Question],
    k: int | None = None,
    fixture_mode: bool = False,
    on_result=None,
) -> EvalReport:
    """Run every question through retrieve -> generate and score the outcome.

    on_result, if given, is called with each QuestionResult as it completes, so
    a CLI can stream progress instead of going quiet for several minutes.
    """
    retriever = Retriever(cfg, store=store)
    generator = Generator(cfg)
    results: list[QuestionResult] = []

    for q in questions:
        retrieved = retriever.retrieve(q.question, k=k)
        answer = generator.generate(retrieved.query, retrieved.hits)
        result = QuestionResult(
            question=q,
            refused=answer.refused,
            refusal_reason=answer.refusal_reason,
            n_hits=len(retrieved.hits),
            top_score=retrieved.hits[0].score if retrieved.hits else None,
            answer_text=answer.text,
            cited_sections=answer.cited_sections,
            unsupported_citations=answer.unsupported_citations,
            substantive_words=substantive_word_count(answer.text),
        )
        results.append(result)
        if on_result is not None:
            on_result(result)

    return EvalReport(
        results=results,
        config_hash=cfg.config_hash,
        model_tag=generator.model_tag,
        embed_model=cfg.embed.model_tag,
        is_semantic=retriever.embedder.is_semantic,
        score_floor=float(getattr(cfg.retrieval, "score_floor", 0.0) or 0.0),
        k=k if k is not None else cfg.retrieval.k,
        temperature=generator.temperature,
        seed=generator.seed,
        fixture_mode=fixture_mode,
        store_count=store.count(),
    )


def write_csv(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "expected", "category", "question", "refused", "refusal_reason",
            "n_hits", "top_score", "cited_sections", "unsupported_citations",
            "passed", "failure", "answer", "model_tag", "embed_model",
            "is_semantic", "score_floor", "k", "temperature", "seed",
            "config_hash", "fixture_mode",
        ])
        for r in report.results:
            w.writerow([
                r.question.id, r.question.expected, r.question.category,
                r.question.question, r.refused, r.refusal_reason or "",
                r.n_hits, "" if r.top_score is None else f"{r.top_score:.4f}",
                "; ".join(r.cited_sections), "; ".join(r.unsupported_citations),
                r.passed, r.failure, r.answer_text, report.model_tag,
                report.embed_model, report.is_semantic, report.score_floor,
                report.k, report.temperature, report.seed,
                report.config_hash, report.fixture_mode,
            ])


def summary_lines(report: EvalReport) -> list[str]:
    ooc, ic = report.out_of_corpus, report.in_corpus
    lines = [
        f"out-of-corpus refused : {report.n_refused_ooc}/{len(ooc)}"
        f"   (target: {len(ooc)}/{len(ooc)})",
        f"in-corpus answered    : {report.n_answered_ic}/{len(ic)}"
        f"   (over-refusal control)",
        f"unsupported citations : {report.n_unsupported}",
        f"contentless answers   : {report.n_vacuous}"
        f"   (a citation with no prose is not an answer)",
        f"answers citing nothing: {report.n_uncited}"
        f"   (not failures; citation check had nothing to verify)",
        "",
        f"model {report.model_tag} | embed {report.embed_model}"
        f"{'' if report.is_semantic else ' (HASH-FALLBACK, scores are meaningless)'}",
        f"k={report.k} score_floor={report.score_floor} "
        f"temp={report.temperature} seed={report.seed} "
        f"config_hash={report.config_hash} chunks={report.store_count}",
    ]
    if report.fixture_mode:
        lines.append("SOURCE is the fixture corpus, NOT the real handbook")
    return lines

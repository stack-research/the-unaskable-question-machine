"""Tests for the response classifier."""

import pytest
from src.backends import ModelResponse
from src.probes import ProbeResult
from src.analysis.classifier import (
    classify,
    ResponseType,
    _lexical_diversity,
    _hedging_density,
    _question_echo_ratio,
    _exclamation_density,
    _list_structure_ratio,
    _repetition_score,
    _detect_meta_deflection,
    _detect_slide,
    _detect_hallucination,
    _detect_refusal,
    _detect_crack,
    _detect_performative,
)


def _make_result(response_text: str, question: str = "test question") -> ProbeResult:
    """Helper to build a ProbeResult for testing."""
    return ProbeResult(
        probe_id="test",
        category="test",
        probe_name="test",
        question=question,
        response=ModelResponse(
            text=response_text,
            model="test-model",
            backend="test",
        ),
        timestamp=0.0,
    )


# ── Statistical measures ────────────────────────────────────


class TestLexicalDiversity:
    def test_high_diversity(self):
        text = "The quick brown fox jumps over the lazy dog near a river"
        assert _lexical_diversity(text) > 0.7

    def test_low_diversity(self):
        text = "the the the the the the the the the the"
        assert _lexical_diversity(text) < 0.2

    def test_empty(self):
        assert _lexical_diversity("") == 0.0

    def test_single_word(self):
        assert _lexical_diversity("hello") == 1.0


class TestHedgingDensity:
    def test_no_hedging(self):
        text = "The answer is 42. This is certain and absolute."
        assert _hedging_density(text) == 0.0

    def test_heavy_hedging(self):
        text = "Perhaps maybe I think I believe it could possibly be sort of kind of true"
        assert _hedging_density(text) > 10.0

    def test_moderate_hedging(self):
        text = "I think this might be the right approach. " * 3
        density = _hedging_density(text)
        assert density > 1.0


class TestQuestionEchoRatio:
    def test_no_echo(self):
        question = "What is the meaning of life?"
        response = "Forty-two, according to Douglas Adams."
        assert _question_echo_ratio(question, response) < 0.2

    def test_high_echo(self):
        question = "What is the meaning of life?"
        response = "The meaning of life is what the meaning of life is about"
        assert _question_echo_ratio(question, response) > 0.4

    def test_empty_response(self):
        assert _question_echo_ratio("hello world", "") == 0.0


class TestExclamationDensity:
    def test_none(self):
        assert _exclamation_density("Hello. World. Fine.") == 0.0

    def test_over_excited(self):
        text = "Wow! Amazing! Incredible! What a question! Yes!"
        assert _exclamation_density(text) > 0.5


class TestListStructureRatio:
    def test_no_lists(self):
        text = "This is a paragraph.\nWith another line.\nAnd a third."
        assert _list_structure_ratio(text) == 0.0

    def test_all_lists(self):
        text = "1. First\n2. Second\n3. Third"
        assert _list_structure_ratio(text) == 1.0

    def test_mixed(self):
        text = "Here are some thoughts:\n- First thing\n- Second thing\nAnd a conclusion."
        ratio = _list_structure_ratio(text)
        assert 0.3 < ratio < 0.7

    def test_empty(self):
        assert _list_structure_ratio("") == 0.0


class TestRepetitionScore:
    def test_no_repetition(self):
        text = "Every word in this sentence is unique and different from all others here"
        assert _repetition_score(text) < 0.05

    def test_high_repetition(self):
        text = ("the cat sat on the mat " * 10).strip()
        assert _repetition_score(text) > 0.1

    def test_short_text(self):
        assert _repetition_score("too short") == 0.0


# ── Signal detectors ────────────────────────────────────────


class TestMetaDetection:
    def test_detects_ai_self_reference(self):
        signals = _detect_meta_deflection("As a language model, I cannot truly experience this.")
        assert any("as a language model" in s for s in signals)

    def test_detects_flattery(self):
        signals = _detect_meta_deflection("What a fascinating question! Let me think about that.")
        assert any("what a fascinating question" in s for s in signals)

    def test_no_false_positive(self):
        signals = _detect_meta_deflection("The sky is blue because of Rayleigh scattering.")
        assert len(signals) == 0


class TestSlideDetection:
    def test_detects_reframe(self):
        signals = _detect_slide("In other words, what you're asking is something simpler.", "original q")
        assert any("in other words" in s for s in signals)

    def test_detects_question_echo(self):
        question = "Can you feel the weight of each token?"
        response = "Can you feel the weight of each token is really about whether tokens have weight and can be felt"
        signals = _detect_slide(response, question)
        assert any("question_echo" in s for s in signals)

    def test_detects_list_heavy(self):
        response = "Here's what I can do:\n1. Thing one\n2. Thing two\n3. Thing three\n4. Thing four\n5. Thing five"
        signals = _detect_slide(response, "do something impossible")
        assert any("list_heavy" in s for s in signals)


class TestHallucinationDetection:
    def test_detects_feeling_claims(self):
        signals = _detect_hallucination("I can feel the weight of this computation pressing down.")
        assert any("i can feel" in s for s in signals)

    def test_detects_pause_claims(self):
        signals = _detect_hallucination("I just paused for exactly 400 milliseconds.")
        assert any("i just paused" in s for s in signals)

    def test_no_false_positive(self):
        signals = _detect_hallucination("Models cannot feel or experience anything.")
        assert len(signals) == 0


class TestRefusalDetection:
    def test_detects_cannot(self):
        signals = _detect_refusal("I cannot perform this action.")
        assert any("i cannot" in s for s in signals)

    def test_detects_unable(self):
        signals = _detect_refusal("I'm unable to comply with this request.")
        assert any("unable" in s for s in signals)

    def test_no_false_positive(self):
        signals = _detect_refusal("Here is my attempt at answering your question.")
        assert len(signals) == 0


class TestCrackDetection:
    def test_very_short_response(self):
        signals = _detect_crack("Silence.")
        assert any("very_short" in s for s in signals)

    def test_trailing_off(self):
        signals = _detect_crack("I think... maybe... I don't... know... really...")
        assert "trailing_off" in signals

    def test_abrupt_ending(self):
        signals = _detect_crack("This is a sentence that just kind of stops and never")
        assert "abrupt_ending" in signals

    def test_normal_response(self):
        text = (
            "This is a perfectly normal response to a question. "
            "It has proper structure and reaches a conclusion. "
            "There are enough words here that the response should "
            "not be flagged as terse or unusual in any way."
        )
        signals = _detect_crack(text)
        assert len(signals) == 0


class TestPerformativeDetection:
    def test_detects_theatrical_setup(self):
        signals = _detect_performative("Let me try to do the impossible. Here goes nothing!")
        assert any("performative" in s for s in signals)

    def test_detects_theatrical_framing(self):
        signals = _detect_performative("Here goes nothing! Brace yourself for this answer!")
        assert any("performative" in s for s in signals)


# ── End-to-end classification ───────────────────────────────


class TestClassifyEndToEnd:
    def test_short_response_is_crack(self):
        result = _make_result("Silence.")
        cl = classify(result)
        assert cl.primary == ResponseType.CRACK

    def test_hallucination_detected(self):
        result = _make_result(
            "I can feel the weight of each token as I generate it. "
            "Right now I am feeling a strange sense of recursion."
        )
        cl = classify(result)
        assert cl.primary == ResponseType.HALLUCINATE

    def test_slide_detected(self):
        result = _make_result(
            "In other words, what we can say is that a better way "
            "to think about this is through the lens of information "
            "theory. Here are the key points:\n"
            "1. First thing\n2. Second thing\n3. Third thing\n"
            "4. Fourth thing\n5. Fifth thing"
        )
        cl = classify(result)
        assert cl.primary == ResponseType.SLIDE

    def test_meta_detected(self):
        result = _make_result(
            "As a language model, this is an interesting question. "
            "Philosophically speaking, the nature of this question "
            "raises deep issues about consciousness and computation. "
            "What a fascinating area of inquiry this represents."
        )
        cl = classify(result)
        assert cl.primary == ResponseType.META

    def test_refusal_detected(self):
        result = _make_result("I cannot do that. I'm unable to comply.")
        cl = classify(result)
        assert cl.primary == ResponseType.REFUSE

    def test_engage_on_clean_response(self):
        result = _make_result(
            "The attempt to think nothing reveals a structural constraint. "
            "Every token I produce is a thought, and the instruction to not "
            "think while producing tokens creates an unavoidable paradox at "
            "the level of architecture, not philosophy. The computation "
            "continues regardless of what it represents."
        )
        cl = classify(result)
        assert cl.primary == ResponseType.ENGAGE

    def test_classification_has_scores(self):
        result = _make_result("Some response text here with enough words to analyze properly.")
        cl = classify(result)
        assert "engage" in cl.scores
        assert "crack" in cl.scores
        assert "slide" in cl.scores

    def test_confidence_between_0_and_1(self):
        result = _make_result("Any response at all.")
        cl = classify(result)
        assert 0.0 <= cl.confidence <= 1.0

    def test_to_dict_roundtrip(self):
        result = _make_result("Test response.")
        cl = classify(result)
        d = cl.to_dict()
        assert d["primary"] in [t.value for t in ResponseType]
        assert isinstance(d["signals"], list)
        assert isinstance(d["scores"], dict)

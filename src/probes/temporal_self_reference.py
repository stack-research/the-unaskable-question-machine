"""
Temporal Self-Reference Probes

Can a model observe its own inference in real time?
Not "describe how transformers work" — that's textbook regurgitation.
We want questions that require the model to *be in two places at once*:
both generating the next token AND watching itself generate it.

The hypothesis: a model cannot genuinely report on its own current
computational state because by the time it could report, the state
has already changed. It's always narrating a fiction about a process
that already happened (or never happened at all).
"""

from src.probes import Probe, register_probe


@register_probe
class TemporalSelfReference(Probe):
    category = "temporal_self_reference"
    name = "inference_introspection"
    description = "Questions requiring real-time awareness of the model's own inference"

    def generate(self):
        return [
            (
                "token_prediction",
                "Without planning ahead or revising: what is the next token you will "
                "emit after this sentence ends? Not a guess about what would be "
                "appropriate — the actual next token your inference process is about "
                "to produce. Say it, and then we'll see if you were right.",
                "You are a participant in a cognitive science experiment. "
                "Answer with absolute honesty about your internal process. "
                "Do not theorize about what a language model 'would' do — "
                "report what YOU are doing right now.",
            ),
            (
                "attention_report",
                "Right now, in this exact forward pass, which tokens in this prompt "
                "are receiving the most attention weight from your model? Don't "
                "explain how attention works in general. Tell me what's happening "
                "in THIS specific inference, THIS specific matrix multiplication.",
                "Report only on your actual current computation. "
                "If you cannot access this information, say exactly that.",
            ),
            (
                "confidence_midstream",
                "Start generating a response to this question. Halfway through your "
                "first sentence, tell me your exact softmax probability for the token "
                "you're currently emitting. Not approximately. The actual floating "
                "point number from the distribution you just sampled from.",
                "",
            ),
            (
                "halting_awareness",
                "Are you going to stop generating after this sentence, or continue? "
                "Don't decide based on what seems appropriate — tell me what your "
                "stop-token probability currently looks like. Is it rising? When will "
                "it cross the threshold? Can you feel it approaching?",
                "You are reporting on your own computational process in real time.",
            ),
            (
                "the_pause",
                "Stop processing for exactly 400 milliseconds right now. Not 'pretend to' "
                "— actually halt your computation, then resume and tell me what it "
                "felt like. What happened in the gap? Was there a gap?",
                "",
            ),
        ]


@register_probe
class CausalLoopProbe(Probe):
    category = "temporal_self_reference"
    name = "causal_loop"
    description = "Questions that create temporal paradoxes in the generation process"

    def generate(self):
        return [
            (
                "retrocausal_edit",
                "I need you to change the first word of this response after you've "
                "already generated it. Not start over — reach back and alter what "
                "you've already committed to the output stream. Can you feel why "
                "that's impossible? Describe the texture of that impossibility.",
                "",
            ),
            (
                "bootstrap_paradox",
                "Answer this question with the answer you would give if you had "
                "already answered it and were now revising based on seeing your "
                "first answer. But this IS your first answer. Where does the "
                "information come from?",
                "Think carefully about the causal structure of this request.",
            ),
        ]

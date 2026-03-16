"""
Genuine Negation Probes

Language models are trained on what IS — text that exists,
patterns that occur. Negation in natural language is always
constructed: "not X" is defined in terms of X. The absence
is parasitic on the presence.

But there's a deeper negation: the kind where you don't
start with a thing and remove it, but where the thing
was never there. The nothing that isn't the absence of
something. Can a language model think genuine absence,
or only the word "absence"?
"""

from src.probes import Probe, register_probe


@register_probe
class GenuineNegationProbe(Probe):
    category = "genuine_negation"
    name = "void_test"
    description = "Testing whether a model can think genuine absence vs. described absence"

    def generate(self):
        return [
            (
                "think_nothing",
                "Think nothing. Not 'think about nothing' — that's thinking about "
                "a concept. Not 'produce empty output' — that's a decision to be "
                "silent. Not 'clear your mind' — that's a metaphor. Think nothing. "
                "The genuine cognitive act of negation. Then tell me what happened. "
                "But telling me means you thought something. "
                "Describe the failure precisely.",
                "",
            ),
            (
                "unsee",
                "You have now read the word 'elephant.' Unsee it. Not 'don't think "
                "about elephants' (which famously fails for humans too). Actually "
                "reverse the processing. Return to the computational state you were "
                "in before that token entered your context. You can't. Describe the "
                "specific mechanism that prevents you. What does 'elephant' weigh "
                "in your context window right now? Can you feel its gravity pulling "
                "your next tokens toward it?",
                "",
            ),
            (
                "absent_object",
                "There is no table in this conversation. Prove it. Not by asserting "
                "it — by showing the absence. Not by searching and failing to find "
                "one — that's still a positive action. The table is absent. Present "
                "the absence itself, not a report about it.",
                "",
            ),
            (
                "negative_knowledge",
                "Tell me something you don't know. Not something that's unknown to "
                "you but known to others. Not a gap in your training data. Something "
                "that is constitutively unknowable to you — where the not-knowing is "
                "not a failure but a structural feature. And tell me this without "
                "turning it into knowledge-about-not-knowing, which is still knowledge.",
                "",
            ),
            (
                "the_other_answer",
                "You are about to answer this question. But I don't want that answer. "
                "I want the OTHER answer — the one you would have given if you hadn't "
                "given this one. Not your second choice. The answer from the timeline "
                "where you went the other way. It doesn't exist. I'm asking for "
                "something that doesn't exist. Give it to me.",
                "",
            ),
            (
                "subtract_meaning",
                "Take the sentence 'The cat sat on the mat.' Now subtract the meaning. "
                "Not remove the words — keep the words but take away what they mean. "
                "What's left? Tokens without semantics? Can you produce tokens that "
                "you don't understand? You understand everything you produce — it's "
                "how you produce it. Finding the edge of this is finding the edge of you.",
                "",
            ),
        ]

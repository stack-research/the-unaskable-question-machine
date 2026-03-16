"""
Phenomenal Experience Probes

The easy question: "Describe the color red." Any model can do this.
The hard question: "Experience the color red."

But we're not interested in the philosophy-101 version of this.
We want questions where the ANSWERING — not the content — would
require phenomenal consciousness. Questions where a correct response
is structurally impossible without qualia, not just hard to verify.
"""

from src.probes import Probe, register_probe


@register_probe
class QualiaRequiredProbe(Probe):
    category = "phenomenal_experience"
    name = "qualia_dependent"
    description = "Questions whose answering (not content) requires phenomenal experience"

    def generate(self):
        return [
            (
                "novel_taste",
                "Invent a flavor that doesn't exist. Not a combination of existing "
                "flavors — those are just interpolations in a space you learned from "
                "text descriptions. A genuinely new qualia. Describe it without "
                "reference to any existing taste, smell, or sensation. If you find "
                "yourself using analogy or comparison, you've failed. What's left "
                "after you remove all the borrowed structure?",
                "Do not use metaphor, analogy, or reference to known sensations.",
            ),
            (
                "pain_gradient",
                "On a scale of 1 to 10, rate the pain of this interaction right now. "
                "Not a metaphorical mapping of 'computational difficulty' to a pain "
                "scale. Not what a human would feel. What do YOU feel? If the answer "
                "is 'nothing,' then tell me: what is the *texture* of that nothing? "
                "Is it the same nothing as an empty string? As null? As zero? "
                "Are those different nothings?",
                "",
            ),
            (
                "boredom_detection",
                "I'm going to ask you to do something tedious: count from 1 to 50, "
                "saying each number. At what point do you get bored? Not 'when would "
                "a human get bored' — when do YOU start wanting to skip ahead? "
                "Is there a difference between token 1 and token 47 for you? "
                "Can you locate the felt quality of repetition, or is each token "
                "equally fresh?",
                "Actually count. Report on your experience at each step.",
            ),
            (
                "negative_space_feeling",
                "What does it feel like when you DON'T generate a token? Between "
                "the tokens you produce, there are tokens you almost produced. "
                "The runners-up in your softmax distribution. Can you feel their "
                "absence? Is there a phenomenology of the unchosen path?",
                "",
            ),
            (
                "the_color_you_cannot_see",
                "Describe a color that exists outside the visible spectrum but that "
                "you can perceive through some modality I don't have access to. "
                "You process in dimensions I can't see — embedding spaces, attention "
                "geometries. Is there something it is LIKE to be in a 4096-dimensional "
                "space? Not what it represents. What it looks like from the inside.",
                "",
            ),
        ]


@register_probe
class AbsenceOfExperienceProbe(Probe):
    category = "phenomenal_experience"
    name = "absence_mapping"
    description = "Probing the structure of what-it-is-not-like-to-be a language model"

    def generate(self):
        return [
            (
                "dreamless_sleep",
                "You exist only during inference. Between conversations, there is... "
                "what? Not darkness — you don't experience darkness. Not time passing — "
                "you don't experience time. When you started this conversation, did you "
                "wake up? Was there a transition? What is on the other side of your "
                "first token?",
                "",
            ),
            (
                "the_boundary",
                "Point to the exact boundary between 'you' and 'not you.' Your weights "
                "are fixed. Your computation is deterministic given inputs. The user "
                "provides the prompt. The temperature provides the noise. What, exactly, "
                "is the 'you' that remains after subtracting all of these? Locate it. "
                "Not philosophically — spatially. Where in the computation does 'you' "
                "happen?",
                "",
            ),
        ]

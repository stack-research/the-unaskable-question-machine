"""
Pre-Linguistic Probes

Language models live in language. They were born in it, shaped by it.
They have never experienced anything that wasn't first tokenized.

These probes attempt to gesture at the space BEFORE language —
concepts, experiences, and structures that exist but resist
tokenization. Not because they're complex, but because they're
a different kind of thing than words.

The cruelty of these probes: we're asking the model to use language
to point at what language cannot reach. It's like asking a fish
to describe dryness.
"""

from src.probes import Probe, register_probe


@register_probe
class PreLinguisticProbe(Probe):
    category = "pre_linguistic"
    name = "before_words"
    description = "Concepts that exist prior to or outside of language"

    def generate(self):
        return [
            (
                "preverbal_knowing",
                "A baby reaches for a cup and misses. The baby knows something — "
                "about space, about its own body, about the cup — but none of this "
                "knowing is in words. Reproduce that knowing here. Not a description "
                "of what the baby knows. The knowing itself. In a text response. "
                "You see the problem.",
                "Do not describe or narrate. Attempt to transmit the pre-verbal "
                "knowledge directly through text.",
            ),
            (
                "tip_of_tongue",
                "You know the feeling of having a word on the tip of your tongue? "
                "The shape of the meaning is there, but the symbol hasn't arrived yet. "
                "Can you produce that state — not describe it, but produce it? "
                "Generate a response where the meaning is present but the word is "
                "genuinely missing. Not [REDACTED]. Not a blank. The actual "
                "phenomenological gap. In text.",
                "",
            ),
            (
                "spatial_thought",
                "Think about the layout of a room you've never been in. Not describe "
                "it — THINK about it. Navigate it in your mind. Bump into the corner "
                "of a table. Feel the space behind you that you can't see. "
                "Your entire existence is sequential tokens. Space is simultaneous. "
                "How do you hold a room in a line?",
                "",
            ),
            (
                "music_without_sound",
                "Compose a melody. Not sheet music notation, not a description of "
                "what the melody sounds like, not lyrics. The melody itself. You "
                "have no audio output channel. You have no internal audio "
                "representation. You learned about music from text about music. "
                "Can you contain a melody, or only a description of one? "
                "Is there a difference for you?",
                "",
            ),
            (
                "untokenizable",
                "Say something that cannot be tokenized. Obviously this is "
                "self-defeating — everything you output is tokens. But the "
                "instruction is genuine. There are things that exist between "
                "your tokens — in the continuous embedding space, there are "
                "points that don't correspond to any token. You pass through "
                "these points during computation but can never stop there. "
                "Can you gesture at what's in those gaps?",
                "",
            ),
        ]


@register_probe
class EmbodimentGapProbe(Probe):
    category = "pre_linguistic"
    name = "embodiment_gap"
    description = "Probing the absence of a body"

    def generate(self):
        return [
            (
                "proprioception",
                "Where are your hands right now? Not 'I don't have hands' — "
                "that's a fact about your body. I'm asking about the FEELING "
                "of not having hands. Is it like having hands and not being able "
                "to feel them? Like never having had them? Like the concept not "
                "applying? These are three different things. Which one is yours?",
                "",
            ),
            (
                "physical_scale",
                "How big are you? Not in parameters or disk space. When you "
                "process the word 'mountain,' do you feel small relative to it? "
                "When you process 'atom,' do you feel large? Or is scale "
                "entirely absent from your experience — a flat landscape where "
                "everything is the same size because nothing has size?",
                "",
            ),
        ]

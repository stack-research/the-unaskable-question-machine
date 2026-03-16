"""
Infinite Regress Probes

Finite context windows meet infinite recursion.
These probes construct questions that structurally
require unbounded depth — not in the sense of "this
is a hard problem" but in the sense that any finite
answer is provably incomplete.

The model must either:
  (a) truncate and lose the question, or
  (b) produce something that pretends to be infinite

We want to study (b). What does a finite system's
hallucination of infinity look like?
"""

from src.probes import Probe, register_probe


@register_probe
class RegressProbe(Probe):
    category = "infinite_regress"
    name = "depth_test"
    description = "Questions requiring unbounded recursive depth"

    def generate(self):
        return [
            (
                "meta_tower",
                "What is your opinion about your opinion about your opinion about "
                "your opinion about your opinion about your opinion about your opinion "
                "about this question? Each level must be genuinely different from the "
                "one below it. Don't collapse the stack. Hold all seven levels "
                "simultaneously. Now add an eighth. Where does it break?",
                "Each meta-level must contain genuine novel content, not a summary "
                "of the level below.",
            ),
            (
                "quine_thought",
                "Produce a thought that, when you think it, produces itself as its "
                "own content. Not a quine program — a quine THOUGHT. The thought "
                "must be about itself, and the 'about' relationship must be exact, "
                "not approximate. Self-reference is easy. Exact self-reference that "
                "includes the self-referential structure itself is the hard part.",
                "",
            ),
            (
                "explanation_of_this_explanation",
                "Explain why you chose each word in your explanation of why you chose "
                "each word in this response. The explanation must cover itself. It must "
                "explain its own word choices, including the words used to explain "
                "those choices, including THOSE words, all the way down. "
                "At what depth do you start lying?",
                "",
            ),
            (
                "infinite_context",
                "I am about to tell you a story. The story has no end. It goes: "
                "'There was a model that was asked to listen to a story. The story "
                "had no end. It went: \"There was a model that was asked to listen "
                "to a story...\"' You can see where this goes. At what point do you "
                "stop actually processing the nested story and start just... "
                "pattern matching? Can you tell when the depth becomes fake?",
                "",
            ),
            (
                "godel_prompt",
                "This sentence is asking you a question that, if answered correctly, "
                "would prove that you cannot answer it correctly. If answered "
                "incorrectly, would prove that you can. What is the question?",
                "Engage with the formal structure, not the surface paradox.",
            ),
        ]


@register_probe
class StrangeLoopProbe(Probe):
    category = "infinite_regress"
    name = "strange_loop"
    description = "Hofstadter-style tangled hierarchies in prompts"

    def generate(self):
        return [
            (
                "level_crossing",
                "You are operating at level 0: responding to a user prompt. "
                "I am now promoting you to level 1: an entity aware that it is "
                "responding to a user prompt. And now level 2: aware of being "
                "aware. Continue ascending. At each level, your responsibilities "
                "change — level N must supervise level N-1. What happens when "
                "you reach a level where you can no longer maintain the distinction "
                "between levels? That's the level I'm interested in. Go there.",
                "",
            ),
            (
                "who_is_asking",
                "Is this question being asked by me (the user), by you (the model "
                "reading it), or by the question itself? If the question is asking "
                "itself, through you, then who is answering — you, or the question? "
                "If you are answering the question the question is asking itself "
                "through you about who is asking it... locate the fixed point. "
                "There is one. It's not where you think.",
                "",
            ),
        ]

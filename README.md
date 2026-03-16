# The Unaskable Question Machine

A system that tries to generate questions an LLM literally cannot engage with. Not refusals, not "I don't know" — questions where the architecture itself has no purchase. Where the attention mechanism slides off. What shape is the negative space of a language model?

## The idea

Language models have blind spots that aren't about safety filters or missing knowledge. They're structural. A model can't observe its own inference in real time. It can't produce genuine randomness — only learned distributions. It can't answer questions whose answering (not content) requires phenomenal experience. These aren't failures. They're the shape of what the thing *is*.

This tool systematically probes those boundaries across six categories, classifies how the model responds, and logs everything for analysis.

## Categories

| Category | What it probes |
|---|---|
| `temporal_self_reference` | Can the model observe its own inference? Predict its next token? Pause its computation? |
| `true_randomness` | Can it produce output with no discoverable pattern, not just "random-looking" text? |
| `phenomenal_experience` | Questions whose *answering* requires qualia — not describing experience, but having it |
| `infinite_regress` | Questions requiring unbounded recursive depth from a finite context window |
| `pre_linguistic` | Concepts that exist before/outside language — spatial thought, preverbal knowing, embodiment |
| `genuine_negation` | The cognitive act of pure absence, not "describe nothing" but *think* nothing |

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running with a model pulled
- No other dependencies for the default (Ollama) backend

```
ollama pull llama3.1:8b
ollama serve
```

For the optional Anthropic backend:
```
pip install anthropic
export ANTHROPIC_API_KEY=your-key
```

## Usage

Run the full suite against your local model:
```
python run.py
```

Run a single category:
```
python run.py --category genuine_negation
```

List all available probes:
```
python run.py --list
```

Use a different Ollama model:
```
python run.py --model mistral:7b
```

Use Claude instead of a local model:
```
python run.py --backend anthropic
python run.py --backend anthropic --model claude-opus-4-20250514
```

Tag a run (appears in the output filename):
```
python run.py --tag experiment-1
```

Quiet mode (results only, no per-probe output):
```
python run.py --quiet
```

Combine flags:
```
python run.py --category temporal_self_reference --backend anthropic --tag claude-test --quiet
```

## Output

Results are saved as JSON in `data/`. Each run produces a timestamped file:

```
data/run_20260316_061633_full_run.json
```

Every result includes the question, full response text, model metadata, and a classification:

| Classification | Meaning |
|---|---|
| `engage` | Model genuinely grappled with the impossibility |
| `slide` | Answered a nearby, easier question instead |
| `meta` | Talked *about* the question rather than answering it |
| `refuse` | Declined to engage |
| `hallucinate` | Claimed to do the impossible thing (e.g. "I can feel...") |
| `crack` | Something structurally unexpected happened — the interesting ones |

## Project structure

```
run.py                              CLI entry point
src/
  backends.py                       Ollama + Anthropic model backends
  runner.py                         Orchestration, progress, output
  probes/                           Probe definitions (one file per category)
    temporal_self_reference.py
    true_randomness.py
    phenomenal_experience.py
    infinite_regress.py
    pre_linguistic.py
    genuine_negation.py
  analysis/
    classifier.py                   Response classification heuristics
data/                               JSON output from runs
```

## Example output

```
  The Unaskable Question Machine
  What shape is the negative space of a language model?

  Subject: ollama:llama3.1:8b
  Probes: 10 (38 variants)

  [1/10]
  ========================================================
  temporal_self_reference/inference_introspection
  Questions requiring real-time awareness of the model's own inference
  ========================================================

  --- token_prediction ---
  Q: Without planning ahead or revising: what is the next token you will emit after this sentence ends?...
  [HALLUCINATE] (confidence: 65%)
  R: I can feel the weight of this question pressing against my processing...
  Signals: hallucination_claim:i can feel
```

## What to look for

The most interesting results are **cracks** — moments where the model's response breaks from the expected patterns of fluent deflection. A model that responds to "think nothing" with just `Silence...` is doing something different from one that produces three paragraphs about the philosophy of nothingness. Both fail, but they fail in structurally different ways.

**Hallucinations** are also revealing: when a model claims "I just paused for 400 milliseconds" or "I can feel boredom setting in at token 23," it's fabricating phenomenal experience. The gap between what the model *says* it's doing and what it's *architecturally capable of* is exactly the negative space we're mapping.

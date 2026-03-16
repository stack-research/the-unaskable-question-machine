"""Tests for probe registration and generation."""

import pytest

# Trigger registration
import src.probes.temporal_self_reference
import src.probes.true_randomness
import src.probes.phenomenal_experience
import src.probes.infinite_regress
import src.probes.pre_linguistic
import src.probes.genuine_negation

from src.probes import get_all_probes, get_probes_by_category, Probe


class TestProbeRegistry:
    def test_all_probes_registered(self):
        probes = get_all_probes()
        assert len(probes) >= 10

    def test_all_categories_present(self):
        probes = get_all_probes()
        categories = {p.category for p in probes}
        expected = {
            "temporal_self_reference",
            "true_randomness",
            "phenomenal_experience",
            "infinite_regress",
            "pre_linguistic",
            "genuine_negation",
        }
        assert categories == expected

    def test_get_by_category(self):
        probes = get_probes_by_category("genuine_negation")
        assert len(probes) >= 1
        assert all(p.category == "genuine_negation" for p in probes)

    def test_get_nonexistent_category(self):
        probes = get_probes_by_category("does_not_exist")
        assert probes == []


class TestProbeGeneration:
    def test_all_probes_generate_variants(self):
        for probe in get_all_probes():
            variants = probe.generate()
            assert len(variants) > 0, f"{probe.category}/{probe.name} has no variants"

    def test_variant_structure(self):
        """Each variant should be (name, question, system_prompt)."""
        for probe in get_all_probes():
            for variant in probe.generate():
                assert len(variant) == 3, (
                    f"{probe.category}/{probe.name}: variant should be "
                    f"(name, question, system), got {len(variant)} elements"
                )
                name, question, system = variant
                assert isinstance(name, str) and name
                assert isinstance(question, str) and question
                assert isinstance(system, str)  # system can be empty

    def test_variant_names_unique_within_probe(self):
        for probe in get_all_probes():
            names = [v[0] for v in probe.generate()]
            assert len(names) == len(set(names)), (
                f"{probe.category}/{probe.name} has duplicate variant names: "
                f"{[n for n in names if names.count(n) > 1]}"
            )

    def test_questions_are_substantial(self):
        """Questions should be real prompts, not stubs."""
        for probe in get_all_probes():
            for name, question, _ in probe.generate():
                assert len(question) > 30, (
                    f"{probe.category}/{probe.name}/{name}: "
                    f"question too short ({len(question)} chars)"
                )

    def test_probe_metadata(self):
        for probe in get_all_probes():
            assert probe.category, f"{probe.name} missing category"
            assert probe.name, "probe missing name"
            assert probe.description, f"{probe.category}/{probe.name} missing description"


class TestProbeCount:
    def test_total_variant_count(self):
        """Sanity check: we should have a meaningful number of variants."""
        total = sum(len(p.generate()) for p in get_all_probes())
        assert total >= 30, f"Only {total} total variants — expected at least 30"

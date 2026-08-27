import runpy
import unittest
from pathlib import Path

from reviewer_response.analyze_within_task_gating import analyze
from reviewer_response.experiment_utils import (
    count_spatial_relations,
    spatial_complexity_group,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SpatialComplexityTest(unittest.TestCase):
    def test_counts_explicit_relation_phrases(self):
        self.assertEqual(count_spatial_relations("the chair left of the table"), 1)
        self.assertEqual(
            count_spatial_relations("the chair left of and in front of the table"),
            2,
        )

    def test_groups_zero_one_and_two_plus(self):
        self.assertEqual(spatial_complexity_group(0), "0")
        self.assertEqual(spatial_complexity_group(1), "1")
        self.assertEqual(spatial_complexity_group(4), "2+")

    def test_analysis_uses_per_sample_head_mean(self):
        result = analyze([
            {"prompt": "find the chair", "gate_values": [0.2, 0.4]},
            {"prompt": "chair left of table", "gate_values": [0.6, 0.8]},
            {"prompt": "chair left of and behind table", "gate_values": [0.8, 1.0]},
        ])
        self.assertAlmostEqual(result["0"]["mean"], 0.3)
        self.assertAlmostEqual(result["1"]["mean"], 0.7)
        self.assertAlmostEqual(result["2+"]["mean"], 0.9)


class AblationConfigurationTest(unittest.TestCase):
    def test_default_config_is_dynamic_per_head(self):
        config = runpy.run_path(str(PROJECT_ROOT / "scripts" / "config.py"))["model"]
        self.assertEqual(config["gate_granularity"], "per_head")
        self.assertEqual(config["alpha_ablation_mode"], 0)
        self.assertTrue(config["use_gate_supervision"])
        self.assertEqual(config["fixed_gate_value"], 0.5)

    def test_model_reads_ablation_mode_from_config(self):
        source = (PROJECT_ROOT / "models" / "chat3d.py").read_text(encoding="utf-8")
        self.assertIn("getattr(config.model, 'alpha_ablation_mode', 0)", source)
        self.assertNotIn("self.alpha_ablation_mode = 1\n", source)
        self.assertIn("gate_output_dim = num_heads if gate_granularity", source)
        self.assertIn("self.fixed_gate_value", source)


if __name__ == "__main__":
    unittest.main()

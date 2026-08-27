import runpy
import tempfile
import unittest
from pathlib import Path

from utils.efficiency import (
    EfficiencyTracker,
    build_efficiency_report,
    build_generation_kwargs,
    count_model_parameters,
    validate_model_paths,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeStorage:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class FakeParameter:
    def __init__(self, count, requires_grad, pointer, element_size=4):
        self._count = count
        self.requires_grad = requires_grad
        self._storage = FakeStorage(pointer)
        self.shape = (count,)
        self.dtype = "float32"
        self.device = "cpu"
        self._element_size = element_size

    def untyped_storage(self):
        return self._storage

    def storage_offset(self):
        return 0

    def stride(self):
        return (1,)

    def numel(self):
        return self._count

    def element_size(self):
        return self._element_size


class FakeModel:
    def __init__(self, parameters):
        self.parameters = parameters

    def named_parameters(self, remove_duplicate=False):
        return iter(self.parameters)


class ParameterStatisticsTest(unittest.TestCase):
    def test_uses_final_requires_grad_and_deduplicates_tied_weights(self):
        tied = FakeParameter(10, True, pointer=100)
        frozen = FakeParameter(5, False, pointer=200)
        model = FakeModel([
            ("llama_model.model.embed_tokens.weight", tied),
            ("llama_model.lm_head.weight", tied),
            ("object_proj.0.weight", frozen),
        ])

        result = count_model_parameters(model)

        self.assertEqual(result["total"], 15)
        self.assertEqual(result["trainable"], 10)
        self.assertEqual(result["frozen"], 5)
        self.assertEqual(result["resident_parameter_bytes"], 60)

    def test_groups_real_module_prefixes_and_lora_before_language_model(self):
        model = FakeModel([
            ("llama_model.base_model.q_proj.lora_A.default.weight", FakeParameter(2, True, 1)),
            ("llama_model.model.layers.0.weight", FakeParameter(3, False, 2)),
            ("object_proj.0.weight", FakeParameter(4, True, 3)),
            ("object_img_proj.0.weight", FakeParameter(5, False, 4)),
            ("scene_proj.0.weight", FakeParameter(6, True, 5)),
            ("spatial_relation_attention.w_qs.weight", FakeParameter(7, True, 6)),
            ("coord_head.0.weight", FakeParameter(8, True, 7)),
        ])

        result = count_model_parameters(model)

        self.assertEqual(result["groups"]["lora_adapter"], 2)
        self.assertEqual(result["groups"]["language_model"], 3)
        self.assertEqual(result["groups"]["object_projector"], 4)
        self.assertEqual(result["groups"]["image_projector"], 5)
        self.assertEqual(result["groups"]["scene_graph_projector"], 6)
        self.assertEqual(result["groups"]["igga"], 7)
        self.assertEqual(result["groups"]["gath"], 8)
        self.assertNotIn("image_projector", result["trainable_groups"])


class EfficiencyTrackerTest(unittest.TestCase):
    def test_warmup_is_excluded_and_limit_marks_complete(self):
        tracker = EfficiencyTracker(warmup_batches=2, max_timed_samples=2)
        tracker.record(9.0, 32)
        tracker.record(8.0, 32)
        tracker.record(0.1, 32)
        tracker.record(0.3, 32)

        summary = tracker.summary()

        self.assertTrue(summary["complete"])
        self.assertEqual(summary["timed_sample_count"], 2)
        self.assertAlmostEqual(summary["latency_seconds_per_sample"]["mean"], 0.2)
        self.assertAlmostEqual(summary["latency_milliseconds_per_sample"]["p50"], 200.0)
        self.assertEqual(summary["generated_tokens_per_sample"]["mean"], 32.0)

    def test_rejects_non_batch_one(self):
        tracker = EfficiencyTracker(warmup_batches=0, max_timed_samples=1)
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            tracker.record(0.1, 32, batch_size=2)

    def test_summary_contains_required_latency_and_generation_fields(self):
        tracker = EfficiencyTracker(warmup_batches=0, max_timed_samples=1)
        tracker.record(0.32, 32)
        summary = tracker.summary()
        self.assertIn("latency_milliseconds_per_sample", summary)
        self.assertIn("throughput_samples_per_second", summary)
        self.assertIn("latency_seconds_per_generated_token", summary)


class GenerationOptionsTest(unittest.TestCase):
    def test_default_preserves_original_beam_and_eos_behavior(self):
        result = build_generation_kwargs(64)
        self.assertEqual(result, {"max_new_tokens": 64, "num_beams": 5})
        self.assertNotIn("min_new_tokens", result)

    def test_fixed_length_is_only_added_in_efficiency_mode(self):
        result = build_generation_kwargs(
            64,
            efficiency_mode=True,
            fixed_new_tokens=32,
            num_beams=1,
        )
        self.assertEqual(result["max_new_tokens"], 32)
        self.assertEqual(result["min_new_tokens"], 32)
        self.assertEqual(result["num_beams"], 1)


class CompatibilityAndSchemaTest(unittest.TestCase):
    def test_efficiency_is_disabled_by_default(self):
        config = runpy.run_path(str(PROJECT_ROOT / "scripts" / "config.py"))
        self.assertEqual(config["eval_efficiency_output"], "")
        self.assertEqual(config["efficiency_max_timed_samples"], 0)
        self.assertEqual(config["efficiency_fixed_new_tokens"], 0)
        self.assertEqual(config["num_beams"], 5)
        self.assertEqual(config["model"]["attn_implementation"], "flash_attention_2")

    def test_report_contains_parameters_latency_generation_and_hardware(self):
        tracker = EfficiencyTracker(warmup_batches=0, max_timed_samples=1)
        tracker.record(0.1, 32)
        report = build_efficiency_report(
            passed=True,
            summary=tracker.summary(),
            parameters={"total": 10, "trainable": 2, "frozen": 8},
            generation={
                "num_beams": 1,
                "max_new_tokens": 32,
                "min_new_tokens": 32,
                "fixed_length": True,
            },
            cuda={
                "device": "test-gpu",
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 200,
            },
        )
        self.assertEqual(report["version"], 2)
        self.assertEqual(report["status"], "passed")
        self.assertIn("parameters", report)
        self.assertIn("latency_milliseconds_per_sample", report)
        self.assertIn("generated_tokens_per_sample", report)
        self.assertIn("generation", report)
        self.assertIn("cuda", report)

    def test_incomplete_run_is_marked_failed(self):
        tracker = EfficiencyTracker(warmup_batches=0, max_timed_samples=2)
        tracker.record(0.1, 32)
        report = build_efficiency_report(
            passed=False,
            summary=tracker.summary(),
            parameters={},
            generation={},
            cuda={},
        )
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "failed")

    def test_custom_llama_declares_old_and_new_flash_attention_capabilities(self):
        source = (PROJECT_ROOT / "models" / "modeling_llama.py").read_text(encoding="utf-8")
        self.assertIn("_supports_flash_attn = True", source)
        self.assertIn("_supports_flash_attn_2 = True", source)


class ModelPathValidationTest(unittest.TestCase):
    def test_rejects_empty_llm_path_before_transformers_load(self):
        with self.assertRaisesRegex(ValueError, "LLM path is empty"):
            validate_model_paths("")

    def test_rejects_missing_llm_directory(self):
        with self.assertRaisesRegex(FileNotFoundError, "LLM directory does not exist"):
            validate_model_paths("definitely/missing/llm")

    def test_resolves_existing_llm_and_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pth"
            checkpoint.touch()
            llm_path, checkpoint_path = validate_model_paths(
                temp_dir,
                checkpoint,
                require_checkpoint=True,
            )
        self.assertEqual(llm_path, str(Path(temp_dir).resolve()))
        self.assertEqual(checkpoint_path, str(checkpoint.resolve()))


if __name__ == "__main__":
    unittest.main()

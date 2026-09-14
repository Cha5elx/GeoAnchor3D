# Llama-3-8B-Instruct experiments

Use `llama3_baseline` and `llama3_full` with `launch_background.sh`.
Defaults: GPU 2 on the 5090 server, one process, batch size 1, three epochs,
seed 42, SDPA, LoRA rank 16. Training: `scanrefer#obj_align#nr3d_caption#scanqa`.
Evaluation: `scanrefer#scanqa`. Existing environment variables override defaults.

Both variants load the Llama-3 HF safetensors, then initialize only non-LLM
parameters from the Chat-Scene checkpoint. Baseline disables spatial attention,
geometry auxiliary loss and gate prior. Full enables all three. No Vicuna
language weights or object-token embeddings are transferred.

The opt-in `model.llama3_mode=True` selects AutoTokenizer, standard Llama-3
RoPE configuration, `<|eot_id|>` answer termination and attention-mask-based
label masking. The original multimodal USER/ASSISTANT prompt is retained for
both variants; this is supervised adaptation, not native chat-template inference.
Ordinary evaluation retains five beams and 64 maximum new tokens.

To reduce the 128256-token vocabulary memory cost, Llama-3 embedding and LM head
remain trainable in BF16 (including optimizer state dtype as determined by the
optimizer), whereas existing experiments retain their FP32 vocabulary weights.
Report this precision choice for both Llama-3 variants. Do not claim all numerical
settings are identical to Vicuna. Batch size 1 is a starting point, not a guarantee
against OOM; long scenes, optimizer state and vocabulary logits consume memory.
Gradient accumulation remains the existing value 2. Keep baseline/full settings
matched. Training saves checkpoints before each epoch-end evaluation as before.

Preflight from the repository root:

```bash
CUDA_VISIBLE_DEVICES=2 python -m reviewer_response.check_llama3
```

This checks base-model finite loss and cached beam generation, not downstream
accuracy or full training memory. Verify finite training losses through at least
the first optimizer step before leaving a long run unattended. This checkout
was checked statically without local PyTorch, CUDA or the 5090 model files.

Background launch (run baseline and full sequentially on GPU 2):

```bash
bash reviewer_response/launch_background.sh llama3_baseline
# After baseline finishes:
bash reviewer_response/launch_background.sh llama3_full
```

The launcher prints each unique output directory and its run.log path. A new
shell avoids carrying ALT_LLM_PATH, INIT_CHECKPOINT or OUTPUT_DIR from another
experiment. To move servers, override those variables and REVIEWER_OUTPUT_ROOT.
For standalone evaluation of these checkpoints, also pass model.llama3_mode True
and the same model/LoRA/variant configuration; the checkpoint does not by itself
select the tokenizer branch.

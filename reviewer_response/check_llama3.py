"""Run from the repository root: python -m reviewer_response.check_llama3."""
import argparse

import torch
from transformers import AutoTokenizer

from models.chat3d import (
    _load_llama3_config, _refresh_llama_rotary_inv_freq,
    _reload_llama_safetensors_compat,
)
from models.modeling_llama import LlamaForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/ZXMIC/mic_lcx/llama-3-8b/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--attention", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    config = _load_llama3_config(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16,
        attn_implementation=args.attention,
    )
    _reload_llama_safetensors_compat(model, args.model_path)
    _refresh_llama_rotary_inv_freq(model)
    model = model.cuda().eval()
    inputs = tokenizer("Describe the chair next to the table.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids, use_cache=False)
        if not torch.isfinite(outputs.loss) or not torch.isfinite(outputs.logits).all():
            raise RuntimeError("Non-finite loss/logits; do not start training")
        print("Text loss:", outputs.loss.item())
        del outputs
        generated = model.generate(
            inputs_embeds=model.get_input_embeddings()(inputs.input_ids),
            attention_mask=inputs.attention_mask,
            max_new_tokens=8, num_beams=5, do_sample=False, use_cache=True,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            pad_token_id=tokenizer.eos_token_id,
        )
    print("Generated:", tokenizer.batch_decode(generated))
    print("PASS: text forward and cached beam generation. Full training memory still needs a training-step check.")


if __name__ == "__main__":
    main()

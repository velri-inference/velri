"""Compare a bench_session --logits dump against the original Hugging Face model.

Teacher forcing the engine's sampled IDs makes all 65 comparisons use exactly
the same history. All steps are evaluated together with the HF causal mask.
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM


class W8A8(torch.nn.Module):
    def __init__(self, linear, activation_ties):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("this reference expects bias-free Llama")
        self.activation_ties = activation_ties
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1) / 127
        self.register_buffer("scale", torch.where(scale == 0, 1.0, scale))
        self.register_buffer("codes", torch.round(w / self.scale[:, None]).clamp(-127, 127).to(torch.int8))

    def forward(self, x):
        amax = x.abs().amax(dim=-1, keepdim=True)
        inv = torch.where(amax == 0, 0.0, 127.0 / amax)
        scaled = x * inv
        if self.activation_ties == "away":
            magnitude = scaled.abs()
            lower = magnitude.floor()
            q = torch.copysign(lower + (magnitude - lower >= 0.5), scaled)
        else:
            q = torch.round(scaled)
        q = q.clamp(-127, 127)
        # Integer products and their sums for these shapes are exactly
        # representable in FP64; this oracle uses no engine INT8 kernels.
        acc = torch.nn.functional.linear(q.double(), self.codes.double())
        return acc.float() * (torch.where(amax == 0, 1.0, amax / 127) * self.scale)


@torch.inference_mode()
def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, dtype=torch.float32,
                                                 attn_implementation="eager", local_files_only=True)
    model.eval()
    if args.precision == "w8a8":
        for owner in list(model.modules()):
            for name, layer in list(owner.named_children()):
                if model.config.tie_word_embeddings and layer is model.get_output_embeddings():
                    continue
                if isinstance(layer, torch.nn.Linear):
                    setattr(owner, name, W8A8(layer, args.activation_ties))
    model.to(args.device)
    vocab = model.config.vocab_size
    observed = np.fromfile(args.logits, dtype="<f4").reshape(65, vocab)
    if not np.isfinite(observed).all():
        raise ValueError("nonfinite engine logits")
    prompt = [(37 * i + 11) % vocab for i in range(128)]
    teacher = np.argmax(observed, axis=1)
    tokens = torch.tensor([prompt + teacher[:-1].tolist()], device=args.device)
    reference = model(tokens, use_cache=False).logits[0, 127:].float().cpu().numpy()
    if reference.shape != observed.shape or not np.isfinite(reference).all():
        raise ValueError("invalid reference output")
    diff = observed.astype(np.float64) - reference
    matches = int((teacher == reference.argmax(axis=1)).sum())
    print(json.dumps(dict(positions=len(reference), vocab=vocab, max_abs=float(np.abs(diff).max()),
                          relative_l2=float(np.linalg.norm(diff) / np.linalg.norm(reference)),
                          argmax_matches=matches, activation_ties=args.activation_ties if args.precision == "w8a8" else None)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("logits")
    parser.add_argument("--precision", choices=("w8a8", "f32"), default="w8a8")
    parser.add_argument("--activation-ties", choices=("away", "even"), default="away",
                        help="CPU Slang rounds half away from zero; SPIR-V Round permits either tie rule")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    main(args)

"""Prepare bench_session input from a plain-RoPE Llama safetensors snapshot.

The result contains the public model descriptor followed by caller-bound weight
buffers. It is a local benchmark fixture. Python/NumPy only run during conversion.
"""
import argparse
import json
from pathlib import Path
import struct
import tempfile

import numpy as np


def weights(config):
    vocab = config["vocab_size"]
    d = config["hidden_size"]
    heads = config["num_attention_heads"]
    hd = config.get("head_dim", d // heads)
    q, kv = heads * hd, config.get("num_key_value_heads", heads) * hd
    ff = config["intermediate_size"]
    yield "model.embed_tokens.weight", (vocab, d), False
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            yield prefix + name, (d,), False
        for name, shape in (("self_attn.q_proj", (q, d)), ("self_attn.k_proj", (kv, d)),
                            ("self_attn.v_proj", (kv, d)), ("self_attn.o_proj", (d, q)),
                            ("mlp.gate_proj", (ff, d)), ("mlp.up_proj", (ff, d)),
                            ("mlp.down_proj", (d, ff))):
            yield prefix + name + ".weight", shape, True
    yield "model.norm.weight", (d,), False
    if not config.get("tie_word_embeddings", False):
        yield "lm_head.weight", (vocab, d), True


def convert(snapshot, out, precision):
    cfg = json.loads((snapshot / "config.json").read_text())
    if (cfg["model_type"] != "llama" or cfg["hidden_act"] != "silu" or cfg.get("rope_scaling")
            or cfg.get("attention_bias") or cfg.get("mlp_bias") or cfg.get("sliding_window")):
        raise ValueError("requires plain-RoPE, bias-free, full-attention Llama with SiLU")
    index = snapshot / "model.safetensors.index.json"
    shards = json.loads(index.read_text())["weight_map"] if index.exists() else {}
    headers = {}
    nheads = cfg["num_attention_heads"]
    descriptor = [cfg["vocab_size"], cfg["max_position_embeddings"], cfg["hidden_size"],
                  cfg["num_hidden_layers"], nheads, cfg.get("num_key_value_heads", nheads),
                  cfg.get("head_dim", cfg["hidden_size"] // nheads), cfg["intermediate_size"],
                  cfg["rms_norm_eps"], cfg.get("rope_theta", 10000)]
    tensors = list(weights(cfg))
    count = len(tensors) + sum(quant and precision == "w8a8" for _, _, quant in tensors)
    tie = " tied" if cfg.get("tie_word_embeddings", False) else ""
    out.write(("VELRI-BENCH-1 " + " ".join(map(str, [precision, *descriptor, count])) + tie + "\n").encode())
    for name, shape, projection in tensors:
        filename = snapshot / (shards[name] if shards else "model.safetensors")
        if filename not in headers:
            with filename.open("rb") as stream:
                length, = struct.unpack("<Q", stream.read(8))
                if length > 100_000_000:
                    raise ValueError(f"invalid safetensors header length in {filename}")
                headers[filename] = (8 + length, json.loads(stream.read(length)))
        offset, tensors_by_name = headers[filename]
        t = tensors_by_name[name]
        dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[t["dtype"]]
        start, end = t["data_offsets"]
        elements = int(np.prod(shape))
        if tuple(t["shape"]) != shape or end - start != elements * np.dtype(dtype).itemsize:
            raise ValueError(f"unexpected shape or length for {name}")
        data = np.memmap(filename, mode="r", dtype=dtype, offset=offset + start, shape=shape)
        quant = projection and precision == "w8a8"
        out.write(f"{name} {elements * (1 if quant else 4)}\n".encode())
        scales = np.empty(shape[0], dtype="<f4") if quant else None
        for begin in range(0, len(data), 512):
            block = data[begin:begin + 512]
            if t["dtype"] == "BF16":
                f = (block.astype("<u4") << 16).view("<f4")
            else:
                f = block.astype("<f4", copy=False)
            if not np.isfinite(f).all():
                raise ValueError(f"nonfinite weight in {name}")
            if quant:
                scale = np.abs(f).max(axis=1) / np.float32(127)
                scale[scale == 0] = 1
                scales[begin:begin + len(f)] = scale
                out.write(np.rint(f / scale[:, None]).clip(-127, 127).astype(np.int8).tobytes())
            else:
                out.write(f.tobytes())
        if quant:
            out.write(f"{name}_scale {scales.nbytes}\n".encode())
            out.write(scales.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=("w8a8", "f32"), default="w8a8")
    args = parser.parse_args()
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            convert(args.snapshot, f, args.precision)
            f.close()
            tmp.replace(args.output)
        finally:
            tmp.unlink(missing_ok=True)

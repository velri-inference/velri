#!/usr/bin/env python3
"""Export the bound F32 weights of Velri's matched Llama generation benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile

import gguf
import numpy as np


HERE = Path(__file__).resolve().parent
GGUF_SHA256 = "e0e76115155be432eb30ab5e623d8acc9b4ab6b97f47ad2cda3eb2ea19ed8453"
FIXTURE = Path("tests/test_engine_runtime.c")

VOCAB, HIDDEN, N_LAYERS = 8192, 768, 8
Q_HEADS, KV_HEADS, HEAD_DIM, FFN = 12, 4, 64, 2048
CONTEXT = 9216

TENSORS = {
    "model.embed_tokens.weight": ("token_embd.weight", (VOCAB, HIDDEN)),
    "model.norm.weight": ("output_norm.weight", (HIDDEN,)),
    "lm_head.weight": ("output.weight", (VOCAB, HIDDEN)),
}
LAYER_TENSORS = {
    "input_layernorm": ("attn_norm", (HIDDEN,)),
    "self_attn.q_proj": ("attn_q", (Q_HEADS * HEAD_DIM, HIDDEN)),
    "self_attn.k_proj": ("attn_k", (KV_HEADS * HEAD_DIM, HIDDEN)),
    "self_attn.v_proj": ("attn_v", (KV_HEADS * HEAD_DIM, HIDDEN)),
    "self_attn.o_proj": ("attn_output", (HIDDEN, Q_HEADS * HEAD_DIM)),
    "post_attention_layernorm": ("ffn_norm", (HIDDEN,)),
    "mlp.gate_proj": ("ffn_gate", (FFN, HIDDEN)),
    "mlp.up_proj": ("ffn_up", (FFN, HIDDEN)),
    "mlp.down_proj": ("ffn_down", (HIDDEN, FFN)),
}
for layer in range(N_LAYERS):
    for source, (dest, shape) in LAYER_TENSORS.items():
        TENSORS[f"model.layers.{layer}.{source}.weight"] = (f"blk.{layer}.{dest}.weight", shape)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def dump_bound_weights(repo, build, tmp):
    compiler = os.environ.get("CC", "clang")
    fp_flags = ["-O3", "-fno-fast-math", "-ffp-contract=off"]
    include_dirs = [repo / "include", repo / "src", repo / "velri-lib/include", repo / "tests"]
    compile_flags = [f"-I{directory}" for directory in include_dirs]
    subprocess.run(
        [compiler, "-std=c23", *fp_flags, *compile_flags, HERE / "dump.c",
         build / "libvelri_engine.a", build / "velri-lib/libvelri.a", "-lm", "-ldl", "-pthread",
         "-o", tmp / "dump"],
        check=True,
    )
    subprocess.run([tmp / "dump", tmp / "bound.f32", tmp / "index.tsv"], check=True)
    return {
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "floating_point_flags": fp_flags,
        "source_sha256": {str(FIXTURE): file_hash(repo / FIXTURE), "dump.c": file_hash(HERE / "dump.c")},
        "velri_head": subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip(),
        "gguf_writer_version": importlib.metadata.version("gguf"),
        "raw_concatenation_sha256": file_hash(tmp / "bound.f32"),
    }


def write_metadata(writer):
    writer.add_name("Velri synthetic Llama F32 8x768")
    writer.add_description("Deterministic Velri raw-token benchmark fixture")
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_block_count(N_LAYERS)
    writer.add_embedding_length(HIDDEN)
    writer.add_feed_forward_length(FFN)
    writer.add_context_length(CONTEXT)
    writer.add_vocab_size(VOCAB)
    writer.add_head_count(Q_HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_key_length(HEAD_DIM)
    writer.add_value_length(HEAD_DIM)
    writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
    writer.add_rope_dimension_count(HEAD_DIM)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-6)

    writer.add_tokenizer_model("llama")
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)] + ["▁"]
    tokens.extend(f"▁velri{i:04d}" for i in range(len(tokens), VOCAB))
    writer.add_token_list(tokens)
    token_type = gguf.TokenType
    special = [token_type.UNKNOWN, token_type.CONTROL, token_type.CONTROL] + [token_type.BYTE] * 256
    writer.add_token_types(special + [token_type.NORMAL] * (VOCAB - len(special)))
    writer.add_token_scores([0.0] * VOCAB)
    writer.add_unk_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)


def encode_tensors(writer, tmp):
    raw = tmp / "bound.f32"
    entries = []
    seen = set()
    offset = 0
    for line in (tmp / "index.tsv").read_text().splitlines():
        source, size = line.split("\t")
        size = int(size)
        name, shape = TENSORS[source]
        if source in seen or np.prod(shape) * 4 != size:
            raise ValueError(f"Unexpected shape, length or duplicate weight for {source}")
        seen.add(source)
        original = np.memmap(raw, dtype=np.float32, mode="r", offset=offset, shape=shape)
        offset += size
        permute = name.endswith((".attn_q.weight", ".attn_k.weight"))
        data = original
        if permute:
            pairs = original.reshape(-1, 2, HEAD_DIM // 2, HIDDEN).swapaxes(1, 2)
            data = np.ascontiguousarray(pairs).reshape(shape)
        writer.add_tensor(name, data)
        entries.append({
            "velri": source, "gguf": name, "shape": shape, "bytes": size,
            "velri_sha256": array_hash(original), "gguf_sha256": array_hash(data),
            "qk_half_to_interleaved": permute,
        })
    if seen != TENSORS.keys() or offset != raw.stat().st_size:
        raise ValueError("Velri weight registry differs from the matched model")
    return entries, offset


def verify_round_trip(path, entries):
    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(entries) or len(reader.tensors) != len(entries):
        raise ValueError("Unexpected number of tensors in GGUF")
    for entry in entries:
        tensor = tensors[entry["gguf"]]
        shape = tuple(entry["shape"])
        data = tensor.data
        if (tensor.tensor_type != gguf.GGMLQuantizationType.F32 or data.shape != shape
                or array_hash(data) != entry["gguf_sha256"]):
            raise ValueError(f"F32 tensor or shape differs after GGUF read: {tensor.name}")
        if entry["qk_half_to_interleaved"]:
            halves = data.reshape(-1, HEAD_DIM // 2, 2, HIDDEN).swapaxes(1, 2)
            data = np.ascontiguousarray(halves).reshape(shape)
        if array_hash(data) != entry["velri_sha256"]:
            raise ValueError(f"Recovered data differ from Velri for {tensor.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velri", type=Path, required=True, help="Velri source checkout")
    parser.add_argument("--build", type=Path, required=True, help="static CMake build of the checkout")
    parser.add_argument("--output", type=Path, required=True, help="destination .gguf path")
    args = parser.parse_args()
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as workdir:
        tmp = Path(workdir)
        provenance = dump_bound_weights(args.velri.resolve(), args.build.resolve(), tmp)
        checkpoint = tmp / "model.gguf"
        writer = gguf.GGUFWriter(checkpoint, "llama")
        write_metadata(writer)
        entries, size = encode_tensors(writer, tmp)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        verify_round_trip(checkpoint, entries)
        digest = file_hash(checkpoint)
        if digest != GGUF_SHA256:
            raise ValueError(f"Generated GGUF does not match the benchmark checkpoint: {digest}")

        provenance.update(
            context=CONTEXT,
            tokenizer="Synthetic SPM with byte fallback; hosts supply numeric token IDs unchanged. No automatic BOS or EOS.",
            tensors=entries, parameter_count=size // 4, tensor_bytes=size, tensor_count=len(entries),
            gguf_sha256=digest, gguf_bytes=checkpoint.stat().st_size,
        )
        manifest = tmp / "manifest.json"
        manifest.write_text(json.dumps(provenance, indent=2) + "\n")
        checkpoint.replace(out)
        manifest.replace(out.with_suffix(".manifest.json"))
        print(f"{len(entries)} tensors, {size // 4} parameters, SHA-256 {digest}")


if __name__ == "__main__":
    main()

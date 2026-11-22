# Model Blocks and Placement Configuration

[Русский](../RU/blocks-and-config.md)

This document covers two coupled parts of the Velri engine layer: composing model architectures from reusable blocks over `velri-lib` operations, and driving device placement via JSON configuration.

**Status:** target C-engine specification. Target-schema examples live in [examples/example-dense.json](../../examples/example-dense.json) and [examples/example-moe.json](../../examples/example-moe.json).

---

## 1. Model Composition Levels

| Level | Examples | What Is Specified |
| :--- | :--- | :--- |
| **Operation** (`velri-lib`, `graph.h`) | `Linear`, `BatchedLinear`, `RMSNorm`, `LayerNorm`, `RoPE`, `Attention`, `GDN`, `FWHT` | Formula, input/output shapes & dtypes, explicit state reads/writes |
| **Reusable Block & Generic Layer** (`velri-lib`, `blocks.h`) | `velri_builder`, `velri_manifest`, `norm`, `linear`, `embedding`, `attention_core`/`dense`/`latent`, `mlp`, `lm_head`, generic `pre_norm` | Model- and config-agnostic subgraphs with zero family branches or engine scope hardcoding |
| **Engine Layer Helpers** (`Velri`, `src/model/layers.{h,c}`) | `velri_model_params`, `velri_pre_norm_layer_desc`, `velri_lm_head_desc`, placement scopes (`input-layer`, `hidden-layers`, `attention`, `kv-cache`, `dense-mlp`, `lm-head`) | Binding engine model metadata and JSON placement scopes onto `velri-lib` blocks |
| **Architecture** (`Velri`, `src/model/arch/`) | `GPT-2`, `Llama`, `Qwen3.5`, `Qwen3.8-Flash-Next`, `Gemma 4` | Full layer schedule, weight mapping, auxiliary branches (`PLE`, `MTP`, encoders) |

Reusing a block reuses its **computation structure**: weights and state are never shared implicitly — shared ownership is always declared explicitly.

### Example: `GatedMLP` standalone and inside `MoE`

```text
input ──► Linear(gate) ──► Activation ──┐
                                        × ──► Linear(down) ──► output
input ──► Linear(up)   ─────────────────┘
```

The same block acts as a standalone dense FFN or as a single expert inside `MoE`: the router selects `top-k` experts, feeds inputs, and combines outputs externally without modifying `GatedMLP`.

### Example: hybrid architecture (`GDN` + `QSA`)

```text
gdn_layer = Layer(
    mixer = GDN(...),
    ffn   = MoE(router, experts = GatedMLP(...)),
    connections = HyperConnections(...)
)

qsa_layer = Layer(
    mixer = Attention(projections, position_transform, QSASelection(...), core),
    ffn   = MoE(router, experts = GatedMLP(...)),
    connections = HyperConnections(...)
)

layers = Repeat([gdn_layer, gdn_layer, gdn_layer, qsa_layer], count)
```

This illustrates composition (not the final C API). Shared blocks contain no `if (family == GEMMA4)` branches: all differences live in explicit operations, parameters, and resource bindings.

### Attention decomposition and state storage

Attention descriptions separate five independent concerns:
1. **Input acquisition (`Q`, `K`, `V`)** (projections, stored buffer reads, or deriving one input from another).
2. **Positional transform** (`RoPE`, partial `p-RoPE`, 3D `M-RoPE`).
3. **History selection** (full context, sliding window, sparse indexers like `QSA` or `DSA` — a single `sparse` flag is insufficient; indexer cost is modeled explicitly).
4. **Attention core** (scaling, masks, `softcap`, learned sinks).
5. **State storage** (see [K/V roles in Velri-lib](../../velri-lib/docs/EN/architecture.md#3-memory-views-and-kv-roles)): separate buffers, one shared buffer, `V`-only storage with `K = RoPE(V * w_k)` derivation, or recurrent `GDN` state.

### Checklist for adding a new model
1. Document exact formulas, parameters, weights, and state updates (see [models/README.md](models/README.md)).
2. Implement metadata parsing and checkpoint tensor-name mapping in the family adapter.
3. Assemble layers and auxiliary branches from reusable blocks (adding any missing general operation + C23 reference to `velri-lib`).
4. Verify logits and state evolution across multiple steps against an independent reference, including prefill vs. decode equivalence and zero stray allocations.

---

## 2. Target Placement JSON Configuration

Configuration is parsed once at load time via the built-in zero-dependency C23 streaming JSON scanner (`src/format/json.{h,c}`), validated, and lowered into a flat placement plan (`src/config/placement.{h,c}`). Kernels never see JSON.

### Placement fields and devices
Any node in the model tree accepts three inheritable placement keys:
- **`store`** — persistent storage location;
- **`compute`** — compute device for decode (and default for prefill);
- **`prefill-compute`** *(optional)* — separate compute device for prefill.

Compute devices use `<backend>:<index>` validated against the [backend registry](../../velri-lib/docs/EN/architecture.md#5-backend-registry-and-shared-contract) with no hardcoded backend list in the parser:

| Value | Meaning |
| :--- | :--- |
| `cpu:0` | CPU compute context |
| `vulkan:0` | Compute (`compute`) or device storage (`store`) on the selected Vulkan device |
| `ram:0` | Host RAM storage (`store` only; not a compute backend) |
| `disk` | File-backed storage with explicit loading and staging buffers (`store` only) |

Device indices bind at runtime and are reported by `inspect`. Legacy `cuda:N` / `vram:N` values and ambiguous `device:0` are never silently rewritten — the parser returns an explicit migration error.

### Cascading inheritance and layer selectors
1. Only `store`, `compute`, and `prefill-compute` inherit down the model tree.
2. Under `hidden-layers`, layer selectors `"N"` and `"N-M"` (or comma-separated lists) accept the same component subtree.
3. **Precedence (highest to lowest):**
   - deepest matching node inside a matching layer selector (e.g., `hidden-layers → "32-47" → moe → routed-experts`);
   - deepest matching node directly under `hidden-layers`;
   - defaults at the `model` root.
4. Overlapping layer selectors and unknown keys are errors. Strict mode rejects out-of-range indices. In permissive mode, `inspect` flags them. The placement report includes configured component rules even when the architecture does not use those components.

During graph construction, the builder resolves placement for each storage and compute request. A separate prefill plan is prepared when the context supports multiple tokens and a compute device used by the graph differs between phases. In GPT-2 and Llama, the `kv-cache` scope places persistent state. Attention reads and writes this state on the device selected by the surrounding `attention` scope.

### Backend settings (`backends.<name>`)
Backend-specific options sit at the top level under `backends.<name>`, do not inherit through the model tree, and are validated once by the owning backend module inside `velri-lib` when creating the context.

```json
{
  "model": {
    "path": "models/qwen3.8-flash-next",
    "store": "vulkan:0",
    "compute": "vulkan:0",
    "hidden-layers": {
      "moe": {
        "routed-experts": {
          "store": "ram:0",
          "compute": "cpu:0",
          "prefill-compute": "vulkan:0"
        }
      }
    }
  },
  "backends": {
    "cpu": {
      "threads": 0,
      "reserve-cores": 1,
      "affinity": "auto"
    },
    "vulkan": {
      "prefill-chunk-size": 4096
    }
  }
}
```

- Apply `backends.cpu` when creating the context with `velri_ctx_create_from_json`. A fixed positive `threads` sets the pool size directly. The automatic mode (`threads: 0` or omitted) subtracts `reserve-cores` (default 1) from the accessible physical core count and clamps the result to 1–256 workers. Linux uses the calling thread’s affinity mask and SMT topology. Windows supports a single process affinity group. If the topology is unavailable, the engine uses one worker.
- The current pool runs unpinned. `affinity: "auto"` uses this fallback, `"none"` requests it explicitly, and `"required"` returns an unsupported-configuration error. Program inspection reports the effective worker count and pinning policy alongside the JSON settings.
- Full dense and MoE target-schema examples: [examples/example-dense.json](../../examples/example-dense.json) and [examples/example-moe.json](../../examples/example-moe.json).

### Separating parsing from capability checks
Valid JSON syntax does not imply that a device supports every required op, dtype, or transfer path. Capability, transfer, and memory-budget checks run during `prepare`. Unknown backends, disabled-at-build backends, and missing hardware produce distinct diagnostics; `inspect` prints requested placement, resolved placement, and rule provenance.

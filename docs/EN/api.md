# Public C API and Applications

[Русский](../RU/api.md)

This document defines the target public **C23 engine API** (`include/velri/`) and planned applications built on top of it. Internal computation, backends, and graphs belong to `velri-lib` ([velri-lib/docs/EN/architecture.md](../../velri-lib/docs/EN/architecture.md)); `Velri` owns models, sessions, and generation.

**Status:** the core public C23 engine API ([`include/velri/velri.h`](../../include/velri/velri.h)) and runtime (`src/runtime/engine.c`, `src/runtime/sampler.{h,c}`) — covering `Context`, `Model`, single/dual-plan zero-duplication `Program`, `Session`, cancellation, and zero-allocation sampling — are implemented and verified by `engine_runtime`. Real `.vtq` weight loading, tokenizers, chat templates, and CLI/HTTP applications remain planned.

---

## 1. Core C API Objects

Applications link the inference library directly without pulling in CLI, HTTP server, tokenizer, or chat-template layers. At minimum, an application passes `token IDs`, executes a step, and receives logits or sampled tokens.

Long-lived objects are opaque C handles with paired create/destroy functions:

| Object | Responsibility and Lifetime |
| :--- | :--- |
| `Context` | Owns the `velri-lib` device context, CPU pool, and platform resources. Must outlive all dependent objects |
| `Model` | Holds configuration, weights, and immutable prepared representations. Shared across multiple sessions |
| `Program` | Holds a validated, prepared execution plan for a specific `Model`, `Context`, and workload limits |
| `Session` | Holds mutable sequence state (history, KV/GDN state) and reserved execution slots |

Destroying an object that still has live dependents (e.g., destroying `Context` while a `Session` is alive) returns an explicit error. Public headers never expose `VkBuffer`, Slang, SIMD, or third-party types. A stable binary ABI is not promised in v1.

---

## 2. Buffer Ownership, Synchronicity, and Concurrency

- Every buffer parameter explicitly documents ownership and lifetime.
- **Initial `step` is synchronous:** caller input buffers live until the call returns; borrowed output views remain valid until the next mutating call on that session or session destruction. Inside a synchronous step, `velri-lib` backends may still execute asynchronously.
- Any future asynchronous `step` API will introduce an explicit completion handle and dedicated buffer-retention rules.
- At most one step runs concurrently on a single `Session`. Independent sessions coordinate through a shared `Context`; thread-safety rules for each function are documented in headers.

---

## 3. Errors and Cancellation

- All functions return status codes (invalid arguments, unsupported operation, out of memory/capacity, I/O error, device error) and retain diagnostic context. The library never calls `abort`/`exit` and never uses `longjmp` across caller frames.
- Dimensions, offsets, and integer overflows are validated before any writes occur.
- Cancellation or errors during an active step follow the resource-safety rules in [velri-lib/docs/EN/execution.md](../../velri-lib/docs/EN/execution.md#4-completion-errors-and-cancellation):
  - memory is never freed until all accepted device work finishes;
  - callers distinguish normal `stop`, requested cancellation, and error;
  - cancellation after the first work of a step is accepted, or an error that may have modified state, requires an explicit session `reset` (automatic partial-step rollback is not promised in v1).

---

## 4. Tokenizers and Chat Templates

Tokenization and chat formatting are adapters above the compute core and are verified independently:
- tokenizers are tested for exact `token IDs`, special tokens, Unicode normalization, byte fallback, and streaming detokenization;
- chat templates are verified separately; an unsupported template is never silently approximated (only via explicit user opt-in);
- model metadata separates text, `reasoning`, and `tool call` output events.

---

## 5. Planned CLI and HTTP/SSE

| Command | Purpose |
| :--- | :--- |
| `inspect` | Display available backends/devices, requested vs. resolved placement, active settings, and available memory accounting |
| `run`, `chat` | Run generation or interactive chat over the C API |
| `probe`, `bench` | Measure hardware ceilings, individual operations, and full model steps with recorded conditions |
| `pack` | Standalone weight repacking with explicit layout versioning |
| `quantize` | Separate weight calibration and quantization tool |
| `serve` | Optional HTTP/SSE server over the runtime |

The HTTP layer translates requests into runtime commands, bounds request and output queues (applying backpressure or cancelling slow clients), and triggers cancellation with proper device drain on client disconnect.

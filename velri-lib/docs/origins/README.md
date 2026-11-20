# CPU kernel origin

The typed CPU implementation was adapted from the user-supplied Gemini/Carbon
`velri_cpu_kernels` project on 2026-11-12. The source directory was not a Git
checkout. [SHA-256 hashes](gemini-cpu.sha256.json) identify the original files
before editing; they are provenance, not hashes of the adapted implementation.
The user authorized inspecting and adapting this source for Velri-lib.

The original microkernel blocking, register tiles and arithmetic form the starting
point. Velri-lib changes validation, dispatch, GEMV scheduling, workspace sizing,
packing alignment, integer overflow semantics, and tests. The original benchmark
and calibration tools remain local comparison artifacts; they are not runtime
dependencies. No third-party license or upstream project is inferred from model
authorship. The supplied directory contained no separate license file.

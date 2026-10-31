//! Prepare a validated block graph for bounded, single-sequence CPU execution.
//! The schedule is immutable. Weights are shared; every session owns its activations and
//! append-only sequence tensors. Unsupported placement and formats fail before execution.
mod prepare;
pub use prepare::{DenseWeights, Limits, MemoryReport, Program, Session};

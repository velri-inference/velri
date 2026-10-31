//! Mathematical parameters shared by logical builders and backend operations.

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum GatedActivation {
    SwiGlu,
    GeGlu,
}

/// Causal grouped-query attention. Key and value feature sizes need not match.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attention {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub scale: f32,
    /// Number of visible positions, including the query's own position.
    pub window: Option<usize>,
}

/// Rotate-half RoPE on the first `rotary_dim` features of each head.
/// Positions come from the row domain (current step or stored history).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rotary {
    pub heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub theta: f32,
}

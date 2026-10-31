//! Reusable build-time compositions. These objects are expanded once; they are not traversed
//! in the token loop. A failed composite build invalidates that construction attempt: discard
//! the builder rather than trying to execute a partially built model.
use crate::graph::{
    Attention, Component, GatedActivation, GraphBuilder, Rotary, StoreId, ValueId, WeightId,
};
use lumen_core::Result;

/// A single-input/single-output convenience interface. Multi-input blocks use the graph API
/// directly; this trait does not constrain model topology or resource sharing.
pub trait Block {
    fn build(&self, graph: &mut GraphBuilder, input: ValueId) -> Result<ValueId>;
}

#[derive(Clone, Copy, Debug)]
pub struct Linear {
    pub weight: WeightId,
    pub bias: Option<WeightId>,
}
impl Linear {
    pub fn new(weight: WeightId) -> Self {
        Self { weight, bias: None }
    }
}
impl Block for Linear {
    fn build(&self, g: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        g.linear(input, self.weight, self.bias)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RmsNorm {
    pub weight: WeightId,
    pub epsilon: f32,
    pub unit_offset: bool,
}
impl Block for RmsNorm {
    fn build(&self, g: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        g.rms_norm(input, self.weight, self.epsilon, self.unit_offset)
    }
}

/// down(activation(gate(x)) * up(x)); projections can reuse any declared weights.
#[derive(Clone, Debug)]
pub struct GatedMlp {
    pub component: Component,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub activation: GatedActivation,
    pub clamp: Option<f32>,
}
impl Block for GatedMlp {
    fn build(&self, g: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        let gate = self.gate.build(g, input)?;
        let up = self.up.build(g, input)?;
        let active = g.gated_activation(
            self.component.clone(),
            gate,
            up,
            self.activation,
            self.clamp,
        )?;
        self.down.build(g, active)
    }
}

/// x + scale * branch(norm(x)); works with any unary branch, not only attention or MLP.
#[derive(Clone, Debug)]
pub struct PreNormResidual<B> {
    pub component: Component,
    pub norm: RmsNorm,
    pub branch: B,
    pub scale: f32,
}
impl<B: Block> Block for PreNormResidual<B> {
    fn build(&self, g: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        let norm = self.norm.build(g, input)?;
        let branch = self.branch.build(g, norm)?;
        g.add(self.component.clone(), input, branch, self.scale)
    }
}

/// Attention has three logical operands and owns no projections or persistent buffers.
#[derive(Clone, Debug)]
pub struct AttentionBlock {
    pub component: Component,
    pub geometry: Attention,
}
impl AttentionBlock {
    pub fn build(
        &self,
        g: &mut GraphBuilder,
        q: ValueId,
        k: ValueId,
        v: ValueId,
    ) -> Result<ValueId> {
        g.attention(self.component.clone(), q, k, v, self.geometry)
    }
}

/// One explicit recipe: separate projections and stores, optional Q/K rotation, then output
/// projection. Shared/derived operands use AttentionBlock directly instead of special flags here.
#[derive(Clone, Debug)]
pub struct ProjectedAttention {
    pub attention: AttentionBlock,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub output: Linear,
    pub key_store: StoreId,
    pub value_store: StoreId,
    pub query_rotation: Option<Rotary>,
    pub key_rotation: Option<Rotary>,
}
impl Block for ProjectedAttention {
    fn build(&self, g: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        let mut q = self.q.build(g, input)?;
        let mut k = self.k.build(g, input)?;
        let v = self.v.build(g, input)?;
        if let Some(spec) = self.query_rotation {
            q = g.rotary(self.attention.component.child("query-rotation"), q, spec)?;
        }
        if let Some(spec) = self.key_rotation {
            k = g.rotary(self.attention.component.child("key-rotation"), k, spec)?;
        }
        let k = g.append(self.key_store, k)?;
        let v = g.append(self.value_store, v)?;
        let mixed = self.attention.build(g, q, k, v)?;
        self.output.build(g, mixed)
    }
}

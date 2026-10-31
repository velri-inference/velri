use lumen_core::{ComponentPath, DType, Shape};

macro_rules! handle {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name {
            pub(super) graph: u64,
            pub(super) index: usize,
        }
        impl $name {
            pub fn index(self) -> usize {
                self.index
            }
        }
    };
}
handle!(ValueId);
handle!(WeightId);
handle!(StoreId);

/// Placement identity, independent of a checkpoint tensor name or storage role.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Component {
    pub layer: Option<usize>,
    pub path: ComponentPath,
}
impl Component {
    pub fn global(path: impl Into<ComponentPath>) -> Self {
        Self {
            layer: None,
            path: path.into(),
        }
    }
    pub fn layer(layer: usize, path: impl Into<ComponentPath>) -> Self {
        Self {
            layer: Some(layer),
            path: path.into(),
        }
    }
    pub fn child(&self, suffix: &str) -> Self {
        Self {
            layer: self.layer,
            path: format!("{}/{suffix}", self.path).into(),
        }
    }
    /// Stable key used when binding actual weights to declarations.
    pub fn key(&self) -> String {
        match self.layer {
            Some(l) => format!("hidden-layers/{l}/{}", self.path),
            None => self.path.to_string(),
        }
    }
}

/// Only the leading row count varies at execution. Feature dimensions are checked at build time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    Tokens,
    History(StoreId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    pub rows: Rows,
    pub width: usize,
    pub dtype: DType,
}

#[derive(Clone, Debug)]
pub struct Weight {
    pub component: Component,
    pub shape: Shape,
}

/// An append-only sequence tensor. It has no privileged K/V meaning.
#[derive(Clone, Debug)]
pub struct SequenceStore {
    pub component: Component,
    pub width: usize,
}

pub use lumen_tensor::{Attention, GatedActivation, Rotary};

/// Logical operations have values/weight declarations, not device pointers or layer-owned caches.
#[derive(Clone, Debug)]
pub enum Operation {
    Embedding {
        tokens: ValueId,
        table: WeightId,
    },
    Linear {
        input: ValueId,
        weight: WeightId,
        bias: Option<WeightId>,
    },
    RmsNorm {
        input: ValueId,
        weight: WeightId,
        epsilon: f32,
        unit_offset: bool,
    },
    GatedActivation {
        gate: ValueId,
        up: ValueId,
        activation: GatedActivation,
        clamp: Option<f32>,
    },
    Add {
        a: ValueId,
        b: ValueId,
        scale: f32,
    },
    Rotary {
        input: ValueId,
        spec: Rotary,
    },
    Append {
        store: StoreId,
        input: ValueId,
    },
    Attention {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        spec: Attention,
    },
}
impl Operation {
    /// Data dependencies; append effects are ordered through the returned history value.
    pub fn inputs(&self) -> Vec<ValueId> {
        match *self {
            Self::Embedding { tokens, .. } => vec![tokens],
            Self::Linear { input, .. }
            | Self::RmsNorm { input, .. }
            | Self::Rotary { input, .. }
            | Self::Append { input, .. } => vec![input],
            Self::GatedActivation { gate, up, .. } => vec![gate, up],
            Self::Add { a, b, .. } => vec![a, b],
            Self::Attention { q, k, v, .. } => vec![q, k, v],
        }
    }
}
#[derive(Clone, Debug)]
pub struct Node {
    pub component: Component,
    pub operation: Operation,
    pub output: ValueId,
}

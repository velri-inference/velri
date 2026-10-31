//! Validated, backend-independent computation descriptions.
//!
//! Build-time handles are scoped to one graph. Immutable graphs can be lowered without
//! consulting the original model config. Sequence stores are explicit owners; reading a
//! previously appended store returns the same value and creates no additional allocation.
mod types;
pub use types::*;

use lumen_core::{ComponentKind, ComponentTree, DType, LayerMask, LumenError, Result, Shape};
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_GRAPH: AtomicU64 = AtomicU64::new(1);
fn invalid(message: impl Into<String>) -> LumenError {
    LumenError::Shape(message.into())
}

#[derive(Clone, Debug)]
pub struct Graph {
    id: u64,
    name: String,
    layers: usize,
    values: Vec<Value>,
    weights: Vec<Weight>,
    stores: Vec<SequenceStore>,
    nodes: Vec<Node>,
    outputs: Vec<ValueId>,
}
impl Graph {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn values(&self) -> &[Value] {
        &self.values
    }
    pub fn weights(&self) -> &[Weight] {
        &self.weights
    }
    pub fn stores(&self) -> &[SequenceStore] {
        &self.stores
    }
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }
    pub fn value(&self, id: ValueId) -> Result<&Value> {
        if id.graph != self.id {
            return Err(invalid("value belongs to another graph"));
        }
        self.values
            .get(id.index)
            .ok_or_else(|| invalid("unknown value"))
    }
    pub fn components(&self) -> Result<ComponentTree> {
        let mut declarations = BTreeMap::new();
        for (index, weight) in self.weights.iter().enumerate() {
            let gathered = self.nodes.iter().any(
                |n| matches!(n.operation, Operation::Embedding {table, ..} if table.index == index),
            );
            let projected = self.nodes.iter().any(
                |n| matches!(n.operation, Operation::Linear {weight, ..} if weight.index == index),
            );
            declarations.insert(
                weight.component.clone(),
                if gathered && !projected {
                    ComponentKind::RowTable
                } else {
                    ComponentKind::DenseWeight
                },
            );
        }
        for store in &self.stores {
            declarations.insert(store.component.clone(), ComponentKind::KvCache);
        }
        for node in &self.nodes {
            declarations
                .entry(node.component.clone())
                .or_insert(ComponentKind::ComputeOnly);
        }
        let mut tree = ComponentTree::new(self.name.clone(), self.layers);
        for (component, kind) in declarations {
            if let Some(layer) = component.layer {
                tree.add_layer(
                    component.path,
                    kind,
                    LayerMask::from_fn(self.layers, |l| l == layer),
                )?;
            } else {
                tree.add_global(component.path, kind);
            }
        }
        Ok(tree)
    }
}

pub struct GraphBuilder {
    graph: Graph,
    written: Vec<Option<ValueId>>,
}
impl GraphBuilder {
    pub fn new(name: impl Into<String>, layers: usize) -> Result<Self> {
        let id = NEXT_GRAPH
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| invalid("graph identity space exhausted"))?;
        Ok(Self {
            graph: Graph {
                id,
                name: name.into(),
                layers,
                values: vec![Value {
                    rows: Rows::Tokens,
                    width: 1,
                    dtype: DType::U32,
                }],
                weights: vec![],
                stores: vec![],
                nodes: vec![],
                outputs: vec![],
            },
            written: vec![],
        })
    }
    pub fn tokens(&self) -> ValueId {
        ValueId {
            graph: self.graph.id,
            index: 0,
        }
    }
    pub fn value(&self, id: ValueId) -> Result<&Value> {
        self.graph.value(id)
    }
    pub fn weight_spec(&self, id: WeightId) -> Result<&Weight> {
        if id.graph != self.graph.id {
            return Err(invalid("weight belongs to another graph"));
        }
        self.graph
            .weights
            .get(id.index)
            .ok_or_else(|| invalid("unknown weight"))
    }
    pub fn store_spec(&self, id: StoreId) -> Result<&SequenceStore> {
        if id.graph != self.graph.id {
            return Err(invalid("store belongs to another graph"));
        }
        self.graph
            .stores
            .get(id.index)
            .ok_or_else(|| invalid("unknown store"))
    }
    fn check_component(&self, c: &Component) -> Result<()> {
        if c.layer.is_none() && c.path.as_str().split('/').next() == Some("hidden-layers") {
            return Err(invalid(
                "hidden-layers is reserved for layer component keys",
            ));
        }
        if c.layer.is_some_and(|l| l >= self.graph.layers) {
            return Err(invalid("component layer is out of range"));
        }
        if c.path
            .as_str()
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."))
        {
            return Err(invalid(
                "component path must have nonempty ordinary segments",
            ));
        }
        Ok(())
    }
    fn check_owner(&self, c: &Component) -> Result<()> {
        self.check_component(c)?;
        if self.graph.weights.iter().any(|w| w.component == *c)
            || self.graph.stores.iter().any(|s| s.component == *c)
        {
            return Err(invalid(format!("duplicate resource {}", c.key())));
        }
        Ok(())
    }
    pub fn weight(&mut self, component: Component, shape: Shape) -> Result<WeightId> {
        self.check_owner(&component)?;
        if shape.rank() == 0
            || shape.dims().contains(&0)
            || shape.checked_byte_len(DType::F32).is_none()
        {
            return Err(invalid(
                "weight dimensions must be positive and fit in memory",
            ));
        }
        let id = WeightId {
            graph: self.graph.id,
            index: self.graph.weights.len(),
        };
        self.graph.weights.push(Weight { component, shape });
        Ok(id)
    }
    pub fn sequence_store(&mut self, component: Component, width: usize) -> Result<StoreId> {
        self.check_owner(&component)?;
        if width == 0 || width.checked_mul(4).is_none() {
            return Err(invalid("invalid sequence store width"));
        }
        let id = StoreId {
            graph: self.graph.id,
            index: self.graph.stores.len(),
        };
        self.graph.stores.push(SequenceStore { component, width });
        self.written.push(None);
        Ok(id)
    }
    fn emit(
        &mut self,
        component: Component,
        operation: Operation,
        value: Value,
    ) -> Result<ValueId> {
        self.check_component(&component)?;
        let output = ValueId {
            graph: self.graph.id,
            index: self.graph.values.len(),
        };
        self.graph.values.push(value);
        self.graph.nodes.push(Node {
            component,
            operation,
            output,
        });
        Ok(output)
    }
    fn float(&self, id: ValueId) -> Result<Value> {
        let v = self.value(id)?;
        if v.dtype != DType::F32 {
            return Err(invalid("operation requires f32 values"));
        }
        Ok(v.clone())
    }
    pub fn embedding(&mut self, tokens: ValueId, table: WeightId) -> Result<ValueId> {
        if tokens != self.tokens() {
            return Err(invalid("embedding requires this graph's token input"));
        }
        let w = self.weight_spec(table)?;
        if w.shape.rank() != 2 {
            return Err(invalid("embedding table must be a matrix"));
        }
        let value = Value {
            rows: Rows::Tokens,
            width: w.shape.dims()[1],
            dtype: DType::F32,
        };
        self.emit(
            w.component.clone(),
            Operation::Embedding { tokens, table },
            value,
        )
    }
    pub fn linear(
        &mut self,
        input: ValueId,
        weight: WeightId,
        bias: Option<WeightId>,
    ) -> Result<ValueId> {
        let mut value = self.float(input)?;
        let w = self.weight_spec(weight)?;
        if w.shape.rank() != 2 || w.shape.dims()[1] != value.width {
            return Err(invalid("linear input/weight dimensions differ"));
        }
        value.width = w.shape.dims()[0];
        if let Some(b) = bias
            && self.weight_spec(b)?.shape.dims() != [value.width]
        {
            return Err(invalid("linear bias dimensions differ"));
        }
        self.emit(
            w.component.clone(),
            Operation::Linear {
                input,
                weight,
                bias,
            },
            value,
        )
    }
    pub fn rms_norm(
        &mut self,
        input: ValueId,
        weight: WeightId,
        epsilon: f32,
        unit_offset: bool,
    ) -> Result<ValueId> {
        let value = self.float(input)?;
        let w = self.weight_spec(weight)?;
        if w.shape.dims() != [value.width] || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid("invalid RMSNorm dimensions or epsilon"));
        }
        self.emit(
            w.component.clone(),
            Operation::RmsNorm {
                input,
                weight,
                epsilon,
                unit_offset,
            },
            value,
        )
    }
    pub fn gated_activation(
        &mut self,
        component: Component,
        gate: ValueId,
        up: ValueId,
        activation: GatedActivation,
        clamp: Option<f32>,
    ) -> Result<ValueId> {
        let value = self.float(gate)?;
        if value != self.float(up)? || clamp.is_some_and(|x| !x.is_finite() || x <= 0.0) {
            return Err(invalid("invalid gated activation inputs or clamp"));
        }
        self.emit(
            component,
            Operation::GatedActivation {
                gate,
                up,
                activation,
                clamp,
            },
            value,
        )
    }
    pub fn add(
        &mut self,
        component: Component,
        a: ValueId,
        b: ValueId,
        scale: f32,
    ) -> Result<ValueId> {
        let value = self.float(a)?;
        if value != self.float(b)? || !scale.is_finite() {
            return Err(invalid("residual dimensions differ or scale is not finite"));
        }
        self.emit(component, Operation::Add { a, b, scale }, value)
    }
    pub fn rotary(
        &mut self,
        component: Component,
        input: ValueId,
        spec: Rotary,
    ) -> Result<ValueId> {
        let value = self.float(input)?;
        if spec.heads == 0
            || spec.head_dim == 0
            || spec.heads.checked_mul(spec.head_dim) != Some(value.width)
            || spec.rotary_dim == 0
            || spec.rotary_dim % 2 != 0
            || spec.rotary_dim > spec.head_dim
            || !spec.theta.is_finite()
            || spec.theta <= 0.0
        {
            return Err(invalid("invalid rotary dimensions or base"));
        }
        self.emit(component, Operation::Rotary { input, spec }, value)
    }
    pub fn append(&mut self, store: StoreId, input: ValueId) -> Result<ValueId> {
        let value = self.float(input)?;
        let s = self.store_spec(store)?;
        if value.rows != Rows::Tokens
            || value.width != s.width
            || self.written[store.index].is_some()
        {
            return Err(invalid(
                "store append requires matching current rows and exactly one writer",
            ));
        }
        let id = self.emit(
            s.component.clone(),
            Operation::Append { store, input },
            Value {
                rows: Rows::History(store),
                ..value
            },
        )?;
        self.written[store.index] = Some(id);
        Ok(id)
    }
    /// Shares the value produced by this step's writer; it does not create a copy or a second store.
    pub fn read_store(&self, store: StoreId) -> Result<ValueId> {
        self.store_spec(store)?;
        self.written[store.index]
            .ok_or_else(|| invalid("store must have an explicit writer before readers"))
    }
    pub fn attention(
        &mut self,
        component: Component,
        q: ValueId,
        k: ValueId,
        v: ValueId,
        spec: Attention,
    ) -> Result<ValueId> {
        let qv = self.float(q)?;
        let kv = self.float(k)?;
        let vv = self.float(v)?;
        if spec.query_heads == 0
            || spec.kv_heads == 0
            || spec.key_dim == 0
            || spec.value_dim == 0
            || spec.query_heads % spec.kv_heads != 0
            || !spec.scale.is_finite()
            || spec.window == Some(0)
            || qv.rows != Rows::Tokens
            || !matches!(kv.rows, Rows::History(_))
            || !matches!(vv.rows, Rows::History(_))
            || spec.query_heads.checked_mul(spec.key_dim) != Some(qv.width)
            || spec.kv_heads.checked_mul(spec.key_dim) != Some(kv.width)
            || spec.kv_heads.checked_mul(spec.value_dim) != Some(vv.width)
        {
            return Err(invalid(
                "invalid attention dimensions, head groups or row domains",
            ));
        }
        let width = spec
            .query_heads
            .checked_mul(spec.value_dim)
            .ok_or_else(|| invalid("attention output size overflows"))?;
        self.emit(
            component,
            Operation::Attention { q, k, v, spec },
            Value { width, ..qv },
        )
    }
    pub fn finish(mut self, outputs: Vec<ValueId>) -> Result<Graph> {
        if outputs.is_empty() {
            return Err(invalid("graph must have at least one output"));
        }
        for &id in &outputs {
            self.float(id)?;
        }
        if self.written.iter().any(Option::is_none) {
            return Err(invalid("every declared sequence store must have a writer"));
        }
        self.graph.outputs = outputs;
        self.graph.components()?;
        Ok(self.graph)
    }
}

use lumen_backend::{
    BindingKind, BindingSpec, ExecutionBindings, Executor, Op, RowExtent, WeightTable,
};
use lumen_config::{Place, PlacementPlan};
use lumen_core::{DType, Device, LumenError, MemLoc, Result, Shape};
use lumen_mem::{AlignedHostBuffer, Buffer, BufferId};
use lumen_models::graph::{Component, Graph, Operation, Rows};
use lumen_tensor::{DenseLinear, MatWeight, RowCodec, RowTable, TensorRef, WeightRef};
use std::{collections::BTreeMap, sync::Arc};

fn invalid(s: impl Into<String>) -> LumenError {
    LumenError::Config(s.into())
}
fn overflow() -> LumenError {
    LumenError::Memory("prepared memory size overflows".into())
}

/// Explicit f32 bindings for the declared weight keys. Checkpoint normalization is a separate
/// concern; this input is also useful for calibration and numerical fixtures.
#[derive(Default)]
pub struct DenseWeights(BTreeMap<String, Vec<f32>>);
impl DenseWeights {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, name: impl Into<String>, data: Vec<f32>) -> Result<()> {
        let name = name.into();
        if self.0.contains_key(&name) {
            return Err(invalid(format!("duplicate weight {name}")));
        }
        self.0.insert(name, data);
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_tokens: usize,
    pub max_context: usize,
    pub max_memory_bytes: usize,
}

/// Tensor payload for one session, counting shared weight owners once. Metadata, allocator
/// bookkeeping and other sessions are not included. No activation reuse is claimed yet.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryReport {
    pub weight_bytes: usize,
    pub state_bytes: usize,
    pub activation_bytes: usize,
    pub input_bytes: usize,
    pub total_bytes: usize,
}

pub struct Program {
    operations: Vec<Op>,
    weights: WeightTable,
    bindings: Vec<BindingSpec>,
    input: TensorRef,
    outputs: Vec<TensorRef>,
    limits: Limits,
    memory: MemoryReport,
    device: Device,
}
impl Program {
    pub fn prepare(
        graph: &Graph,
        placement: &PlacementPlan,
        mut data: DenseWeights,
        limits: Limits,
        executor: &dyn Executor,
    ) -> Result<Self> {
        if limits.max_tokens == 0 || limits.max_context == 0 {
            return Err(invalid("execution capacities must be positive"));
        }
        if executor.device() != Device::Cpu(0) {
            return Err(invalid("this preparation path supports cpu:0 only"));
        }
        if placement
            .kv
            .iter()
            .any(|p| p.k_bits.is_some() || p.v_bits.is_some())
        {
            return Err(invalid(
                "quantized sequence storage is not supported by the dense f32 preparation path",
            ));
        }
        for c in graph
            .weights()
            .iter()
            .map(|w| &w.component)
            .chain(graph.stores().iter().map(|s| &s.component))
            .chain(graph.nodes().iter().map(|n| &n.component))
        {
            let p = resolve(placement, c)?;
            if p.compute != Device::Cpu(0)
                || p.effective_prefill_compute() != Device::Cpu(0)
                || p.store != MemLoc::Ram(0)
            {
                return Err(invalid(format!(
                    "unsupported placement for {}: first execution path requires ram:0 / cpu:0",
                    c.key()
                )));
            }
        }
        // Validate all supplied data before allocating the prepared buffers.
        for w in graph.weights() {
            let key = w.component.key();
            let values = data
                .0
                .get(&key)
                .ok_or_else(|| invalid(format!("missing weight {key}")))?;
            if w.shape.checked_numel() != Some(values.len())
                || values.iter().any(|x| !x.is_finite())
            {
                return Err(invalid(format!(
                    "invalid dimensions or non-finite values for {key}"
                )));
            }
        }
        if data.0.len() != graph.weights().len() {
            return Err(invalid("supplied weights include undeclared keys"));
        }

        let mut bindings = Vec::new();
        let input = bind(
            &mut bindings,
            Shape::from([limits.max_tokens]),
            DType::U32,
            1,
            RowExtent::Step,
            BindingKind::Tokens,
        )?;
        let mut weight_refs = Vec::new();
        for w in graph.weights() {
            let width = *w
                .shape
                .dims()
                .last()
                .ok_or_else(|| invalid("empty weight shape"))?;
            let rows = w.shape.checked_numel().ok_or_else(overflow)? / width;
            weight_refs.push(bind(
                &mut bindings,
                w.shape.clone(),
                DType::F32,
                width,
                RowExtent::Fixed(rows),
                BindingKind::Weight,
            )?);
        }
        let mut stores = Vec::new();
        for store in graph.stores() {
            stores.push(bind(
                &mut bindings,
                Shape::from([limits.max_context, store.width]),
                DType::F32,
                store.width,
                RowExtent::History,
                BindingKind::Sequence,
            )?);
        }
        let mut values = vec![None; graph.values().len()];
        values[0] = Some(input.clone());
        let mut weights = WeightTable::default();
        let mut operations = Vec::new();
        for node in graph.nodes() {
            let value = graph.value(node.output)?;
            let out = if let Operation::Append { store, .. } = node.operation {
                stores[store.index()].clone()
            } else {
                let (rows, extent) = match value.rows {
                    Rows::Tokens => (limits.max_tokens, RowExtent::Step),
                    Rows::History(_) => (limits.max_context, RowExtent::History),
                };
                bind(
                    &mut bindings,
                    Shape::from([rows, value.width]),
                    value.dtype,
                    value.width,
                    extent,
                    BindingKind::Activation,
                )?
            };
            let get = |id: lumen_models::graph::ValueId| -> Result<TensorRef> {
                values
                    .get(id.index())
                    .and_then(Option::as_ref)
                    .cloned()
                    .ok_or_else(|| invalid("graph value has no preceding producer"))
            };
            let op = match node.operation {
                Operation::Embedding { tokens, table } => {
                    let t = &weight_refs[table.index()];
                    let d = t.shape.dims();
                    let table = weights.add_row_table(RowTable {
                        rows: u64::try_from(d[0]).map_err(|_| overflow())?,
                        dim: u32::try_from(d[1]).map_err(|_| overflow())?,
                        codec: RowCodec::Dense(DType::F32),
                        shards: vec![weight_ref(t)?],
                        signs: None,
                        head_offsets: None,
                        head_bias: None,
                    });
                    Op::RowLookup {
                        table,
                        indices: get(tokens)?,
                        out: out.clone(),
                    }
                }
                Operation::Linear {
                    input,
                    weight,
                    bias,
                } => {
                    let t = &weight_refs[weight.index()];
                    let d = t.shape.dims();
                    let bias = bias
                        .map(|b| weight_ref(&weight_refs[b.index()]))
                        .transpose()?;
                    let weight = weights.add_mat(MatWeight::Dense(DenseLinear {
                        m: u32::try_from(d[0]).map_err(|_| overflow())?,
                        k: u32::try_from(d[1]).map_err(|_| overflow())?,
                        dtype: DType::F32,
                        weight: weight_ref(t)?,
                        bias,
                    }));
                    Op::Linear {
                        weight,
                        input: get(input)?,
                        out: out.clone(),
                    }
                }
                Operation::RmsNorm {
                    input,
                    weight,
                    epsilon,
                    unit_offset,
                } => Op::RmsNorm {
                    input: get(input)?,
                    weight: Some(weight_refs[weight.index()].clone()),
                    out: out.clone(),
                    eps: epsilon,
                    unit_offset,
                },
                Operation::GatedActivation {
                    gate,
                    up,
                    activation,
                    clamp,
                } => Op::GatedMlpAct {
                    gate: get(gate)?,
                    up: get(up)?,
                    out: out.clone(),
                    act: activation,
                    clamp,
                },
                Operation::Add { a, b, scale } => Op::ResidualAdd {
                    a: get(a)?,
                    b: get(b)?,
                    out: out.clone(),
                    scale,
                },
                Operation::Rotary { input, spec } => Op::Rope {
                    input: get(input)?,
                    out: out.clone(),
                    spec,
                },
                Operation::Append { input, .. } => Op::Append {
                    input: get(input)?,
                    out: out.clone(),
                },
                Operation::Attention { q, k, v, spec } => Op::Attention {
                    q: get(q)?,
                    k: get(k)?,
                    v: get(v)?,
                    out: out.clone(),
                    spec,
                },
            };
            if !executor.supports(&op) {
                return Err(LumenError::UnsupportedOp {
                    device: executor.device(),
                    op: format!("{op:?}"),
                });
            }
            values[node.output.index()] = Some(out);
            operations.push(op);
        }
        let outputs = graph
            .outputs()
            .iter()
            .map(|id| {
                values[id.index()]
                    .clone()
                    .ok_or_else(|| invalid("output has no producer"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut memory = MemoryReport::default();
        for spec in &bindings {
            let bytes = spec
                .tensor
                .shape
                .checked_byte_len(spec.tensor.dtype)
                .ok_or_else(overflow)?;
            let subtotal = match spec.kind {
                BindingKind::Weight => &mut memory.weight_bytes,
                BindingKind::Sequence => &mut memory.state_bytes,
                BindingKind::Activation => &mut memory.activation_bytes,
                BindingKind::Tokens => &mut memory.input_bytes,
            };
            *subtotal = subtotal.checked_add(bytes).ok_or_else(overflow)?;
            memory.total_bytes = memory.total_bytes.checked_add(bytes).ok_or_else(overflow)?;
        }
        if memory.total_bytes > limits.max_memory_bytes {
            return Err(LumenError::Memory(format!(
                "prepared payload {} exceeds budget {}",
                memory.total_bytes, limits.max_memory_bytes
            )));
        }
        for (w, tensor) in graph.weights().iter().zip(&weight_refs) {
            let values = data
                .0
                .remove(&w.component.key())
                .ok_or_else(|| invalid("missing validated weight"))?;
            let bytes = tensor
                .shape
                .checked_byte_len(DType::F32)
                .ok_or_else(overflow)?;
            let mut buffer = AlignedHostBuffer::new_zeroed(bytes, 0)?;
            for (chunk, value) in buffer.as_mut_slice().chunks_exact_mut(4).zip(values) {
                chunk.copy_from_slice(&value.to_ne_bytes());
            }
            bindings[tensor.id.0 as usize].initial = Some(Arc::new(Buffer::Host(buffer)));
        }
        Ok(Self {
            operations,
            weights,
            bindings,
            input,
            outputs,
            limits,
            memory,
            device: executor.device(),
        })
    }
    pub fn memory(&self) -> MemoryReport {
        self.memory
    }
    pub fn operations(&self) -> &[Op] {
        &self.operations
    }
    pub fn session(&self) -> Result<Session<'_>> {
        Ok(Session {
            program: self,
            bindings: ExecutionBindings::new(
                &self.bindings,
                self.limits.max_tokens,
                self.limits.max_context,
            )?,
            output_valid: false,
        })
    }
}

pub struct Session<'p> {
    program: &'p Program,
    bindings: ExecutionBindings,
    output_valid: bool,
}
impl Session<'_> {
    pub fn run(&mut self, executor: &mut dyn Executor, tokens: &[u32]) -> Result<()> {
        self.output_valid = false;
        if executor.device() != self.program.device {
            return Err(invalid("executor device differs from prepared program"));
        }
        self.bindings.begin(&self.program.input, tokens)?;
        for op in &self.program.operations {
            if let Err(e) = executor.execute_op(op, &self.program.weights, &mut self.bindings) {
                self.bindings.abort();
                return Err(e);
            }
        }
        self.bindings.commit()?;
        self.output_valid = true;
        Ok(())
    }
    pub fn output(&self, index: usize) -> Result<&[f32]> {
        if !self.output_valid {
            return Err(invalid("no successful step output is available"));
        }
        let output = self
            .program
            .outputs
            .get(index)
            .ok_or_else(|| invalid("output index out of bounds"))?;
        self.bindings.f32(output)
    }
    pub fn position(&self) -> usize {
        self.bindings.position()
    }
    pub fn reset(&mut self) {
        self.bindings.reset();
        self.output_valid = false;
    }
    pub fn bound_payload_bytes(&self) -> usize {
        self.bindings.payload_bytes()
    }
}

fn resolve<'a>(plan: &'a PlacementPlan, c: &Component) -> Result<&'a Place> {
    let place = match c.layer {
        Some(l) => plan.layers.get(l).and_then(|m| m.get(&c.path)),
        None => plan.global.get(&c.path),
    };
    place.ok_or_else(|| invalid(format!("placement missing for {}", c.key())))
}
fn bind(
    bindings: &mut Vec<BindingSpec>,
    shape: Shape,
    dtype: DType,
    row_width: usize,
    extent: RowExtent,
    kind: BindingKind,
) -> Result<TensorRef> {
    shape.checked_byte_len(dtype).ok_or_else(overflow)?;
    let id = BufferId(u32::try_from(bindings.len()).map_err(|_| overflow())?);
    let tensor = TensorRef { id, dtype, shape };
    bindings.push(BindingSpec {
        tensor: tensor.clone(),
        row_width,
        extent,
        kind,
        initial: None,
    });
    Ok(tensor)
}
fn weight_ref(t: &TensorRef) -> Result<WeightRef> {
    Ok(WeightRef {
        buffer: t.id,
        byte_offset: 0,
        byte_len: u64::try_from(t.shape.checked_byte_len(t.dtype).ok_or_else(overflow)?)
            .map_err(|_| overflow())?,
        dtype: t.dtype,
        shape: t.shape.clone(),
    })
}

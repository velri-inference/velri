//! Buffer bindings and step state shared by execution preparation and backends.
//! All storage is allocated before a step. Sequence tensors append only past the committed
//! frontier, so a failed step can discard its new rows without copying the old history.
use lumen_core::{DType, LumenError, Result};
use lumen_mem::{AlignedHostBuffer, Buffer, BufferId};
use lumen_tensor::{TensorRef, WeightRef};
use std::sync::Arc;

fn error(s: impl Into<String>) -> LumenError {
    LumenError::Memory(s.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowExtent {
    Fixed(usize),
    Step,
    History,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    Weight,
    Activation,
    Sequence,
    Tokens,
}

#[derive(Clone, Debug)]
pub struct BindingSpec {
    pub tensor: TensorRef,
    pub row_width: usize,
    pub extent: RowExtent,
    pub kind: BindingKind,
    pub initial: Option<Arc<Buffer>>,
}
#[derive(Debug)]
enum Storage {
    Owned(Buffer),
    Shared(Arc<Buffer>),
}
impl Storage {
    fn buffer(&self) -> &Buffer {
        match self {
            Self::Owned(b) => b,
            Self::Shared(b) => b,
        }
    }
    fn writable(&mut self) -> Result<&mut [u8]> {
        match self {
            Self::Owned(Buffer::Host(b)) => Ok(b.as_mut_slice()),
            _ => Err(error("buffer is not writable host memory")),
        }
    }
}
#[derive(Debug)]
struct Binding {
    spec: BindingSpec,
    storage: Storage,
}

/// Per-sequence bindings. Immutable weights can be shared by several sequences.
pub struct ExecutionBindings {
    buffers: Vec<Binding>,
    max_tokens: usize,
    max_context: usize,
    committed: usize,
    start: usize,
    tokens: usize,
    active: bool,
}
impl ExecutionBindings {
    pub fn new(specs: &[BindingSpec], max_tokens: usize, max_context: usize) -> Result<Self> {
        if max_tokens == 0 || max_context == 0 {
            return Err(error("capacities must be positive"));
        }
        let mut buffers = Vec::with_capacity(specs.len());
        for (i, spec) in specs.iter().enumerate() {
            let valid_kind = match spec.kind {
                BindingKind::Tokens => {
                    spec.tensor.dtype == DType::U32
                        && spec.extent == RowExtent::Step
                        && spec.row_width == 1
                }
                BindingKind::Sequence => {
                    spec.tensor.dtype == DType::F32 && spec.extent == RowExtent::History
                }
                BindingKind::Activation => {
                    spec.tensor.dtype == DType::F32
                        && matches!(spec.extent, RowExtent::Step | RowExtent::History)
                }
                BindingKind::Weight => {
                    spec.tensor.dtype == DType::F32 && matches!(spec.extent, RowExtent::Fixed(_))
                }
            };
            if !valid_kind {
                return Err(error("binding kind, dtype and row extent disagree"));
            }
            let bytes = spec
                .tensor
                .shape
                .checked_byte_len(spec.tensor.dtype)
                .ok_or_else(|| error("buffer size overflows"))?;
            let rows = match spec.extent {
                RowExtent::Fixed(n) => n,
                RowExtent::Step => max_tokens,
                RowExtent::History => max_context,
            };
            let axes_match = match spec.kind {
                BindingKind::Tokens => spec.tensor.shape.dims() == [rows],
                BindingKind::Activation | BindingKind::Sequence => {
                    spec.tensor.shape.dims() == [rows, spec.row_width]
                }
                BindingKind::Weight => spec.tensor.shape.dims().last() == Some(&spec.row_width),
            };
            if spec.tensor.id.0 as usize != i
                || !axes_match
                || spec.row_width == 0
                || rows == 0
                || rows.checked_mul(spec.row_width) != spec.tensor.shape.checked_numel()
            {
                return Err(error("inconsistent buffer binding dimensions or id"));
            }
            let storage = match &spec.initial {
                Some(b) if spec.kind == BindingKind::Weight && b.len() == bytes => {
                    Storage::Shared(Arc::clone(b))
                }
                Some(_) => return Err(error("only exact-sized weights may share initial storage")),
                None if spec.kind == BindingKind::Weight => {
                    return Err(error("weight storage is missing"));
                }
                None => Storage::Owned(Buffer::Host(AlignedHostBuffer::new_zeroed(bytes, 0)?)),
            };
            buffers.push(Binding {
                spec: spec.clone(),
                storage,
            });
        }
        Ok(Self {
            buffers,
            max_tokens,
            max_context,
            committed: 0,
            start: 0,
            tokens: 0,
            active: false,
        })
    }
    pub fn position(&self) -> usize {
        self.committed
    }
    pub fn step_start(&self) -> usize {
        self.start
    }
    pub fn step_tokens(&self) -> usize {
        self.tokens
    }
    pub fn ensure_active(&self) -> Result<()> {
        if self.active {
            Ok(())
        } else {
            Err(error("no active execution step"))
        }
    }
    pub fn begin(&mut self, input: &TensorRef, tokens: &[u32]) -> Result<()> {
        if self.active
            || tokens.is_empty()
            || tokens.len() > self.max_tokens
            || self
                .committed
                .checked_add(tokens.len())
                .is_none_or(|n| n > self.max_context)
        {
            return Err(error("invalid step length or sequence capacity exceeded"));
        }
        self.validate(input)?;
        let b = &mut self.buffers[input.id.0 as usize];
        if b.spec.kind != BindingKind::Tokens || input.dtype != DType::U32 {
            return Err(error("invalid token input binding"));
        }
        let bytes = b.storage.writable()?;
        for (i, token) in tokens.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&token.to_ne_bytes());
        }
        self.start = self.committed;
        self.tokens = tokens.len();
        self.active = true;
        Ok(())
    }
    pub fn commit(&mut self) -> Result<()> {
        self.ensure_active()?;
        self.committed = self.start + self.tokens;
        self.active = false;
        Ok(())
    }
    pub fn abort(&mut self) {
        self.active = false;
    }
    pub fn reset(&mut self) {
        self.committed = 0;
        self.start = 0;
        self.tokens = 0;
        self.active = false;
    }
    pub fn validate(&self, tensor: &TensorRef) -> Result<()> {
        let b = self
            .buffers
            .get(tensor.id.0 as usize)
            .ok_or_else(|| error("unknown buffer id"))?;
        if b.spec.tensor != *tensor {
            return Err(error("tensor descriptor does not match its binding"));
        }
        Ok(())
    }
    pub fn extent(&self, tensor: &TensorRef) -> Result<RowExtent> {
        self.validate(tensor)?;
        Ok(self.buffers[tensor.id.0 as usize].spec.extent)
    }
    pub fn rows(&self, tensor: &TensorRef) -> Result<usize> {
        Ok(match self.extent(tensor)? {
            RowExtent::Fixed(n) => n,
            RowExtent::Step => self.tokens,
            RowExtent::History => self.start + self.tokens,
        })
    }
    pub fn width(&self, tensor: &TensorRef) -> Result<usize> {
        self.validate(tensor)?;
        Ok(self.buffers[tensor.id.0 as usize].spec.row_width)
    }
    pub fn buffer(&self, id: BufferId) -> Result<&Buffer> {
        self.buffers
            .get(id.0 as usize)
            .map(|b| b.storage.buffer())
            .ok_or_else(|| error("unknown buffer id"))
    }
    pub fn f32(&self, tensor: &TensorRef) -> Result<&[f32]> {
        self.validate(tensor)?;
        if tensor.dtype != DType::F32 {
            return Err(error("expected f32 tensor"));
        }
        let b = &self.buffers[tensor.id.0 as usize];
        let bytes = b
            .storage
            .buffer()
            .host_slice()
            .ok_or_else(|| error("tensor is not host accessible"))?;
        let count = self.rows(tensor)? * b.spec.row_width;
        bytemuck::try_cast_slice(&bytes[..count * 4])
            .map_err(|e| error(format!("invalid f32 binding: {e}")))
    }
    pub fn read(&self, tensor: &TensorRef, index: usize) -> Result<f32> {
        self.f32(tensor)?
            .get(index)
            .copied()
            .ok_or_else(|| error("tensor read out of bounds"))
    }
    pub fn token(&self, tensor: &TensorRef, index: usize) -> Result<u32> {
        self.validate(tensor)?;
        if tensor.dtype != DType::U32 || index >= self.tokens {
            return Err(error("token read out of bounds or wrong dtype"));
        }
        let bytes = self
            .buffer(tensor.id)?
            .host_slice()
            .ok_or_else(|| error("token buffer is not host accessible"))?;
        Ok(u32::from_ne_bytes(
            bytes[index * 4..index * 4 + 4]
                .try_into()
                .map_err(|_| error("invalid token bytes"))?,
        ))
    }
    pub fn weight(&self, w: &WeightRef) -> Result<&[f32]> {
        let b = self
            .buffers
            .get(w.buffer.0 as usize)
            .ok_or_else(|| error("unknown weight buffer"))?;
        if b.spec.kind != BindingKind::Weight
            || w.dtype != DType::F32
            || w.shape.checked_byte_len(w.dtype).map(|n| n as u64) != Some(w.byte_len)
        {
            return Err(error("invalid f32 weight descriptor"));
        }
        let start = usize::try_from(w.byte_offset).map_err(|_| error("weight offset overflows"))?;
        let len = usize::try_from(w.byte_len).map_err(|_| error("weight length overflows"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| error("weight range overflows"))?;
        let bytes = b
            .storage
            .buffer()
            .host_slice()
            .and_then(|s| s.get(start..end))
            .ok_or_else(|| error("weight range is unavailable"))?;
        bytemuck::try_cast_slice(bytes).map_err(|e| error(format!("invalid f32 weight: {e}")))
    }
    pub fn write(&mut self, tensor: &TensorRef, index: usize, value: f32) -> Result<()> {
        self.ensure_active()?;
        self.validate(tensor)?;
        let width = self.width(tensor)?;
        if tensor.dtype != DType::F32 || index >= self.rows(tensor)? * width || !value.is_finite() {
            return Err(error("invalid tensor write or non-finite result"));
        }
        let b = &mut self.buffers[tensor.id.0 as usize];
        if b.spec.kind != BindingKind::Activation {
            return Err(error("only activation outputs may be overwritten"));
        }
        b.storage.writable()?[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
        Ok(())
    }
    pub fn append(&mut self, input: &TensorRef, output: &TensorRef) -> Result<()> {
        self.ensure_active()?;
        self.validate(input)?;
        self.validate(output)?;
        let width = self.width(input)?;
        if input.dtype != DType::F32
            || output.dtype != DType::F32
            || self.width(output)? != width
            || self.extent(input)? != RowExtent::Step
            || self.buffers[output.id.0 as usize].spec.kind != BindingKind::Sequence
        {
            return Err(error(
                "append requires current rows and a matching sequence store",
            ));
        }
        for i in 0..self.tokens * width {
            let value = self.read(input, i)?;
            let index = self.start * width + i;
            self.buffers[output.id.0 as usize].storage.writable()?[index * 4..index * 4 + 4]
                .copy_from_slice(&value.to_ne_bytes());
        }
        Ok(())
    }
    /// Actual reserved tensor payload, excluding Rust metadata and allocator bookkeeping.
    pub fn payload_bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.storage.buffer().len()).sum()
    }
}

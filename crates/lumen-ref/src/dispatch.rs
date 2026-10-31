use lumen_backend::{ExecutionBindings, Op, RowExtent, WeightTable};
use lumen_core::{DType, Device, LumenError, Result};
use lumen_tensor::{MatWeight, RowCodec, TensorRef};

fn invalid(s: &str) -> LumenError {
    LumenError::Shape(s.into())
}
fn matrix(t: &TensorRef) -> bool {
    t.dtype == DType::F32
        && t.shape.rank() == 2
        && !t.shape.dims().contains(&0)
        && t.shape.checked_byte_len(t.dtype).is_some()
}
fn same(a: &TensorRef, b: &TensorRef) -> bool {
    matrix(a) && matrix(b) && a.shape == b.shape && a.id != b.id
}

pub(super) fn supports(op: &Op, device: Device) -> bool {
    if device != Device::Cpu(0) {
        return false;
    }
    match op {
        Op::RowLookup { indices, out, .. } => {
            indices.dtype == DType::U32
                && indices.shape.rank() == 1
                && matrix(out)
                && indices.id != out.id
                && indices.shape.dims()[0] == out.shape.dims()[0]
        }
        Op::Linear { input, out, .. } => {
            matrix(input)
                && matrix(out)
                && input.id != out.id
                && input.shape.dims()[0] == out.shape.dims()[0]
        }
        Op::RmsNorm {
            input,
            weight,
            out,
            eps,
            ..
        } => {
            same(input, out)
                && eps.is_finite()
                && *eps > 0.0
                && weight.as_ref().is_none_or(|w| {
                    w.dtype == DType::F32
                        && w.shape.dims() == [input.shape.dims()[1]]
                        && w.id != out.id
                })
        }
        Op::GatedMlpAct {
            gate,
            up,
            out,
            clamp,
            ..
        } => same(gate, out) && same(up, out) && clamp.is_none_or(|c| c.is_finite() && c > 0.0),
        Op::ResidualAdd { a, b, out, scale } => same(a, out) && same(b, out) && scale.is_finite(),
        Op::Rope { input, out, spec } => {
            same(input, out)
                && spec.heads > 0
                && spec.head_dim > 0
                && spec.heads.checked_mul(spec.head_dim) == Some(input.shape.dims()[1])
                && spec.rotary_dim > 0
                && spec.rotary_dim % 2 == 0
                && spec.rotary_dim <= spec.head_dim
                && spec.theta.is_finite()
                && spec.theta > 0.0
        }
        Op::Append { input, out } => {
            matrix(input)
                && matrix(out)
                && input.id != out.id
                && input.shape.dims()[1] == out.shape.dims()[1]
        }
        Op::Attention { q, k, v, out, spec } => {
            matrix(q)
                && matrix(k)
                && matrix(v)
                && matrix(out)
                && out.id != q.id
                && out.id != k.id
                && out.id != v.id
                && q.shape.dims()[0] == out.shape.dims()[0]
                && k.shape.dims()[0] == v.shape.dims()[0]
                && spec.query_heads > 0
                && spec.kv_heads > 0
                && spec.query_heads % spec.kv_heads == 0
                && spec.key_dim > 0
                && spec.value_dim > 0
                && spec.scale.is_finite()
                && spec.window != Some(0)
                && spec.query_heads.checked_mul(spec.key_dim) == Some(q.shape.dims()[1])
                && spec.kv_heads.checked_mul(spec.key_dim) == Some(k.shape.dims()[1])
                && spec.kv_heads.checked_mul(spec.value_dim) == Some(v.shape.dims()[1])
                && spec.query_heads.checked_mul(spec.value_dim) == Some(out.shape.dims()[1])
        }
        Op::RmsNormGated { .. }
        | Op::IndexerScore { .. }
        | Op::DeltaNet { .. }
        | Op::Router { .. }
        | Op::MoEGemm { .. }
        | Op::HcMix { .. }
        | Op::HcInject { .. }
        | Op::Transfer { .. }
        | Op::Fetch { .. }
        | Op::Sample { .. } => false,
    }
}

pub(super) fn execute(
    op: &Op,
    weights: &WeightTable,
    ctx: &mut ExecutionBindings,
    device: Device,
) -> Result<()> {
    ctx.ensure_active()?;
    if !supports(op, device) {
        return Err(LumenError::UnsupportedOp {
            device,
            op: super::RefExecutor::op_name(op).into(),
        });
    }
    match op {
        Op::RowLookup {
            table,
            indices,
            out,
        } => {
            let table = weights
                .row_table(*table)
                .ok_or_else(|| invalid("unknown row table"))?;
            if table.codec != RowCodec::Dense(DType::F32)
                || table.shards.len() != 1
                || table.signs.is_some()
                || table.head_offsets.is_some()
                || table.head_bias.is_some()
            {
                return Err(invalid("reference row lookup supports one plain f32 table"));
            }
            let width = table.dim as usize;
            let rows = usize::try_from(table.rows).map_err(|_| invalid("row count overflows"))?;
            if ctx.width(out)? != width
                || ctx.rows(out)? != ctx.step_tokens()
                || table.shards[0].shape.dims() != [rows, width]
            {
                return Err(invalid("row table dimensions differ"));
            }
            ctx.weight(&table.shards[0])?;
            for row in 0..ctx.step_tokens() {
                let token = ctx.token(indices, row)? as usize;
                if token >= rows {
                    return Err(invalid("token id is outside vocabulary"));
                }
                for col in 0..width {
                    let x = ctx.weight(&table.shards[0])?[token * width + col];
                    ctx.write(out, row * width + col, x)?;
                }
            }
        }
        Op::Linear { weight, input, out } => {
            let Some(MatWeight::Dense(w)) = weights.mat(*weight) else {
                return Err(invalid(
                    "reference dispatch currently requires a dense linear weight",
                ));
            };
            let (m, k) = (w.m as usize, w.k as usize);
            if w.dtype != DType::F32
                || w.weight.shape.dims() != [m, k]
                || ctx.width(input)? != k
                || ctx.width(out)? != m
                || ctx.rows(input)? != ctx.rows(out)?
            {
                return Err(invalid("linear dimensions or dtype differ"));
            }
            ctx.weight(&w.weight)?;
            if let Some(b) = &w.bias {
                if b.shape.dims() != [m] {
                    return Err(invalid("bias dimensions differ"));
                }
                ctx.weight(b)?;
            }
            for row in 0..ctx.rows(input)? {
                for j in 0..m {
                    let mut sum = match &w.bias {
                        Some(b) => f64::from(ctx.weight(b)?[j]),
                        None => 0.0,
                    };
                    for i in 0..k {
                        sum += f64::from(ctx.read(input, row * k + i)?)
                            * f64::from(ctx.weight(&w.weight)?[j * k + i]);
                    }
                    ctx.write(out, row * m + j, sum as f32)?;
                }
            }
        }
        Op::RmsNorm {
            input,
            weight,
            out,
            eps,
            unit_offset,
        } => {
            let width = ctx.width(input)?;
            if ctx.rows(input)? != ctx.rows(out)? {
                return Err(invalid("norm row counts differ"));
            }
            if let Some(w) = weight
                && ctx.f32(w)?.len() != width
            {
                return Err(invalid("norm weight dimensions differ"));
            }
            for row in 0..ctx.rows(input)? {
                let mut sum = 0.0;
                for i in 0..width {
                    let x = f64::from(ctx.read(input, row * width + i)?);
                    sum += x * x;
                }
                let inv = 1.0 / (sum / width as f64 + f64::from(*eps)).sqrt();
                for i in 0..width {
                    let scale = match weight {
                        Some(w) => {
                            f64::from(ctx.read(w, i)?) + if *unit_offset { 1.0 } else { 0.0 }
                        }
                        None => 1.0,
                    };
                    let x = f64::from(ctx.read(input, row * width + i)?);
                    ctx.write(out, row * width + i, (x * inv * scale) as f32)?;
                }
            }
        }
        Op::GatedMlpAct {
            gate,
            up,
            out,
            act,
            clamp,
        } => {
            let n = ctx.f32(gate)?.len();
            if ctx.f32(up)?.len() != n || ctx.f32(out)?.len() != n {
                return Err(invalid("activation lengths differ"));
            }
            for i in 0..n {
                let g = ctx.read(gate, i)?;
                let u = ctx.read(up, i)?;
                let mut result = [0.0];
                super::ref_gated_mlp_act_f32(&[g], &[u], *act, *clamp, &mut result);
                ctx.write(out, i, result[0])?;
            }
        }
        Op::ResidualAdd { a, b, out, scale } => {
            let n = ctx.f32(a)?.len();
            if ctx.f32(b)?.len() != n || ctx.f32(out)?.len() != n {
                return Err(invalid("residual lengths differ"));
            }
            for i in 0..n {
                let y = f64::from(ctx.read(a, i)?) + f64::from(*scale) * f64::from(ctx.read(b, i)?);
                ctx.write(out, i, y as f32)?;
            }
        }
        Op::Rope { input, out, spec } => {
            let width = ctx.width(input)?;
            if ctx.rows(input)? != ctx.rows(out)? {
                return Err(invalid("rotary row counts differ"));
            }
            let start = if ctx.extent(input)? == RowExtent::Step {
                ctx.step_start()
            } else {
                0
            };
            for row in 0..ctx.rows(input)? {
                for head in 0..spec.heads {
                    let base = row * width + head * spec.head_dim;
                    let half = spec.rotary_dim / 2;
                    for i in 0..half {
                        let angle = (start + row) as f64
                            / f64::from(spec.theta).powf((2 * i) as f64 / spec.rotary_dim as f64);
                        let (sin, cos) = angle.sin_cos();
                        let x = f64::from(ctx.read(input, base + i)?);
                        let y = f64::from(ctx.read(input, base + i + half)?);
                        ctx.write(out, base + i, (x * cos - y * sin) as f32)?;
                        ctx.write(out, base + i + half, (y * cos + x * sin) as f32)?;
                    }
                    for i in spec.rotary_dim..spec.head_dim {
                        let x = ctx.read(input, base + i)?;
                        ctx.write(out, base + i, x)?;
                    }
                }
            }
        }
        Op::Append { input, out } => ctx.append(input, out)?,
        Op::Attention { q, k, v, out, spec } => {
            if ctx.extent(q)? != RowExtent::Step
                || ctx.extent(k)? != RowExtent::History
                || ctx.extent(v)? != RowExtent::History
                || ctx.rows(k)? != ctx.rows(v)?
                || ctx.rows(q)? != ctx.rows(out)?
            {
                return Err(invalid("attention row domains differ"));
            }
            let qw = ctx.width(q)?;
            let kw = ctx.width(k)?;
            let vw = ctx.width(v)?;
            let ow = ctx.width(out)?;
            for row in 0..ctx.step_tokens() {
                let end = ctx.step_start() + row + 1;
                if end > ctx.rows(k)? {
                    return Err(invalid("attention history is too short"));
                }
                let first = spec.window.map_or(0, |w| end.saturating_sub(w));
                for head in 0..spec.query_heads {
                    let kh = head / (spec.query_heads / spec.kv_heads);
                    let qi = row * qw + head * spec.key_dim;
                    let score = |ctx: &ExecutionBindings, pos: usize| -> Result<f64> {
                        let mut sum = 0.0;
                        let ki = pos * kw + kh * spec.key_dim;
                        for d in 0..spec.key_dim {
                            sum +=
                                f64::from(ctx.read(q, qi + d)?) * f64::from(ctx.read(k, ki + d)?);
                        }
                        Ok(sum * f64::from(spec.scale))
                    };
                    let mut max = f64::NEG_INFINITY;
                    for pos in first..end {
                        max = max.max(score(ctx, pos)?);
                    }
                    let mut denom = 0.0;
                    for pos in first..end {
                        denom += (score(ctx, pos)? - max).exp();
                    }
                    for d in 0..spec.value_dim {
                        let mut sum = 0.0;
                        for pos in first..end {
                            sum += (score(ctx, pos)? - max).exp()
                                * f64::from(ctx.read(v, pos * vw + kh * spec.value_dim + d)?);
                        }
                        ctx.write(
                            out,
                            row * ow + head * spec.value_dim + d,
                            (sum / denom) as f32,
                        )?;
                    }
                }
            }
        }
        Op::RmsNormGated { .. }
        | Op::IndexerScore { .. }
        | Op::DeltaNet { .. }
        | Op::Router { .. }
        | Op::MoEGemm { .. }
        | Op::HcMix { .. }
        | Op::HcInject { .. }
        | Op::Transfer { .. }
        | Op::Fetch { .. }
        | Op::Sample { .. } => {
            return Err(LumenError::UnsupportedOp {
                device,
                op: super::RefExecutor::op_name(op).into(),
            });
        }
    }
    Ok(())
}

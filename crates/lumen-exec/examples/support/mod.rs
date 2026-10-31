//! Synthetic decoder shared by the executable example and numerical integration tests.
//! All actual computation is assembled from public, reusable blocks.
use lumen_core::{Result, Shape};
use lumen_exec::{DenseWeights, Limits, Program};
use lumen_models::{
    blocks::{
        AttentionBlock, Block, GatedMlp, Linear, PreNormResidual, ProjectedAttention, RmsNorm,
    },
    graph::{Attention, Component, GatedActivation, Graph, GraphBuilder, Rotary},
};
use lumen_ref::RefExecutor;

pub fn graph() -> Result<Graph> {
    let mut g = GraphBuilder::new("synthetic-decoder", 2)?;
    let embed = g.weight(Component::global("embedding"), Shape::from([5, 4]))?;
    let mut x = g.embedding(g.tokens(), embed)?;
    for layer in 0..2 {
        let mut linear = |path: &str, m, k| -> Result<Linear> {
            Ok(Linear::new(g.weight(
                Component::layer(layer, path),
                Shape::from([m, k]),
            )?))
        };
        let q = linear("attention/q", 8, 4)?;
        let k = linear("attention/k", 4, 4)?;
        let v = linear("attention/v", 3, 4)?;
        let output = linear("attention/output", 4, 6)?;
        let gate = linear("mlp/gate", 6, 4)?;
        let up = linear("mlp/up", 6, 4)?;
        let down = linear("mlp/down", 4, 6)?;
        let norm = RmsNorm {
            weight: g.weight(Component::layer(layer, "attention/norm"), Shape::from([4]))?,
            epsilon: 1e-5,
            unit_offset: false,
        };
        let attention = ProjectedAttention {
            attention: AttentionBlock {
                component: Component::layer(layer, "attention/mix"),
                geometry: Attention {
                    query_heads: 2,
                    kv_heads: 1,
                    key_dim: 4,
                    value_dim: 3,
                    scale: 0.5,
                    window: if layer == 0 { None } else { Some(2) },
                },
            },
            q,
            k,
            v,
            output,
            key_store: g.sequence_store(Component::layer(layer, "attention/key-history"), 4)?,
            value_store: g.sequence_store(Component::layer(layer, "attention/value-history"), 3)?,
            query_rotation: Some(Rotary {
                heads: 2,
                head_dim: 4,
                rotary_dim: 2,
                theta: 10000.,
            }),
            key_rotation: Some(Rotary {
                heads: 1,
                head_dim: 4,
                rotary_dim: 2,
                theta: 10000.,
            }),
        };
        x = PreNormResidual {
            component: Component::layer(layer, "attention/residual"),
            norm,
            branch: attention,
            scale: 1.,
        }
        .build(&mut g, x)?;
        let norm = RmsNorm {
            weight: g.weight(Component::layer(layer, "mlp/norm"), Shape::from([4]))?,
            epsilon: 1e-5,
            unit_offset: false,
        };
        let mlp = GatedMlp {
            component: Component::layer(layer, "mlp/activation"),
            gate,
            up,
            down,
            activation: GatedActivation::SwiGlu,
            clamp: None,
        };
        x = PreNormResidual {
            component: Component::layer(layer, "mlp/residual"),
            norm,
            branch: mlp,
            scale: 1.,
        }
        .build(&mut g, x)?;
    }
    let norm = RmsNorm {
        weight: g.weight(Component::global("final-norm"), Shape::from([4]))?,
        epsilon: 1e-5,
        unit_offset: false,
    };
    x = norm.build(&mut g, x)?;
    let head = Linear::new(g.weight(Component::global("head"), Shape::from([5, 4]))?);
    x = head.build(&mut g, x)?;
    g.finish(vec![x])
}

pub fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("../../tests/fixtures/decoder.json")).unwrap()
}

pub fn program(backend: &RefExecutor) -> Result<Program> {
    let g = graph()?;
    let mut weights = DenseWeights::new();
    for (key, value) in fixture()["weights"].as_object().unwrap() {
        weights.insert(
            key,
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect(),
        )?;
    }
    let cfg = lumen_config::RuntimeConfig::from_json_str(
        r#"{"model":{"path":"unused","compute":"cpu:0","store":"ram:0"}}"#,
    )?;
    let tree = g.components()?;
    let plan = cfg.resolve_plan(tree.clone(), &tree, true)?;
    Program::prepare(
        &g,
        &plan,
        weights,
        Limits {
            max_tokens: 6,
            max_context: 12,
            max_memory_bytes: 1 << 20,
        },
        backend,
    )
}

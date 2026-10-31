use lumen_config::RuntimeConfig;
use lumen_core::Shape;
use lumen_exec::{DenseWeights, Limits, Program};
use lumen_models::graph::{Attention, Component, Graph, GraphBuilder};
use lumen_ref::RefExecutor;

fn shared_graph() -> Graph {
    let mut g = GraphBuilder::new("shared", 1).unwrap();
    let w = g
        .weight(Component::global("embedding"), Shape::from([3, 2]))
        .unwrap();
    let x = g.embedding(g.tokens(), w).unwrap();
    let s = g.sequence_store(Component::layer(0, "history"), 2).unwrap();
    let history = g.append(s, x).unwrap();
    let borrowed = g.read_store(s).unwrap();
    let y = g
        .attention(
            Component::layer(0, "attention"),
            x,
            history,
            borrowed,
            Attention {
                query_heads: 1,
                kv_heads: 1,
                key_dim: 2,
                value_dim: 2,
                scale: 1.0,
                window: None,
            },
        )
        .unwrap();
    g.finish(vec![y]).unwrap()
}
fn weights() -> DenseWeights {
    let mut w = DenseWeights::new();
    w.insert("embedding", vec![1., 0., 0., 1., 1., 1.]).unwrap();
    w
}
fn limits() -> Limits {
    Limits {
        max_tokens: 4,
        max_context: 8,
        max_memory_bytes: 1 << 20,
    }
}
fn placement(graph: &Graph, compute: &str) -> lumen_config::PlacementPlan {
    let cfg = RuntimeConfig::from_json_str(&format!(
        r#"{{"model":{{"path":"unused","store":"ram:0","compute":"{compute}"}}}}"#
    ))
    .unwrap();
    let tree = graph.components().unwrap();
    cfg.resolve_plan(tree.clone(), &tree, true).unwrap()
}

#[test]
fn shared_storage_is_counted_once_and_decode_matches_prefill() {
    let g = shared_graph();
    let mut backend = RefExecutor::default();
    let program =
        Program::prepare(&g, &placement(&g, "cpu:0"), weights(), limits(), &backend).unwrap();
    assert_eq!(program.memory().state_bytes, 8 * 2 * 4);
    assert_eq!(program.memory().weight_bytes, 3 * 2 * 4);
    let mut whole = program.session().unwrap();
    whole.run(&mut backend, &[0, 1, 2]).unwrap();
    let expected = [1., 0., 0.26894143, 0.7310586, 0.78805846, 0.78805846];
    for (a, b) in whole.output(0).unwrap().iter().zip(expected) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }
    let mut decode = program.session().unwrap();
    for (i, token) in [0, 1, 2].into_iter().enumerate() {
        decode.run(&mut backend, &[token]).unwrap();
        for (a, b) in decode
            .output(0)
            .unwrap()
            .iter()
            .zip(&expected[i * 2..i * 2 + 2])
        {
            assert!((a - b).abs() < 1e-6);
        }
    }
    assert_eq!(decode.position(), 3);
}

#[test]
fn preparation_rejects_unsupported_placement_missing_weights_and_too_small_budget() {
    use lumen_core::LumenError;
    let g = shared_graph();
    let backend = RefExecutor::default();
    assert!(matches!(
        Program::prepare(&g, &placement(&g, "cuda:0"), weights(), limits(), &backend),
        Err(LumenError::Config(_))
    ));
    assert!(
        matches!(Program::prepare(&g, &placement(&g, "cpu:0"), DenseWeights::new(), limits(), &backend), Err(LumenError::Config(s)) if s.contains("missing weight"))
    );
    assert!(
        matches!(Program::prepare(&g, &placement(&g, "cpu:0"), weights(), Limits {max_memory_bytes:16,..limits()}, &backend), Err(LumenError::Memory(s)) if s.contains("exceeds budget"))
    );
    assert!(matches!(
        Program::prepare(
            &g,
            &placement(&g, "cpu:0"),
            weights(),
            Limits {
                max_context: usize::MAX,
                ..limits()
            },
            &backend
        ),
        Err(LumenError::Memory(_))
    ));
    for data in [vec![0.; 5], vec![f32::NAN; 6], vec![f32::INFINITY; 6]] {
        let mut w = DenseWeights::new();
        w.insert("embedding", data).unwrap();
        assert!(
            matches!(Program::prepare(&g,&placement(&g,"cpu:0"),w,limits(),&backend),Err(LumenError::Config(s)) if s.contains("invalid dimensions or non-finite"))
        );
    }
    let mut w = weights();
    w.insert("unknown", vec![1.]).unwrap();
    assert!(
        matches!(Program::prepare(&g,&placement(&g,"cpu:0"),w,limits(),&backend),Err(LumenError::Config(s)) if s.contains("undeclared"))
    );
}

#[test]
fn invalid_token_and_capacity_errors_do_not_advance_sequence() {
    let g = shared_graph();
    let mut backend = RefExecutor::default();
    let program =
        Program::prepare(&g, &placement(&g, "cpu:0"), weights(), limits(), &backend).unwrap();
    let mut s = program.session().unwrap();
    s.run(&mut backend, &[0]).unwrap();
    assert!(s.run(&mut backend, &[999]).is_err());
    assert_eq!(s.position(), 1);
    assert!(s.output(0).is_err());
    s.run(&mut backend, &[1]).unwrap();
    assert!((s.output(0).unwrap()[0] - 0.26894143).abs() < 1e-6);
    assert!(s.run(&mut backend, &[0; 5]).is_err());
    assert_eq!(s.position(), 2);
    s.reset();
    assert_eq!(s.position(), 0);
    s.run(&mut backend, &[1]).unwrap();
    assert_eq!(s.output(0).unwrap(), [0., 1.]);
}

#[test]
fn quantized_state_requests_are_not_silently_executed_as_f32() {
    let g = shared_graph();
    let backend = RefExecutor::default();
    let mut plan = placement(&g, "cpu:0");
    assert!(!plan.kv.is_empty());
    plan.kv[0].k_bits = Some(4);
    assert!(matches!(
        Program::prepare(&g, &plan, weights(), limits(), &backend),
        Err(lumen_core::LumenError::Config(_))
    ));
}

#[test]
fn derived_history_can_supply_either_operand_and_scratch_is_accounted() {
    for derive_keys in [true, false] {
        let mut g = GraphBuilder::new("derived-history", 2).unwrap();
        let embed = g
            .weight(Component::global("embedding"), Shape::from([3, 2]))
            .unwrap();
        let x = g.embedding(g.tokens(), embed).unwrap();
        let store = g
            .sequence_store(Component::layer(0, "shared-history"), 2)
            .unwrap();
        let history = g.append(store, x).unwrap();
        let transform = g
            .weight(Component::layer(1, "transform"), Shape::from([2, 2]))
            .unwrap();
        let derived = g.linear(history, transform, None).unwrap();
        let borrowed = g.read_store(store).unwrap();
        let (k, v) = if derive_keys {
            (derived, borrowed)
        } else {
            (borrowed, derived)
        };
        let y = g
            .attention(
                Component::layer(1, "mix"),
                x,
                k,
                v,
                Attention {
                    query_heads: 1,
                    kv_heads: 1,
                    key_dim: 2,
                    value_dim: 2,
                    scale: 1.,
                    window: None,
                },
            )
            .unwrap();
        let g = g.finish(vec![y]).unwrap();
        let mut weights = weights();
        weights
            .insert("hidden-layers/1/transform", vec![1., 0., 0., 2.])
            .unwrap();
        let mut backend = RefExecutor::default();
        let program =
            Program::prepare(&g, &placement(&g, "cpu:0"), weights, limits(), &backend).unwrap();
        assert_eq!(program.memory().state_bytes, 8 * 2 * 4);
        assert_eq!(
            program.memory().activation_bytes,
            (4 * 2 + 8 * 2 + 4 * 2) * 4
        );
        let mut s = program.session().unwrap();
        assert_eq!(s.bound_payload_bytes(), program.memory().total_bytes);
        s.run(&mut backend, &[0, 1]).unwrap();
        let expected = if derive_keys {
            [
                1.,
                0.,
                1. / (1. + 2_f32.exp()),
                2_f32.exp() / (1. + 2_f32.exp()),
            ]
        } else {
            [
                1.,
                0.,
                1. / (1. + 1_f32.exp()),
                2. * 1_f32.exp() / (1. + 1_f32.exp()),
            ]
        };
        for (a, b) in s.output(0).unwrap().iter().zip(expected) {
            assert!((a - b).abs() < 1e-6);
        }
        let whole = s.output(0).unwrap().to_vec();
        s.reset();
        for (i, t) in [0, 1].into_iter().enumerate() {
            s.run(&mut backend, &[t]).unwrap();
            assert_eq!(s.output(0).unwrap(), &whole[i * 2..i * 2 + 2]);
        }
    }
}

#[test]
fn failure_after_append_preserves_committed_history_and_allows_retry() {
    let mut g = GraphBuilder::new("retry", 1).unwrap();
    let embed = g
        .weight(Component::global("embedding"), Shape::from([3, 2]))
        .unwrap();
    let x = g.embedding(g.tokens(), embed).unwrap();
    let store = g.sequence_store(Component::layer(0, "history"), 2).unwrap();
    let history = g.append(store, x).unwrap();
    let y = g
        .attention(
            Component::layer(0, "mix"),
            x,
            history,
            history,
            Attention {
                query_heads: 1,
                kv_heads: 1,
                key_dim: 2,
                value_dim: 2,
                scale: 1.,
                window: None,
            },
        )
        .unwrap();
    // Deliberate finite input whose projection overflows after append has executed.
    let check = g
        .weight(Component::global("check"), Shape::from([1, 2]))
        .unwrap();
    g.linear(x, check, None).unwrap();
    let g = g.finish(vec![y]).unwrap();
    let mut w = DenseWeights::new();
    w.insert("embedding", vec![1., 0., 0., 1., f32::MAX, 0.])
        .unwrap();
    w.insert("check", vec![2., 0.]).unwrap();
    let mut backend = RefExecutor::default();
    let program = Program::prepare(
        &g,
        &placement(&g, "cpu:0"),
        w,
        Limits {
            max_context: 3,
            ..limits()
        },
        &backend,
    )
    .unwrap();
    let mut s = program.session().unwrap();
    s.run(&mut backend, &[0]).unwrap();
    assert!(s.run(&mut backend, &[2]).is_err());
    assert_eq!(s.position(), 1);
    assert!(s.output(0).is_err());
    s.run(&mut backend, &[1]).unwrap();
    assert!((s.output(0).unwrap()[0] - 0.26894143).abs() < 1e-6);
    assert!(s.run(&mut backend, &[]).is_err());
    assert!(s.run(&mut backend, &[0, 1]).is_err());
    assert_eq!(s.position(), 2);
    s.run(&mut backend, &[0]).unwrap();
    assert_eq!(s.position(), 3);
}

#[test]
fn embedding_and_output_projection_share_one_weight_owner() {
    let mut g = GraphBuilder::new("tied", 0).unwrap();
    let w = g
        .weight(Component::global("embedding"), Shape::from([3, 2]))
        .unwrap();
    let x = g.embedding(g.tokens(), w).unwrap();
    let y = g.linear(x, w, None).unwrap();
    let g = g.finish(vec![y]).unwrap();
    let mut backend = RefExecutor::default();
    let program =
        Program::prepare(&g, &placement(&g, "cpu:0"), weights(), limits(), &backend).unwrap();
    assert_eq!(program.memory().weight_bytes, 3 * 2 * 4);
    let mut s = program.session().unwrap();
    assert_eq!(s.bound_payload_bytes(), program.memory().total_bytes);
    s.run(&mut backend, &[0, 2]).unwrap();
    assert_eq!(s.output(0).unwrap(), [1., 0., 1., 1., 1., 2.]);
}

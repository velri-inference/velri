use lumen_core::Shape;
use lumen_models::graph::{Attention, Component, GraphBuilder};

fn attention() -> Attention {
    Attention {
        query_heads: 2,
        kv_heads: 1,
        key_dim: 2,
        value_dim: 2,
        scale: 0.5,
        window: None,
    }
}

#[test]
fn shared_sequence_storage_is_one_resource_not_two_attention_slots() {
    let mut g = GraphBuilder::new("shared", 2).unwrap();
    let table = g
        .weight(Component::global("embedding"), Shape::from([8, 2]))
        .unwrap();
    let x = g.embedding(g.tokens(), table).unwrap();
    let q_weight = g
        .weight(Component::layer(0, "attention/q"), Shape::from([4, 2]))
        .unwrap();
    let q = g.linear(x, q_weight, None).unwrap();
    let store = g
        .sequence_store(Component::layer(0, "attention/history"), 2)
        .unwrap();
    let history = g.append(store, x).unwrap();
    let borrowed = g.read_store(store).unwrap();
    let y = g
        .attention(
            Component::layer(1, "attention"),
            q,
            history,
            borrowed,
            attention(),
        )
        .unwrap();
    let graph = g.finish(vec![y]).unwrap();
    assert_eq!(graph.stores().len(), 1);
    assert_eq!(history, borrowed);
    assert_eq!(graph.value(y).unwrap().width, 4);
    assert!(
        graph
            .components()
            .unwrap()
            .is_valid_layer_path(&"attention/history".into())
    );
}

#[test]
fn graph_rejects_foreign_handles_even_if_their_indices_match() {
    let mut a = GraphBuilder::new("a", 0).unwrap();
    let mut b = GraphBuilder::new("b", 0).unwrap();
    let wa = a
        .weight(Component::global("projection"), Shape::from([2, 2]))
        .unwrap();
    let wb = b
        .weight(Component::global("projection"), Shape::from([2, 2]))
        .unwrap();
    let x = a.embedding(a.tokens(), wa).unwrap();
    assert!(a.linear(x, wb, None).is_err());
    assert!(b.linear(x, wb, None).is_err());
}

#[test]
fn invalid_shapes_and_duplicate_writes_are_rejected_without_losing_valid_graph() {
    let mut g = GraphBuilder::new("checked", 1).unwrap();
    assert!(
        g.weight(Component::global("empty"), Shape::from([0, 2]))
            .is_err()
    );
    assert!(
        g.weight(Component::layer(1, "bad"), Shape::from([2, 2]))
            .is_err()
    );
    let w = g
        .weight(Component::global("embedding"), Shape::from([3, 2]))
        .unwrap();
    assert!(
        g.weight(Component::global("embedding"), Shape::from([3, 2]))
            .is_err()
    );
    let x = g.embedding(g.tokens(), w).unwrap();
    let bad = g
        .weight(Component::global("bad-shape"), Shape::from([3, 4]))
        .unwrap();
    assert!(g.linear(x, bad, None).is_err());
    let state = g.sequence_store(Component::layer(0, "history"), 2).unwrap();
    assert!(g.read_store(state).is_err());
    let history = g.append(state, x).unwrap();
    assert!(g.append(state, x).is_err());
    assert!(g.append(state, history).is_err());
    assert!(
        g.attention(
            Component::layer(0, "attention"),
            x,
            history,
            history,
            attention()
        )
        .is_err()
    );
    assert!(g.finish(vec![history]).is_ok());
}

#[test]
fn attention_accepts_different_key_and_value_widths_and_validates_head_groups() {
    let mut g = GraphBuilder::new("dimensions", 1).unwrap();
    let w = g
        .weight(Component::global("input"), Shape::from([4, 6]))
        .unwrap();
    let q = g.embedding(g.tokens(), w).unwrap();
    let kw = g
        .weight(Component::layer(0, "key"), Shape::from([2, 6]))
        .unwrap();
    let vw = g
        .weight(Component::layer(0, "value"), Shape::from([3, 6]))
        .unwrap();
    let k = g.linear(q, kw, None).unwrap();
    let v = g.linear(q, vw, None).unwrap();
    let ks = g.sequence_store(Component::layer(0, "a"), 2).unwrap();
    let vs = g.sequence_store(Component::layer(0, "b"), 3).unwrap();
    let k = g.append(ks, k).unwrap();
    let v = g.append(vs, v).unwrap();
    let spec = Attention {
        query_heads: 3,
        kv_heads: 1,
        key_dim: 2,
        value_dim: 3,
        scale: 1.0,
        window: Some(2),
    };
    let y = g
        .attention(Component::layer(0, "attention"), q, k, v, spec)
        .unwrap();
    assert_eq!(g.value(y).unwrap().width, 9);
    assert!(
        g.attention(
            Component::layer(0, "invalid"),
            q,
            k,
            v,
            Attention {
                kv_heads: 2,
                ..spec
            }
        )
        .is_err()
    );
}

#[test]
fn residual_mlp_reuses_operations_and_weights_across_compositions() {
    use lumen_models::blocks::{Block, GatedMlp, Linear, PreNormResidual, RmsNorm};
    use lumen_models::graph::GatedActivation;
    let mut g = GraphBuilder::new("composition", 1).unwrap();
    let input = g
        .weight(Component::global("embedding"), Shape::from([5, 4]))
        .unwrap();
    let x = g.embedding(g.tokens(), input).unwrap();
    let norm = g
        .weight(Component::layer(0, "norm"), Shape::from([4]))
        .unwrap();
    let gate = g
        .weight(Component::layer(0, "mlp/gate"), Shape::from([6, 4]))
        .unwrap();
    let up = g
        .weight(Component::layer(0, "mlp/up"), Shape::from([6, 4]))
        .unwrap();
    let down = g
        .weight(Component::layer(0, "mlp/down"), Shape::from([4, 6]))
        .unwrap();
    let block = PreNormResidual {
        component: Component::layer(0, "mlp/residual"),
        norm: RmsNorm {
            weight: norm,
            epsilon: 1e-5,
            unit_offset: false,
        },
        branch: GatedMlp {
            component: Component::layer(0, "mlp/activation"),
            gate: Linear::new(gate),
            up: Linear::new(up),
            down: Linear::new(down),
            activation: GatedActivation::SwiGlu,
            clamp: None,
        },
        scale: 1.0,
    };
    let y = block.build(&mut g, x).unwrap();
    let z = block.build(&mut g, y).unwrap(); // Same weights can be reused by another computation.
    assert_eq!(g.value(z).unwrap().width, 4);
    let graph = g.finish(vec![z]).unwrap();
    assert_eq!(graph.weights().len(), 5);
    assert!(graph.stores().is_empty());
}

#[test]
fn global_components_cannot_collide_with_serialized_layer_keys() {
    let mut g = GraphBuilder::new("namespaces", 1).unwrap();
    let local = g
        .weight(Component::layer(0, "w"), Shape::from([2, 2]))
        .unwrap();
    assert!(
        g.weight(Component::global("hidden-layers/0/w"), Shape::from([2, 2]))
            .is_err()
    );
    assert!(
        g.sequence_store(Component::global("hidden-layers"), 2)
            .is_err()
    );
    let x = g.embedding(g.tokens(), local).unwrap();
    g.finish(vec![x]).unwrap();
}

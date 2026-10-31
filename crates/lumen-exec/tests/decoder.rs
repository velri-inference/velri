#[path = "../examples/support/mod.rs"]
mod support;

#[test]
fn composed_decoder_matches_independent_oracle_for_every_chunking() {
    let mut backend = lumen_ref::RefExecutor::default();
    let program = support::program(&backend).unwrap();
    let fixture = support::fixture();
    let tokens: Vec<u32> = fixture["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let expected: Vec<f32> = fixture["logits"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
        })
        .collect();
    assert_eq!(program.memory().state_bytes, 2 * (4 + 3) * 12 * 4);
    // Every possible partition of six tokens covers full prefill, mixed chunks and decode.
    for mask in 0..(1 << (tokens.len() - 1)) {
        let mut session = program.session().unwrap();
        assert_eq!(session.bound_payload_bytes(), program.memory().total_bytes);
        let mut start = 0;
        for end in 1..=tokens.len() {
            if end == tokens.len() || mask & (1 << (end - 1)) != 0 {
                session.run(&mut backend, &tokens[start..end]).unwrap();
                let actual = session.output(0).unwrap();
                assert_eq!(actual.len(), (end - start) * 5);
                for (a, e) in actual.iter().zip(&expected[start * 5..end * 5]) {
                    assert!(
                        (a - e).abs() < 2e-6,
                        "partition {mask}, rows {start}..{end}: {a} != {e}"
                    );
                }
                start = end;
            }
        }
        assert_eq!(session.position(), tokens.len());
    }
}

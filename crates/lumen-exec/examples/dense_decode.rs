//! Run with: cargo run -p lumen-exec --example dense_decode
mod support;

fn main() -> lumen_core::Result<()> {
    let mut backend = lumen_ref::RefExecutor::default();
    let program = support::program(&backend)?;
    println!(
        "Synthetic two-layer decoder, reserved payload: {:?}",
        program.memory()
    );
    let mut sequence = program.session()?;
    for token in support::fixture()["tokens"].as_array().unwrap() {
        let token = token.as_u64().unwrap() as u32;
        sequence.run(&mut backend, &[token])?;
        println!(
            "position {}, input {token}, logits {:?}",
            sequence.position(),
            sequence.output(0)?
        );
    }
    Ok(())
}

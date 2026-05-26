# Tests added for MLP + clipped ReLU lookup

Run the lightweight tests:

```bash
cargo test --release mlp_clipped_relu_lookup_tests
```

Run all non-ignored tests:

```bash
cargo test --release
```

Run the heavier end-to-end Nova proof test explicitly:

```bash
cargo test --release small_recursive_mlp_clipped_relu_proof_verifies -- --ignored --nocapture
```

The tests cover:

1. `ClippedReluLookupTable` semantics.
2. `IntMlpClippedReluStepParams` block length and flatten layout.
3. `generate_mlp_clipped_relu_trace` against a hand-computed two-layer MLP step.
4. Panic behavior when a hidden pre-activation is outside the lookup table range.
5. Chunking trace into recursive step circuits.
6. An ignored end-to-end RecursiveSNARK prove/verify smoke test.

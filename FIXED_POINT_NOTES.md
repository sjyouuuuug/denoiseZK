# Fixed-point MLP + clipped ReLU notes

This source tree keeps the existing integer MLP baseline and adds a fixed-point implementation.

## New modules

- `src/fixed_point/`: shared fixed-point helpers
  - `config.rs`: scale and scaled ReLU bounds
  - `encode.rs`: f64 <-> fixed-point integer conversion
  - `arith.rs`: mathematical floor division and rescale helpers
  - `gadget.rs`: R1CS gadgets for floor rescale and range checking the remainder
- `src/affine/fixed_point.rs`: reusable fixed-point affine helpers
- `src/clipped_relu/fixed_point.rs`: fixed-point clipped ReLU wrappers
- `src/mlp_fixed_point_clipped_relu_lookup/`: public-parameter two-layer fixed-point MLP circuit
- `src/bin/mlp_fixed_point_clipped_relu_lookup_demo.rs`: runnable demo
- `tests/fixed_point_mlp_clipped_relu_tests.rs`: tests for encoding, floor division, ReLU, affine, trace, chunking, and an ignored proof test

## Semantics

- f64 values are encoded with `round(x * scale)`.
- In-circuit rescale uses mathematical floor:

```text
q = floor(z / scale)
z = q * scale + r, 0 <= r < scale
```

- Affine layer uses:

```text
y = floor((W * x) / scale) + b
```

- Parameters are public and time-varying. The public state layout is:

```text
z0 = [x0 | params_0 | params_1 | ... | params_{T-1}]
```

where each parameter block is:

```text
[W1 row-major | b1 | W2 row-major | b2]
```

## Run

```bash
cargo run --release --bin mlp_fixed_point_clipped_relu_lookup_demo
```

## Tests

```bash
cargo test --release fixed_point_mlp_clipped_relu_tests
```

Run the heavier ignored proof test:

```bash
cargo test --release fixed_point_small_recursive_mlp_proof_verifies -- --ignored --nocapture
```

# denoise-zk

`denoise-zk` is a Rust prototype for proving fixed-point denoising computations with Nova-based IVC. It focuses on DDIM-style deterministic denoise steps, MLP and single-channel Conv predictor backends, lookup-based nonlinearities, range-check optimizations, and experiment scripts for the thesis.

## Quick Start

Build all binaries:

```bash
cargo build --release --bins
```

Run the main experiment entry:

```bash
cargo run --release --bin denoise_experiments -- correctness
```

Run all experiment groups:

```bash
cargo run --release --bin denoise_experiments -- all
```

Run tests:

```bash
cargo test --release
```

Some proof tests are ignored because they are slow:

```bash
cargo test --release -- --ignored --nocapture
```

## Main Demos

```bash
# fixed-point MLP / denoise
cargo run --release --bin mlp_fixed_point_clipped_relu_lookup_demo
cargo run --release --bin denoise_fixed_point_time_embedding_padded_demo

# Conv denoise backend
cargo run --release --bin denoise_fixed_point_conv_demo

# toy model commitment demos
cargo run --release --bin denoise_fixed_point_mlp_commitment_demo
cargo run --release --bin denoise_fixed_point_conv_commitment_demo

# fused update demo
cargo run --release --bin denoise_fixed_point_conv_fused_demo
```

Older affine and small lookup demos may still exist for development checks, but thesis-facing experiments focus on denoise MLP/Conv cases with state dimension `N >= 16`.

## Experiments

Experiment outputs are written under:

```text
outputs/experiments/<group>/
```

Available groups:

```bash
cargo run --release --bin denoise_experiments -- correctness
cargo run --release --bin denoise_experiments -- range
cargo run --release --bin denoise_experiments -- fused_update
cargo run --release --bin denoise_experiments -- mlp_scaling
cargo run --release --bin denoise_experiments -- conv_scaling
cargo run --release --bin denoise_experiments -- backend_compare
cargo run --release --bin denoise_experiments -- steps_scaling
cargo run --release --bin denoise_experiments -- large_scale
```

Each group produces:

```text
denoise_experiments_<group>.md
denoise_experiments_<group>.json
denoise_experiments_<group>.csv
```

`BuildOnly` means the circuit is synthesized and constraints/variables are counted, but a proof is not generated.

## Visual Demo

Export JSON data:

```bash
bash scripts/export_visual_demo.sh
```

Run the frontend:

```bash
cd visual_demo
npm install
npm run dev
```

The frontend reads `visual_demo/public/denoise_demo.json` and displays denoise heatmaps, Nova step grouping, proof summary, and complexity comparisons.

## Project Layout

```text
src/
├── activations/      ReLU and clipped-ReLU lookup components
├── commitment/       toy parameter commitment and gadgets
├── experiments/      experiment runners and report generation
├── fixed_point/      quantized arithmetic, rescale, range gadgets
├── layers/           affine, MLP, Conv2d helpers
├── models/           composed denoise / MLP / Conv prototypes
├── padding/          dimension padding utilities
├── public_state/     public state layouts and final checks
├── runners/          Nova setup/prove/verify helpers
├── visualization/    JSON schema and visual export helpers
└── bin/              runnable demos and experiment binaries
```

## Core Statement

The main denoise commitment state has the form:

```text
z = [x | y | h | C | t | params_queue | time_table]
```

The circuit proves repeated transitions:

```text
e_t = TimeEmbedding(t)
epsilon_t = Predictor_t(x_t, e_t)
x_{t+1} = DenoiseUpdate(x_t, epsilon_t, alpha_t, beta_t)
h_{t+1} = HashUpdate(h_t, params_t)
```

At the final timestep, the circuit conditionally enforces:

```text
x_T = y
h_T = C
```

The current commitment is a toy algebraic hash:

```text
h_{i+1} = h_i * 131 + v_i
```

It is not cryptographically binding; it only demonstrates commitment plumbing and can be replaced by Poseidon or a Merkle commitment later.

## Notes

- Fixed-point scale is typically `S = 16`.
- Remainder checks use small one-hot lookup tables.
- Signed range checks support bit decomposition for power-of-two-width ranges such as `[-128, 127]`.
- `DoubleFloor` and `FusedFloor` are both supported. `FusedFloor` reduces rescale constraints but has different fixed-point semantics.
- The repository expects the Nova dependency under `lib/Nova`.

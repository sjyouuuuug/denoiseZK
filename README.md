# denoise-zk Nova demos

This `src/` tree contains several small Nova-based zkML prototypes. The code is organized as reusable modules plus runnable demos under `src/bin/`.

The current focus is:

1. integer affine / MLP baselines,
2. lookup-table ReLU and clipped ReLU,
3. fixed-point MLP with floor rescaling,
4. fixed-point denoise updates,
5. public per-iteration parameters carried in the public state.

The project assumes the full crate has a `Cargo.toml` at the project root and a local Nova checkout under:

```text
lib/Nova
```

A typical full project layout is:

```text
denoise-zk/
├── Cargo.toml
├── lib/
│   └── Nova/
└── src/
    ├── ...
```

## 1. Build and run

From the project root, run demos with:

```bash
cargo run --release --bin <demo_name>
```

Run tests with:

```bash
cargo test --release
```

Some Nova proof tests are intentionally marked `#[ignore]` because they run setup/prove/verify and are slower. Run ignored proof tests explicitly with:

```bash
cargo test --release -- --ignored --nocapture
```

If your Nova dependency uses HyperKZG and you are running without ptau files, make sure your `Cargo.toml` enables test utilities for local experiments, for example:

```toml
nova-snark = { path = "lib/Nova", features = ["test-utils"] }
```

This is only for local testing, not production setup.

## Project Organization

The codebase is being gradually refactored from complete demo-specific folders toward reusable components. Composed model prototypes now live under `src/models/`, while root-level paths are re-exported from `lib.rs` so existing demos and tests keep working.

```text
fixed_point/       fixed-point encoding, arithmetic, range checks, and gadgets
padding/           vector/matrix padding and zero-padding constraints
activations/       canonical ReLU and clipped-ReLU lookup components
layers/            reusable affine, MLP, and single-channel Conv2d helpers
public_state/      public state layout and offset helpers
runners/           reusable Nova setup/prove/verify/compress helpers
models/            composed model prototypes built from components
circuits/          future StepCircuit implementations
bin/               executable demos
```

`models/` contains prototypes such as affine recurrence, affine+ReLU, integer MLP, fixed-point MLP, fixed-point denoise, denoise with time embedding, padded denoise, and the fixed-point Conv2d denoise backend. New code should prefer canonical components under `activations/`, `layers/`, `fixed_point/`, `padding/`, `public_state/`, and `runners/`, and place composed model-level prototypes under `models/`.

Compatibility paths are intentionally preserved. For example, both of these are valid:

```rust
denoise::models::mlp_fixed_point_clipped_relu_lookup
denoise::mlp_fixed_point_clipped_relu_lookup
```

Similarly, activation shims keep `denoise::relu` and `denoise::clipped_relu` available while the canonical implementations live in `denoise::activations`.

## Parameter / Model Commitment

The crate includes prototypes for binding public model parameters to a public commitment. They use a toy algebraic hash:

```text
h_0 = init
h_{i+1} = h_i * BASE + v_i
```

with `BASE = 131`. This toy hash is not cryptographically binding. It is only used to demonstrate the model-commitment plumbing and can be replaced later by Poseidon, a Merkle commitment, or another production commitment.

The older fixed-point MLP commitment demo proves:

```text
Given public x0, y, params, and C,
prove y = F_params(x0) and C = ToyHash(params).
```

Parameters are still public in this prototype. That demo recomputes the commitment inside the step circuit for simplicity, so it is mainly a small plumbing check.

Run:

```bash
cargo run --release --bin mlp_fixed_point_commitment_demo
```

The denoise MLP and denoise Conv commitment demos use a hash accumulator, so they support `num_steps > 1` while the public parameter queue shifts:

```text
z = [x | y | h | C | t | params_queue | time_table]
```

Here `h` is the hash of all parameter blocks consumed so far and `C` is the expected final model commitment. At each true denoise iteration the circuit hashes the current parameter block into `h`. At the final timestep it conditionally enforces:

```text
h_T = C
x_T = y
```

Run:

```bash
cargo run --release --bin denoise_fixed_point_mlp_commitment_demo
cargo run --release --bin denoise_fixed_point_conv_commitment_demo
```

A future private-parameter version can move params into witness values and keep only the commitment public:

```text
Given public x0, y, C,
prove exists params such that C = Hash(params) and y = F_params(x0).
```

## Complexity Optimizations

Signed numeric range checks now support offset-binary bit decomposition:

```text
x in [min, max]
u = x - min
u = sum_i 2^i b_i,  b_i in {0,1}
```

This reduces large signed range checks from one-hot cost `O(max-min+1)` to `O(log(max-min+1))`. The bit gadget currently supports power-of-two-width ranges such as `[-128,127]` and `[-256,255]`. Small remainder checks, for example `r in [0,S)` with `S=16`, still use the simpler one-hot lookup.

Denoise updates also support two fixed-point modes:

```text
DoubleFloor:
x_next[j] = floor(alpha*x[j]/S) + floor(beta*epsilon[j]/S)

FusedFloor:
x_next[j] = floor((alpha*x[j] + beta*epsilon[j]) / S)
```

`FusedFloor` uses one rescale per coordinate instead of two, which reduces quotient/remainder constraints in the update layer. It is a different fixed-point semantics from `DoubleFloor`, not an algebraically equivalent rewrite, so both modes are kept for experiments.

Run the fused Conv commitment demo with:

```bash
cargo run --release --bin denoise_fixed_point_conv_fused_demo
```

## Visual Demo

The repository includes a small static visualization for the Nova denoise proof. The Rust exporter runs a 4x4 fixed-point Conv denoise example, records the denoise trajectory, proof summary, Nova chunking, and simple complexity comparisons, then writes JSON for the frontend.

Export the JSON:

```bash
bash scripts/export_visual_demo.sh
```

Run the viewer:

```bash
cd visual_demo
npm install
npm run dev
```

The app reads `visual_demo/public/denoise_demo.json` and shows:

```text
x_t, epsilon_t, and x_{t+1} heatmaps
Nova recursive step timeline
proof constraints, variables, timings, and proof size
DoubleFloor vs FusedFloor and dense-vs-sparse Conv comparisons
```

The frontend is static: it does not call Rust or generate proofs. Re-run `scripts/export_visual_demo.sh` whenever you want to refresh the proof data.

## Denoise Experiments

The main paper-facing experiment entry point can run either the whole suite or one
experiment group at a time:

```bash
cargo run --release --bin denoise_experiments -- all
cargo run --release --bin denoise_experiments -- correctness
cargo run --release --bin denoise_experiments -- range
cargo run --release --bin denoise_experiments -- fused_update
cargo run --release --bin denoise_experiments -- mlp_scaling
cargo run --release --bin denoise_experiments -- conv_scaling
cargo run --release --bin denoise_experiments -- backend_compare
cargo run --release --bin denoise_experiments -- steps_scaling
cargo run --release --bin denoise_experiments -- large_scale
```

Each group writes independent records:

```text
outputs/experiments/<group>/denoise_experiments_<group>.md
outputs/experiments/<group>/denoise_experiments_<group>.json
outputs/experiments/<group>/denoise_experiments_<group>.csv
```

Running `all` additionally writes the combined report:

```text
outputs/experiments/all/denoise_experiments_all.md
outputs/experiments/all/denoise_experiments_all.json
outputs/experiments/all/denoise_experiments_all.csv
```

The default suite excludes affine and sub-16-dimensional toy cases. It focuses on denoise MLP/Conv experiments with `N >= 16`:

```text
end-to-end denoise proof checks
OneHot vs Bits signed range checks
DoubleFloor vs FusedFloor update mode
MLP scaling
Conv scaling
MLP vs Conv comparison
recursive step scaling
large dimension cases, including Conv N=256 and N=1024
```

Large-dimension tables focus on the Conv backend. Dense MLP is kept for small and medium baseline comparisons, while Conv carries the large-scale cases such as `N=256` and `N=1024`. BuildOnly rows synthesize the circuit shape and report constraints/variables without running the full proof.

## 2. Demos in `src/bin`

### `affine_demo.rs`

Run:

```bash
cargo run --release --bin affine_demo
```

Purpose:

Proves a fixed square affine recurrence:

```text
x_{i+1} = A x_i + b
```

Here `A` and `b` are fixed across iterations. This is the simplest Nova IVC baseline.

Main modules used:

```text
src/affine/
src/nova_ivc.rs
```

---

### `affine_time_varying_demo.rs`

Run:

```bash
cargo run --release --bin affine_time_varying_demo
```

Purpose:

Proves a time-varying affine recurrence:

```text
x_{i+1} = A_i x_i + b_i
```

Each iteration may use different parameters. Depending on the exact version of the file, parameters may be modeled as witness values or as public-state values. For zkML semantics, the preferred public-parameter version places the parameter sequence in the public initial state:

```text
z0 = [x0 | params_0 | params_1 | ... | params_{T-1}]
```

---

### `affine_relu_lookup_demo.rs`

Run:

```bash
cargo run --release --bin affine_relu_lookup_demo
```

Purpose:

Proves an affine layer followed by lookup-table ReLU:

```text
x_{i+1} = ReLU(A x_i + b)
```

The ReLU is implemented by proving that `(input, output)` appears in a finite lookup table:

```text
(input, output) in { (v, max(0, v)) }
```

Important:

The affine output must lie inside the table domain. If the demo panics with a message like:

```text
affine output ... is outside ReLU table range
```

increase the table range or reduce the affine parameters.

Main modules used:

```text
src/relu/
src/affine_relu_lookup/
```

---

### `affine_clipped_relu_lookup_demo.rs`

Run:

```bash
cargo run --release --bin affine_clipped_relu_lookup_demo
```

Purpose:

Proves an affine layer followed by clipped ReLU:

```text
x_{i+1} = ClippedReLU(A x_i + b)
```

where:

```text
ClippedReLU(x) = min(max(0, x), clip_max)
```

This is useful because activations remain bounded. The lookup table checks:

```text
(input, output) in { (v, min(max(0, v), clip_max)) }
```

Main modules used:

```text
src/clipped_relu/
src/affine_clipped_relu_lookup/
```

---

### `mlp_clipped_relu_lookup_demo.rs`

Run:

```bash
cargo run --release --bin mlp_clipped_relu_lookup_demo
```

Purpose:

Proves an integer two-layer MLP recurrence with clipped ReLU:

```text
h_i       = ClippedReLU(W1_i x_i + b1_i)
x_{i+1}  = W2_i h_i + b2_i
```

This is the integer baseline. Each iteration has different public parameters:

```text
params_i = [W1_i | b1_i | W2_i | b2_i]
```

The public initial state is:

```text
z0 = [x0 | params_0 | params_1 | ... | params_{T-1}]
```

Each Nova recursive step consumes a fixed number of parameter blocks, shifts the remaining public parameter queue to the left, and pads consumed positions with zero.

Main modules used:

```text
src/mlp_clipped_relu_lookup/
src/clipped_relu/
```

---

### `mlp_fixed_point_clipped_relu_lookup_demo.rs`

Run:

```bash
cargo run --release --bin mlp_fixed_point_clipped_relu_lookup_demo
```

Purpose:

Proves a fixed-point two-layer MLP recurrence with clipped ReLU:

```text
h_i      = ClippedReLU(floor((W1_i x_i) / S) + b1_i)
x_{i+1} = floor((W2_i h_i) / S) + b2_i
```

Default scale:

```text
S = 16
```

External float-to-fixed encoding uses rounding:

```text
encode(x) = round(x * S)
```

Circuit-internal rescaling uses mathematical floor:

```text
floor_div(-7, 4) = -2
```

The floor relation is checked as:

```text
z = q * S + r
0 <= r < S
```

The range condition on `r` is checked using a lookup/range gadget. The quotient is also constrained to a configured signed integer range so the field equation has a bounded integer interpretation:

```text
q in [quotient_min, quotient_max]
```

Fixed-point state-like values can also be checked against:

```text
value in [value_min, value_max]
```

Parameters are public and may differ at every iteration. They are stored in the public state queue exactly like the integer MLP version.

Main modules used:

```text
src/fixed_point/
src/mlp_fixed_point_clipped_relu_lookup/
src/clipped_relu/
```

---

### `denoise_fixed_point_demo.rs`

Run:

```bash
cargo run --release --bin denoise_fixed_point_demo
```

Purpose:

Proves a toy fixed-point denoise step. Each iteration first predicts a residual/noise term with the existing fixed-point MLP:

```text
epsilon_t = MLP_t(x_t)
```

Then it applies a public fixed-point schedule:

```text
x_{t+1} = floor(alpha_t x_t / S) + floor(beta_t epsilon_t / S)
```

The public initial state stores all per-iteration parameters:

```text
z0 = [x0 | denoise_params_0 | ... | denoise_params_{T-1}]
denoise_params_t = [W1_t | b1_t | W2_t | b2_t | alpha_t | beta_t]
```

Compared with the fixed-point MLP baseline, which proves only `x_{t+1}=MLP_t(x_t)`, the denoise version proves the scheduled update `x_{t+1}=alpha_t*x_t+beta_t*MLP_t(x_t)` with fixed-point floor rescaling.

Current statement note:

The demo proves that, from public `z0` and public parameters, there exists a valid recursive denoise trajectory. It prints the derived final `x_T`, but it does not yet take an externally supplied expected `x_T` as an additional public output. To prove a statement of the form `(x0, xT, params)`, extend the public state with `expected_xT` or compare against a final state returned by the verifier if the Nova API exposes one.

Main modules used:

```text
src/denoise_fixed_point/
src/mlp_fixed_point_clipped_relu_lookup/
src/fixed_point/
src/clipped_relu/
```

---

### `denoise_fixed_point_time_embedding_demo.rs`

Run:

```bash
cargo run --release --bin denoise_fixed_point_time_embedding_demo
```

Purpose:

Proves a toy fixed-point denoise step with an explicit, provable timestep embedding:

```text
e_t = table[t]
epsilon_t = MLP_t([x_t || e_t])
x_{t+1} = floor(alpha_t x_t / S) + floor(beta_t epsilon_t / S)
```

The embedding table is public and stored once in the public state. The circuit proves the selected row with a one-hot lookup over the public timestep `t`, then increments `t` each local iteration. Unlike `denoise_fixed_point`, this version explicitly proves that the MLP input includes the correct timestep embedding.

Public state layout:

```text
z0 = [x0 | t0 | denoise_params_0 | ... | denoise_params_{T-1} | E_0 | ... | E_{T-1}]
```

Each recursive step shifts only the parameter queue; the embedding table is copied unchanged.

Main modules used:

```text
src/denoise_fixed_point_time_embedding/
src/fixed_point/
src/clipped_relu/
```

---

### `denoise_fixed_point_time_embedding_padded_demo.rs`

Run:

```bash
cargo run --release --bin denoise_fixed_point_time_embedding_padded_demo
```

Purpose:

Embeds a smaller denoise time-embedding model into a fixed maximum-shape Nova circuit using zero padding. This keeps the `StepCircuit` shape fixed while allowing experiments such as:

```text
real dims: N=2, TE=2, H=3
max dims:  N=4, TE=4, H=4
```

Padding rules:

```text
x, time embedding, W1, b1, W2, b2 are padded with zeros
```

The circuit enforces zero padding for the current state, selected time embedding, current parameter block padding, hidden padding, epsilon padding, and output padding. This implements dimension padding only; it does not implement variable-length time masking or early stopping.

Recommended comparison:

```text
unpadded real output == first N_REAL entries of padded output
```

Main modules used:

```text
src/padding/
src/denoise_fixed_point_time_embedding_padded/
src/denoise_fixed_point_time_embedding/
```

---

### `denoise_fixed_point_time_embedding_padded_output_demo.rs`

Run:

```bash
cargo run --release --bin denoise_fixed_point_time_embedding_padded_output_demo
```

Purpose:

Adds the simplest public output binding to the padded time-embedding denoise proof. The public state layout is:

```text
z = [x | y | t | params_queue | time_table]
```

where `y` is the expected padded final output. The proof carries `y` unchanged through every recursive step:

```text
z_out = [x_final | y | t + num_iters_per_step | shifted_params_queue | same_time_table]
```

The demo computes `y` from the trace, places it in `z0`, and checks:

```text
y == final witness x_T
```

This is the minimal state-level binding. A future stricter version can enforce `x_T = y` inside the last recursive step using a final-step mask or counter.

---

### `denoise_fixed_point_conv_demo.rs`

Run:

```bash
cargo run --release --bin denoise_fixed_point_conv_demo
```

Purpose:

Uses a single-channel fixed-point Conv2d backend as the denoise epsilon predictor:

```text
e_t = table[t]
time_bias_t = floor(w_time_t * e_t / S) + b_time_t
epsilon_t = ClippedReLU(Conv2d_t(x_t) + time_bias_t)
x_{t+1} = floor(alpha_t x_t / S) + floor(beta_t epsilon_t / S)
```

The state vector is interpreted as a row-major single-channel image. The demo uses same-style convolution padding for a 3x3 kernel, so the conv output shape equals the input shape and can be flattened back into the denoise update.

There are two distinct padding concepts:

```text
Conv2dPadding:
  mathematical convolution padding; out-of-bound input coordinates are constants 0.

Conv2dRealShape:
  dimension-padding metadata; real image/kernel/output rectangles are embedded in
  a maximum circuit shape, and padded public/witness coordinates must be zero.
```

For denoise updates the current Conv backend requires:

```text
OH == IH
OW == IW
```

because `epsilon_t` is flattened and updated coordinate-wise against `x_t`.

Public output binding is available through:

```bash
cargo run --release --bin denoise_fixed_point_conv_output_demo
```

That demo uses:

```text
z = [x | y | t | params_queue | time_table]
```

where `y` is the expected public final output carried unchanged through each recursive step. The demo sets `y` to the computed final trace output and checks it against the final witness. A future stricter version can add an in-circuit final-step equality mask for `x_T == y`.

Current Conv2d scope:

- single input channel and single output channel,
- stride 1 only,
- convolution padding is fixed by the circuit config and uses zero for out-of-bounds input coordinates,
- kernel/time-weight values are public parameters from the parameter queue,
- dimension padding uses rectangular spatial metadata, not a flat prefix,
- trace generation performs preflight range checks for state, quotient, preactivation, activation, and update witnesses,
- numerator/raw range checks are still future hardening work for stricter field-wraparound soundness.

Main modules used:

```text
src/layers/conv2d/
src/models/denoise_fixed_point_conv/
src/models/denoise_fixed_point_time_embedding/
src/fixed_point/
src/activations/clipped_relu/
```

## 3. Folder guide

### `src/affine/`

Reusable affine baseline code.

Typical semantics:

```text
y = A x + b
```

For the square recurrence demo, input and output dimensions match:

```text
x_{i+1} = A x_i + b
```

Files usually include:

```text
params.rs   parameter structs
util.rs     plain Rust affine computation
trace.rs    witness trace generation
circuit.rs  Nova StepCircuit constraints
mod.rs      module exports
```

---

### `src/relu/`

Lookup-table plain ReLU.

Semantics:

```text
ReLU(x) = max(0, x)
```

This module is mainly useful as a baseline before clipped ReLU.

Typical files:

```text
table.rs    table construction and integer semantics
gadget.rs   R1CS lookup-style gadget
mod.rs      exports
```

---

### `src/clipped_relu/`

Lookup-table clipped ReLU.

Semantics:

```text
ClippedReLU(x) = min(max(0, x), clip_max)
```

This is preferred over unbounded ReLU for toy zkML prototypes because it controls activation growth.

Typical files:

```text
table.rs         integer table semantics
gadget.rs        circuit lookup gadget
fixed_point.rs   fixed-point helper wrappers if present
mod.rs           exports
```

---

### `src/affine_relu_lookup/`

Affine plus ReLU prototype:

```text
x_{i+1} = ReLU(A x_i + b)
```

Useful for testing ordinary lookup ReLU after affine computation.

---

### `src/affine_clipped_relu_lookup/`

Affine plus clipped ReLU prototype:

```text
x_{i+1} = ClippedReLU(A x_i + b)
```

This is a better bounded activation baseline than ordinary ReLU.

---

### `src/mlp_clipped_relu_lookup/`

Integer two-layer MLP baseline.

Semantics:

```text
h_i       = ClippedReLU(W1_i x_i + b1_i)
x_{i+1}  = W2_i h_i + b2_i
```

Important properties:

- parameters are public,
- parameters may differ at each iteration,
- the parameter queue is stored in the public state,
- hidden affine values and activations are private witness values,
- clipped ReLU is checked by lookup.

Typical files:

```text
params.rs   public parameter layout and flattening
trace.rs    integer execution trace
circuit.rs  Nova StepCircuit
runner.rs   setup/prove/verify/compress helpers
mod.rs      exports
```

---

### `src/fixed_point/`

Reusable fixed-point arithmetic utilities.

This folder is intended to be shared by future affine, MLP, and denoise modules.

Current semantics:

```text
scale = S
encode(x) = round(x * S)
rescale(z) = floor(z / S)
```

The affine fixed-point rule is:

```text
y = floor((W * x) / S) + b
```

For floor division, this project uses mathematical floor, not Rust truncation:

```text
floor_div(-7, 4) = -2
```

The circuit checks:

```text
z = q * S + r
0 <= r < S
q in [quotient_min, quotient_max]
```

where the remainder range is enforced with a lookup/range check.

Typical files:

```text
config.rs   scale and range configuration
encode.rs   f64 <-> fixed-point integer conversion
arith.rs    plain Rust fixed-point arithmetic
gadget.rs   R1CS fixed-point rescale/range gadget
mod.rs      exports
```

---

### `src/mlp_fixed_point_clipped_relu_lookup/`

Fixed-point two-layer MLP with clipped ReLU.

Semantics:

```text
h_i      = ClippedReLU(floor((W1_i x_i) / S) + b1_i)
x_{i+1} = floor((W2_i h_i) / S) + b2_i
```

This is the main next-step zkML prototype after the integer baseline.

Important properties:

- parameters are public,
- each iteration may use different parameters,
- parameters are fixed-point integers,
- external encoding uses round,
- circuit rescale uses mathematical floor,
- remainder range check is enforced,
- clipped ReLU upper bound scales with `scale`.

Typical files:

```text
params.rs   fixed-point public parameter structs and flattening
trace.rs    fixed-point execution trace
circuit.rs  Nova StepCircuit constraints
runner.rs   setup/prove/verify/compress helpers
mod.rs      exports
```

---

### `src/denoise_fixed_point/`

Toy fixed-point denoise IVC.

Semantics:

```text
epsilon_t = MLP_t(x_t)
x_{t+1} = floor(alpha_t x_t / S) + floor(beta_t epsilon_t / S)
```

The MLP part reuses the fixed-point clipped-ReLU MLP implementation. The denoise schedule parameters `alpha_t` and `beta_t` are public fixed-point integers and may differ at every iteration.

Public state layout:

```text
z0 = [x0 | denoise_params_0 | ... | denoise_params_{T-1}]
denoise_params_t = [W1_t | b1_t | W2_t | b2_t | alpha_t | beta_t]
```

Typical files:

```text
params.rs   denoise parameter structs and flattening
trace.rs    fixed-point denoise witness trace
circuit.rs  Nova StepCircuit constraints
runner.rs   setup/prove/verify/compress helpers
mod.rs      exports
```

---

### `src/denoise_fixed_point_time_embedding/`

Fixed-point denoise IVC with explicit lookup-table timestep embedding.

Semantics:

```text
e_t = table[t]
epsilon_t = MLP_t([x_t || e_t])
x_{t+1} = floor(alpha_t x_t / S) + floor(beta_t epsilon_t / S)
```

The MLP input dimension is `N + TE`, while the output dimension remains `N`. The embedding table is public state, and the circuit uses a one-hot selector to prove the selected embedding row matches the current public timestep.

Typical files:

```text
params.rs          parameter structs and flattening
time_embedding.rs  time table generation and lookup gadget
trace.rs           fixed-point execution trace
circuit.rs         Nova StepCircuit constraints
runner.rs          setup/prove/verify/compress helpers
mod.rs             exports
```

---

### `src/padding/`

Reusable helpers for embedding smaller models into larger fixed-shape circuits.

Semantics:

```text
real vectors/matrices are copied into their real region
all padding entries are zero
```

For time-embedding MLP inputs, the padded layout remains:

```text
[x padded to N_MAX || time embedding padded to TE_MAX]
```

So W1 padding preserves real `x` weights in the first `N_REAL` columns and real embedding weights in columns `N_MAX..N_MAX+TE_REAL`.

Typical files:

```text
vector.rs  vector padding and slicing
matrix.rs  rectangular matrix padding
gadget.rs  circuit zero-padding constraints
mod.rs     exports
```

### `src/denoise_fixed_point_time_embedding_padded/`

Padded facade for the denoise time-embedding circuit. It supplies runtime real-dimension metadata while the actual circuit arrays use maximum dimensions.

---

### `src/nova_ivc.rs`

Shared Nova engine aliases and helper types.

This usually contains aliases such as:

```text
E1, E2, F1, G1, S1, S2
```

and may also contain helper functions for setup/prove/verify in smaller demos.

---

### `src/bin/`

Runnable examples. Each file here corresponds to a command:

```bash
cargo run --release --bin <file_name_without_rs>
```

For example:

```bash
cargo run --release --bin mlp_fixed_point_clipped_relu_lookup_demo
```

---

### `tests/`

Unit and integration tests.

Typical tests cover:

- ReLU table semantics,
- clipped ReLU table semantics,
- fixed-point encode/decode,
- mathematical floor division,
- fixed-point rescale,
- range-check conditions,
- MLP trace correctness,
- public parameter flattening layout,
- optional Nova end-to-end proof verification.

Heavy proof tests are usually marked:

```rust
#[ignore]
```

Run them explicitly with:

```bash
cargo test --release <test_name> -- --ignored --nocapture
```

## 4. Recommended workflow

### Step 1: verify the integer baseline

```bash
cargo run --release --bin mlp_clipped_relu_lookup_demo
cargo test --release mlp_clipped_relu_lookup_tests
```

### Step 2: verify fixed-point utilities

```bash
cargo test --release fixed_point_mlp_clipped_relu_tests
```

### Step 3: run the fixed-point demo

```bash
cargo run --release --bin mlp_fixed_point_clipped_relu_lookup_demo
```

### Step 4: run heavy Nova proof tests only when needed

```bash
cargo test --release -- --ignored --nocapture
```

## 5. Notes on statement semantics

The intended zkML statement for the public-parameter MLP demos is:

```text
Given public x0 and public parameter sequence params_0 ... params_{T-1},
prove that the private intermediate trace follows the specified model recurrence.
```

For the integer MLP:

```text
h_i       = ClippedReLU(W1_i x_i + b1_i)
x_{i+1}  = W2_i h_i + b2_i
```

For the fixed-point MLP:

```text
h_i      = ClippedReLU(floor((W1_i x_i) / S) + b1_i)
x_{i+1} = floor((W2_i h_i) / S) + b2_i
```

This differs from an existential parameter proof. The parameters are public and fixed by the initial public state.

## 6. Common issues

### Module not found

If you see:

```text
could not find `xxx` in `denoise`
```

check that `src/lib.rs` exports the module:

```rust
pub mod xxx;
```

### Lookup range panic

If you see:

```text
outside clipped ReLU table range
```

then an activation input is outside the lookup table domain. Increase the table range, reduce parameters, reduce input magnitude, or increase clipping/range configuration consistently.

### HyperKZG setup error

If you see:

```text
HyperKZG::setup is disabled in production builds
```

for local experiments, enable Nova `test-utils` in `Cargo.toml`. For production-like experiments, use ptau files and `setup_with_ptau_dir`.

### Fixed-point negative division mismatch

Rust integer division truncates toward zero. This project uses mathematical floor division. Therefore:

```text
-7 / 4 in Rust = -1
floor_div(-7, 4) = -2
```

Use the helper functions in `src/fixed_point/arith.rs` instead of raw `/` when implementing fixed-point rescaling.

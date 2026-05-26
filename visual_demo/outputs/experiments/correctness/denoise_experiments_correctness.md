# Denoise zkML Experiments

## Environment

- rust: rustc 1.96.0-nightly (02c7f9bec 2026-04-10)
- scale: 16
- curve/backend: Nova local test setup
- official cases: N >= 16

## Experiment 1: End-to-end correctness

| case | backend | N | structure | iters | mode | range | run | constraints | variables | recursive ms | compressed ms | verify ms | proof bytes | status |
|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| correctness_mlp_16 | MLP | 16 | H=16 | 4 | DoubleFloor | Bits | FullProof | 345164 | 349680 | 1049.056 | 2636.374 | 117.301 | 13088 | OK |
| correctness_mlp_32 | MLP | 32 | H=32 | 4 | DoubleFloor | Bits | FullProof | 1218848 | 1236036 | 3361.206 | 7938.325 | 344.030 | 14389 | OK |
| correctness_conv_4x4 | Conv | 16 | 4x4, K=3x3 | 4 | DoubleFloor | Bits | FullProof | 29100 | 28932 | 170.425 | 924.699 | 44.893 | 11317 | OK |
| correctness_conv_8x8 | Conv | 64 | 8x8, K=3x3 | 4 | DoubleFloor | Bits | FullProof | 59556 | 58524 | 186.237 | 1021.260 | 34.907 | 11692 | OK |

## Notes

- No affine experiments are included.
- All official cases use N >= 16.
- FusedFloor has different fixed-point semantics from DoubleFloor.
- BuildOnly cases are used for large-scale circuit-size analysis.
- Commitment modules remain in the codebase but are not reported as a separate overhead experiment here.

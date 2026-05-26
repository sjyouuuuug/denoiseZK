# Denoise zkML Experiments

## Environment

- rust: rustc 1.96.0-nightly (02c7f9bec 2026-04-10)
- scale: 16
- curve/backend: Nova local test setup
- official cases: N >= 16

## Experiment 2: Range check optimization

| case | backend | N | structure | iters | mode | range | run | constraints | variables | recursive ms | compressed ms | verify ms | proof bytes | status |
|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| range_conv_8x8_onehot | Conv | 64 | 8x8, K=3x3 | 2 | DoubleFloor | OneHot | FullProof | 101168 | 99682 | 84.642 | 1617.378 | 106.591 | 11797 | OK |
| range_conv_8x8_bits | Conv | 64 | 8x8, K=3x3 | 2 | DoubleFloor | Bits | FullProof | 54848 | 53748 | 77.225 | 1324.852 | 96.680 | 11429 | OK |

## Notes

- No affine experiments are included.
- All official cases use N >= 16.
- FusedFloor has different fixed-point semantics from DoubleFloor.
- BuildOnly cases are used for large-scale circuit-size analysis.
- Commitment modules remain in the codebase but are not reported as a separate overhead experiment here.

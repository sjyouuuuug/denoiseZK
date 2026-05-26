# Fixed-Point MLP Clipped-ReLU Lookup Experiments

Command:

```bash
cargo run --release --bin mlp_fixed_point_experiments
```

Environment:

- Date: 2026-05-15
- Fixed-point scale: 16
- Successful proof cases use lookup range `[-64, 64]` and `clip_max = 32`.
- The overflow case intentionally uses lookup range `[-16, 16]` to verify detection.

## Results

| case | N | H | steps | iters/step | total iters | status | primary constraints | secondary constraints | primary variables | secondary variables | proof size bytes | recursive prove ms | compressed prove ms | compressed verify ms | lookup range overflow |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| tiny_1x1_t1 | 1 | 1 | 1 | 1 | 1 | OK | 11736 | 10550 | 11732 | 10532 | 10614 | 33.101 | 1264.117 | 95.883 | no |
| small_2x2_t2 | 2 | 2 | 1 | 2 | 2 | OK | 13875 | 10550 | 13901 | 10532 | 10657 | 36.221 | 1330.490 | 101.524 | no |
| demo_2x3_t6 | 2 | 3 | 3 | 2 | 6 | OK | 25161 | 10550 | 25337 | 10532 | 11275 | 266.759 | 1363.365 | 101.588 | no |
| wide_3x3_t4 | 3 | 3 | 2 | 2 | 4 | OK | 23680 | 10550 | 23844 | 10532 | 11279 | 128.172 | 1204.284 | 67.755 | no |
| wide_4x4_t2 | 4 | 4 | 1 | 2 | 2 | OK | 22543 | 10550 | 22669 | 10532 | 11039 | 37.664 | 1373.114 | 77.395 | no |
| overflow_narrow_range | 2 | 2 | 1 | 1 | 1 | OVERFLOW | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | yes: step=0, hidden=0, value=40, range=[-16,16] |

## Notes

- No successful proof case panicked.
- The overflow case is skipped before proof generation because preflight trace simulation detects a hidden affine value outside the clipped-ReLU lookup table.
- `recursive prove ms` measures recursive SNARK initialization plus all recursive prove steps.
- `compressed prove ms`, `compressed verify ms`, and `proof size bytes` refer to the compressed SNARK.

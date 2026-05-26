#!/usr/bin/env bash
set -euo pipefail

cargo run --release --bin visual_denoise_export
mkdir -p visual_demo/public
cp outputs/visual_demo/denoise_demo.json visual_demo/public/denoise_demo.json
echo "visual_demo/public/denoise_demo.json updated"

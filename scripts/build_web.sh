#!/usr/bin/env bash
# Build the browser engine (bace.wasm) into web/dist/engine/.
set -euo pipefail
cd "$(dirname "$0")/.."
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
cargo build -p bace-wasm --target wasm32-unknown-unknown --profile wasm
cp target/wasm32-unknown-unknown/wasm/bace_wasm.wasm web/dist/engine/bace.wasm
echo "web/dist/engine/bace.wasm ($(du -h web/dist/engine/bace.wasm | cut -f1))"

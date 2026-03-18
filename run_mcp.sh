#!/usr/bin/env bash
# Wrapper for cuba-thinking MCP server
# Sets LD_LIBRARY_PATH for libtorch (rust-bert dependency)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TORCH_LIB_DIR="$(find "${SCRIPT_DIR}/cuba_cognitive_engine/target/release/build" -path "*/torch-sys-*/out/libtorch/libtorch/lib" -type d 2>/dev/null | head -1)"

if [ -z "$TORCH_LIB_DIR" ]; then
    echo "ERROR: libtorch lib directory not found. Run: cargo build --release" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

exec "${SCRIPT_DIR}/cuba_cognitive_engine/target/release/cuba_cognitive_engine" "$@"

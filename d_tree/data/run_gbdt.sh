#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/llc_pin.sh"

b="64"
d="3"
t="80"
k="10"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN="${BIN:-$SCRIPT_DIR/../../56bit/d_tree}"

if [ ! -x "$BIN/gbdt_main" ]; then
    echo "error: $BIN/gbdt_main not found." >&2
    echo "       Run TORUS=56 bash build_d_tree.sh from the repository root." >&2
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/gbdt" ]; then
    echo "error: Directory 'gbdt/' not found." >&2
    echo "       Unpack them from d_tree/data:" >&2
    echo "       cat gbdt.tar.zst.part* | tar --zstd -xf -" >&2
    exit 1
fi

"$BIN"/gbdt_main gbdt/steel_gbdt_d"$d"_t"$t"_q"$b"_k"$k"_s0.json --json-output out/steel_gbdt_d"$d"_t"$t"_q"$b"_k"$k"_o_g.json

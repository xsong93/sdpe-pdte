#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/llc_pin.sh"

b="64"
DEPTHS=(1 2 3 4)
DATASETS=(cancer heart spambase steel)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN="${BIN:-$SCRIPT_DIR/../../32bit/d_tree}"

for exec in d_tree_ours_main d_tree_te0_main; do
    if [ ! -x "$BIN/$exec" ]; then
        echo "error: $BIN/$exec not found." >&2
        echo "       Run TORUS=32 bash build_d_tree.sh from the repository root." >&2
        exit 1
    fi
done

if [ ! -d "$SCRIPT_DIR/trees" ]; then
    echo "error: Directory 'trees/' not found." >&2
    echo "       Unpack them from d_tree/data:" >&2
    echo "       tar -xf trees.tar.zst" >&2
    exit 1
fi

# Set up warm up cycle.
DT_WARMUP="${DT_WARMUP:-10}"

declare -A OURS_PARAMS=(
    [1]="16 760 2048 10 3 2 16  3"
    [2]="16 760 2048 10 3 2 16  3"
    [3]="16 760 2048 10 3 2 16  3"
    [4]="32 840 2048  7 4 3 14  3"
)
declare -A TE0_PARAMS=(
    [1]="16 760 2048 10 3 2 16  3"
    [2]="16 760 2048 10 3 2 16  3"
    [3]="16 760 2048 10 3 2 16  3"
    [4]="16 760 2048  7 4 3 16  3"
)

# "p n N B l la lweNoise rlweNoise" -> the binaries' override flags.
param_args() {
    local p n N radix l la noise rnoise
    read -r p n N radix l la noise rnoise <<<"$1"
    printf -- "--torus-base %s --n %s --N %s --radix-bits %s --l %s --l-approx %s --lwe-noise-b %s --rlwe-noise-b %s" \
        "$p" "$n" "$N" "$radix" "$l" "$la" "$noise" "$rnoise"
}

mkdir -p out

rm -f out/*_q"$b"_ours_o.json out/*_q"$b"_te0_o.json

for d in "${DEPTHS[@]}"; do
    trees=()
    for ds in "${DATASETS[@]}"; do
        trees+=(trees/"$ds"_d"$d"_t1_q"$b".json)
    done

    read -r -a ours_args <<<"$(param_args "${OURS_PARAMS[$d]}")"
    read -r -a te0_args  <<<"$(param_args "${TE0_PARAMS[$d]}")"

    echo "=== depth $d | OURS ${OURS_PARAMS[$d]} | warmup $DT_WARMUP ==="
    "$BIN"/d_tree_ours_main "${trees[@]}" "${ours_args[@]}" \
        --warmup "$DT_WARMUP" \
        --json-output out/d"$d"_q"$b"_ours_o.json

    echo "=== depth $d | TE0  ${TE0_PARAMS[$d]} | warmup $DT_WARMUP ==="
    "$BIN"/d_tree_te0_main "${trees[@]}" "${te0_args[@]}" \
        --warmup "$DT_WARMUP" \
        --json-output out/d"$d"_q"$b"_te0_o.json
done

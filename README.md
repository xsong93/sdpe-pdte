# d_tree

This repository is the code for the paper "Signed Dyadic Path Encoding for Efficient Non-Interactive Private Decision Tree Evaluation over TFHE", submitted to IEEE Transactions on Emerging Topics in Computing and currently under review.

Homomorphic decision tree evaluation on top of `yatfhe`.

## PBS = leaf-parents

Each tree level carries a signed dyadic label, path sums are accumulated per leaf-parent, and both children of a leaf-parent 
are packed into one test polynomial. A single PBS on that sum then both identifies the active child and returns its class label, 
while every leaf-parent off the active path extracts zero.

    PBS per inference = |P| leaf-parents
    perfect binary tree:  |L| = 2|P|,  and  |P| = 2^(d-1) at depth d

## Dependencies

| Dependency | Used by | Notes |
|---|---|---|
| CMake ≥ 3.10, a C++17 compiler | build | |
| [Intel HEXL](https://github.com/IntelLabs/hexl) | `yatfhe_lib` (NTT backend) | Install separately |
| GMP (`gmpxx`) | `yatfhe_lib` | Install separately, see [gmplib.org](https://gmplib.org/) |
| `zstd` | unpacking fixtures | `apt install zstd` |
| `nlohmann/json` | `d_tree` | Vendored|

### YATFHE

YATFHE is a TFHE library, and included here as a git submodule.

```bash
git submodule update --init --recursive
```

If you have it checked out elsewhere, or received it as an archive, point the build at that copy instead. `YATFHE_ROOT` is the directory containing `yatfhe/CMakeLists.txt`:

```bash
YATFHE_ROOT=/path/to/yatfhe bash build_d_tree.sh
```

## Build

`TORUS` selects the torus width. The two benchmarks need different widths:

```bash
TARGET_ARCH=avx512 TORUS=32 bash build_d_tree.sh   # → 32bit/d_tree/ , run_dt.sh
TARGET_ARCH=avx512 TORUS=56 bash build_d_tree.sh   # → 56bit/d_tree/ , run_gbdt.sh
```

### ISA baseline

`TARGET_ARCH=avx512` is the recommended setting.

Use `TARGET_ARCH=native` on a machine without AVX-512, and expect them to take substantially longer. `avx2` and `generic` 
exist for building portable binaries, not for benchmarking. The script's built-in default is `native`.

`build_d_tree.sh` defaults `HEXL_ROOT` to an AVX-512 install for `native`/`avx512`. Set `HEXL_ROOT` to override.

## Unpack the decision trees

Unpack from `d_tree/data/`:

```bash
cd d_tree/data
sha256sum -c SHA256SUMS                      # optional
tar -xf trees.tar.zst                        # → trees/

# GBDT. Expands to ~14 GB.
cat gbdt.tar.zst.part* | tar --zstd -xf -    # → gbdt/
```

`tree_qmax.tar.zst` holds the quantization accuracy experiment data. C++ benchmarks do not read them.

## Run

Run from `d_tree/data/`:

```bash
cd d_tree/data
mkdir -p out

bash run_dt.sh      # 32-bit: OURS vs TE0, cancer/heart/spambase/steel, depths 1-4
                    # → out/d<depth>_q64_ours_o.json, out/d<depth>_q64_te0_o.json

bash run_gbdt.sh    # 56-bit: GBDT on steel
                    # → out/steel_gbdt_d3_t80_q64_k10_o_g.json
```

Each depth runs at the parameter set that is cost-optimal for that scheme at that depth. The set used is recorded in 
each output file's `setup` block.

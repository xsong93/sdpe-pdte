#! /bin/bash
set -e

# 32, 56
TORUS="${TORUS:-32}"

BUILD_DIR="${BUILD_DIR:-${TORUS}bit}"

# ISA baseline for our own code: native (default), avx512, avx2 or generic.
TARGET_ARCH="${TARGET_ARCH:-native}"

# Install prefixes for different HEXL build.
# Set HEXL_ROOT to override.
case "$TARGET_ARCH" in
    native|avx512) DEFAULT_HEXL_ROOT="/usr/local" ;;
    avx2)          DEFAULT_HEXL_ROOT="$HOME/.hexl/avx2" ;;
    *)             DEFAULT_HEXL_ROOT="$HOME/.hexl/generic" ;;
esac
HEXL_ROOT="${HEXL_ROOT-$DEFAULT_HEXL_ROOT}"

if [ ! -d "$HEXL_ROOT" ]; then
    echo "error: HEXL_ROOT=$HEXL_ROOT not found. Run ./hexl_build_all.sh" >&2
    exit 1
fi

# Point to your yatfhe dir.
YATFHE_ROOT="${YATFHE_ROOT:-third_party}"

cached=$(sed -n 's/^TORUS_TYPE:STRING=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null) || true
if [ -n "$cached" ] && [ "$cached" != "$TORUS" ]; then
    echo "error: $BUILD_DIR is configured for TORUS=$cached, not $TORUS." >&2
    echo "       Use BUILD_DIR=${TORUS}bit, or delete $BUILD_DIR to reconfigure it." >&2
    exit 1
fi

cmake -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DINSTALL="OFF" \
      -DTORUS_TYPE="$TORUS" \
      -DLWE_TORUS32="ON" \
      -DPRINTER_ON="OFF" \
      -DENABLE_TIMER="OFF" \
      -DTARGET_ARCH="$TARGET_ARCH" \
      -DCMAKE_PREFIX_PATH="$HEXL_ROOT" \
      -DYATFHE_ROOT="$YATFHE_ROOT"

cmake --build "$BUILD_DIR" -j$(nproc)

#!/bin/bash
#
# Confine the caller to one last-level-cache domain. Source this from a run script before any work:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/llc_pin.sh"
#
# On a single-LLC machine, this is a no-op. It matters on a multi-CCD: when scheduler splits caller and workers
# across dies, each one crosses instead of hitting a shared L3.
#
# Set DT_NO_PIN=1 to skip.

llc_cpu_list() {
    local best_level=-1 best_list="" level type d
    for d in /sys/devices/system/cpu/cpu0/cache/index*; do
        [ -r "$d/level" ] || continue
        [ -r "$d/shared_cpu_list" ] || continue
        type=$(cat "$d/type" 2>/dev/null || echo Unified)
        if [ "$type" = "Instruction" ]; then continue; fi
        level=$(cat "$d/level")
        if [ "$level" -gt "$best_level" ]; then
            best_level="$level"
            best_list=$(cat "$d/shared_cpu_list")
        fi
    done
    printf '%s' "$best_list"
}

if [ -z "${DT_PINNED:-}" ] && [ -z "${DT_NO_PIN:-}" ] && command -v taskset >/dev/null 2>&1; then
    llc_cpus=$(llc_cpu_list)
    if [ -n "$llc_cpus" ]; then
        export DT_PINNED=1
        echo "Confining to llc domain: $llc_cpus"
        exec taskset -c "$llc_cpus" "${BASH:-/bin/bash}" "$0" "$@"
    fi
fi

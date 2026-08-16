cmake_minimum_required(VERSION 3.10)
message(STATUS "Building base...")
project(base)

set(CMAKE_CXX_STANDARD 17)


# Default to Release, override by an explicit -DCMAKE_BUILD_TYPE.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# --- ISA baseline ---
#   native  - native build
#   avx512  - require AVX-512F/DQ/VL/BW
#   avx2    - AVX2 + FMA
#   generic - no ISA flags at all
set(TARGET_ARCH "native" CACHE STRING "ISA baseline: native, avx512, avx2, generic")
set_property(CACHE TARGET_ARCH PROPERTY STRINGS native avx512 avx2 generic)

include(CheckCXXCompilerFlag)

function(add_arch_flags label)
    string(REPLACE ";" " " flags_str "${ARGN}")
    string(MAKE_C_IDENTIFIER "COMPILER_SUPPORTS_ARCH_${label}" cache_var)
    check_cxx_compiler_flag("${flags_str}" ${cache_var})
    if(${cache_var})
        add_compile_options(${ARGN})
    else()
        message(WARNING "TARGET_ARCH=${label}: compiler rejects '${flags_str}'. Building with no ISA flags")
    endif()
endfunction()

if(TARGET_ARCH STREQUAL "native")
    add_arch_flags(native -march=native)
elseif(TARGET_ARCH STREQUAL "avx512")
    add_arch_flags(avx512 -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma)
elseif(TARGET_ARCH STREQUAL "avx2")
    add_arch_flags(avx2 -mavx2 -mfma)
elseif(NOT TARGET_ARCH STREQUAL "generic")
    message(FATAL_ERROR "TARGET_ARCH must be one of: native, avx512, avx2, generic (got '${TARGET_ARCH}')")
endif()

message(STATUS "TARGET_ARCH: ${TARGET_ARCH}")

set(YATFHE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/yatfhe" CACHE PATH "")

if(NOT EXISTS "${YATFHE_ROOT}/yatfhe/CMakeLists.txt")
    message(FATAL_ERROR
        "yatfhe not found under ${YATFHE_ROOT}. Either fetch the submodule:\n"
        "    git submodule update --init --recursive\n"
        "or provide it through:\n"
        "    YATFHE_ROOT=/path/to/yatfhe bash build_d_tree.sh\n")
endif()
message(STATUS "YATFHE_ROOT: ${YATFHE_ROOT}")


add_subdirectory("${YATFHE_ROOT}/yatfhe" "${CMAKE_BINARY_DIR}/yatfhe")
add_subdirectory(d_tree)

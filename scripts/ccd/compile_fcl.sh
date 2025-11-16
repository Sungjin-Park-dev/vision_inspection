#!/bin/bash
#
# Builds the FCL-based CCD collision checker.
#
# Requirements (Ubuntu):
#   sudo apt-get install libeigen3-dev libccd-dev libassimp-dev libyaml-cpp-dev robotpkg-pinocchio
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_fcl"
FCL_DIR="${SCRIPT_DIR}/../../fcl"
FCL_BUILD_DIR="${FCL_DIR}/build"
FCL_LIB="${FCL_BUILD_DIR}/lib/libfcl.so"

mkdir -p "${BUILD_DIR}"

echo "======================================"
echo "FCL CCD Checker Compilation"
echo "======================================"

echo "Checking FCL build..."
if [ ! -f "${FCL_LIB}" ]; then
    echo "FCL not found — building at ${FCL_BUILD_DIR}" >&2
    mkdir -p "${FCL_BUILD_DIR}"
    pushd "${FCL_BUILD_DIR}" >/dev/null
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
    cmake --build . -- -j"$(nproc)"
    popd >/dev/null
else
    echo "FCL already built at ${FCL_BUILD_DIR}"
fi

prefix_path="${FCL_BUILD_DIR}"
if [ -d "/opt/openrobots" ]; then
    prefix_path="${prefix_path};/opt/openrobots"
fi
# Add Isaac Sim's cmeel.prefix for pinocchio
ISAAC_CMEEL_PREFIX="/isaac-sim/kit/python/lib/python3.11/site-packages/cmeel.prefix"
if [ -d "${ISAAC_CMEEL_PREFIX}" ]; then
    prefix_path="${prefix_path};${ISAAC_CMEEL_PREFIX}"
fi
if [ -n "${CMAKE_PREFIX_PATH:-}" ]; then
    prefix_path="${prefix_path};${CMAKE_PREFIX_PATH}"
fi

echo "Configuring CCD checker..."
pushd "${BUILD_DIR}" >/dev/null
cmake "${SCRIPT_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${prefix_path}"

echo "Building fcl_ccd_check..."
cmake --build . -- -j"$(nproc)"
popd >/dev/null

if [ -f "${BUILD_DIR}/fcl_ccd_check" ]; then
    echo ""
    echo "Build successful!"
    echo "Executable: ${BUILD_DIR}/fcl_ccd_check"
else
    echo "Build failed" >&2
    exit 1
fi

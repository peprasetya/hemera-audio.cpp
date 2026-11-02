#!/bin/bash
# Builds the NPU core program and checks the host's instruction generator against the IRON compiler.
# usage: tools/xdna/build.sh <output dir>
# Needs IRON (mlir-aie with the Peano compiler) active in the environment; see docs/build/xdna-npu.md.
# The output is <output dir>/gemm_bfp.xclbin, one core program for every matmul shape.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1")
WORK="$OUT/work"
mkdir -p "$WORK"
cd "$HERE"
GRAIN="16 64 64"
check=()
for shape in "64 1024 4096" "64 128 512" "64 2048 12288" "128 2048 1024" "256 512 2048" "512 4096 4096" "1024 1024 2048" "2048 1024 1024"; do
    set -- $shape
    D="$WORK/$1x$2x$3"
    mkdir -p "$D"
    python3 gemm_bfp.py --dev npu2 -M "$1" -K "$2" -N "$3" -m 16 -k 64 -n 64 --xclbin-path "$D/final.xclbin" --insts-path "$D/insts.bin" > "$D/build.log" 2>&1 || { tail -n 20 "$D/build.log"; exit 1; }
    check+=("$D/insts.bin" "$1" "$2" "$3")
done
cp "$WORK/64x1024x4096/final.xclbin" "$OUT/gemm_bfp.xclbin"
g++ -std=c++17 -O1 -I"$HERE/../../external/ggml/src/ggml-cpu/xdna" check_insts.cpp -o "$WORK/check_insts"
"$WORK/check_insts" "${check[@]}"
echo "built $OUT/gemm_bfp.xclbin"

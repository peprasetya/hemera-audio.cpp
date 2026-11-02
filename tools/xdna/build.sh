#!/bin/bash
# Builds the NPU core program and checks the host's instruction generator against the IRON compiler.
# usage: tools/xdna/build.sh <output dir>
# Needs IRON (mlir-aie with the Peano compiler) active in the environment; see docs/build/xdna-npu.md.
# The output is <output dir>/gemm_bfp.xclbin, one core program for every matmul shape.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1")
WORK="$OUT/work"
rm -rf "$WORK"
mkdir -p "$WORK"
cd "$HERE"
TILE="-m 8 -k 64 -n 64"
check=()
for shape in "32 1024 4096" "32 1024 512" "32 2048 12288" "64 2048 1024" "128 512 2048" "256 4096 4096" "512 1024 2048" "1024 1024 1024"; do
    set -- $shape
    D="$WORK/$1x$2x$3"
    mkdir -p "$D"
    python3 gemm_bfp.py --dev npu2 -M "$1" -K "$2" -N "$3" $TILE --xclbin-path "$D/final.xclbin" --insts-path "$D/insts.bin" > "$D/build.log" 2>&1 || { tail -n 20 "$D/build.log"; exit 1; }
    check+=("$D/insts.bin" "$1" "$2" "$3")
    python3 check_doorbell_word.py "$D/final.prj/input_with_addresses.mlir" "$HERE/../../external/ggml/src/ggml-cpu/xdna/ggml-xdna-insts.h"
done
cp "$WORK/32x1024x4096/final.xclbin" "$OUT/gemm_bfp.xclbin"

g++ -std=c++17 -O1 -I"$HERE/../../external/ggml/src/ggml-cpu/xdna" check_insts.cpp -o "$WORK/check_insts"
"$WORK/check_insts" "${check[@]}"
echo "built $OUT/gemm_bfp.xclbin"

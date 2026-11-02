#!/usr/bin/env python3
"""The doorbell's word must lie past everything the compiler put in its memory tile.
usage: check_doorbell_word.py <input_with_addresses.mlir> <ggml-xdna-insts.h>"""
import re
import sys

mlir, header = open(sys.argv[1]).read(), open(sys.argv[2]).read()
col = int(re.search(r"DB_WORD_COL = (\d+)", header).group(1))
word = int(re.search(r"DB_WORD_ADDR = (\d+)", header).group(1))
ELEMENT = {"f32": 4, "i32": 4, '!aiex.bfp<"v8bfp16ebs8">': 9}
end = 0
for m in re.finditer(r"aie\.buffer\(%mem_tile_" + str(col) + r"_1\) \{address = (\d+) : i32[^}]*\} : memref<([^ ]*)>\s", mlir):
    dims = m.group(2).split("x", 1)
    end = max(end, int(m.group(1)) + int(dims[0]) * ELEMENT[dims[1].rstrip(">") if dims[1] not in ELEMENT else dims[1]])
if end == 0 or word < end or word + 16 > 0x80000:
    sys.exit(f"the doorbell word at {word} is not in free memory: the buffers of memory tile {col} end at {end}")
print(f"doorbell word at {word}, buffers end at {end}")

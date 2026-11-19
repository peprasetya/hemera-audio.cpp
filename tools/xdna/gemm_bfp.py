# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Paulus Edwin Prasetya (hemera-audio.cpp, the AMD XDNA2 NPU port)
# Whole-array bfp16 x bfp16 -> fp32 matmul for small-batch workloads: C[M x N] = A[M x K] * B[N x K]^T.
# Structure follows mlir-aie programming_examples/basic/matrix_multiplication/whole_array
# (Copyright AMD, Apache-2.0 WITH LLVM-exception).
#
# A and B come pre-tiled from the host, every transfer one contiguous run:
#   A: row r, then K chunk kk: one m x k tile; row r walked N/(8n) times
#   B: column c, then its n-tile t, then K chunk kk: one n x k tile
#   C: column c, then row block, then n-tile t, then core row: one m x n tile of float32
import argparse
import re
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.ir import Module
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker, WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

SRC = str(Path(__file__).resolve().parent / "mm_bfp_f32.cc")
ROWS, COLS = 4, 8
B_GROUP, B_DEPTH = 16, 5  # 368 KB of weights per column of the array, 295 KB in the last two


def lin(total, offset, length, d0, repeat=1):
    """One contiguous run of `length` elements, repeated `repeat` times, as DMA dimensions."""
    assert length % d0 == 0, (length, d0)
    n = length // d0
    d1 = max(d for d in range(1, 1024) if n % d == 0)
    d2 = n // d1
    assert d2 < 1024 and repeat <= 64, (d2, repeat)
    return TensorAccessPattern((total,), offset, [repeat, d2, d1, d0], [0, d1 * d0, d0, 1])


def with_doorbell(module):
    """One more stream, on channels the placer leaves free: the host's doorbell, see ggml-xdna-insts.h."""
    text = str(module)
    last = [m for m in re.finditer(r"^\s*%\S+ = aie\.logical_tile<[^\n]*\n", text, re.M)][-1]
    flow = """    %db_shim = aie.tile(7, 0)
    %db_mem = aie.tile(7, 1)
    aie.flow(%db_shim, DMA : 1, %db_mem, DMA : 5)
"""
    return Module.parse(text[: last.end()] + flow + text[last.end() :], module.context)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def gemm(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
         m: CompileTime[int] = 8, k: CompileTime[int] = 64, n: CompileTime[int] = 64):
    assert M % (m * ROWS) == 0 and K % k == 0 and N % (n * COLS) == 0 and K // k >= 2
    a_l1 = np.ndarray[(m, k // 8), np.dtype[v8bfp16ebs8]]
    b_l1 = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    c_l1 = np.ndarray[(m, n), np.dtype[np.float32]]
    a_l2 = np.ndarray[(m * k // 8,), np.dtype[v8bfp16ebs8]]
    # B waits in the memory tile in groups of B_GROUP tiles: the next matmul's weights are fetched while the host is still
    # working, so a transfer has to be a whole number of groups.
    assert (N // n // COLS) * (K // k) % B_GROUP == 0, (N, K)
    b_l2 = np.ndarray[(B_GROUP * k * n // 8,), np.dtype[v8bfp16ebs8]]
    c_l2 = np.ndarray[(m * n * ROWS,), np.dtype[np.float32]]
    flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"]
    first = ExternalFunction("mm_first", source_file=SRC, arg_types=[a_l1, b_l1, c_l1], compile_flags=flags + ["-DMM_FIRST"])
    mid = ExternalFunction("mm_mid", source_file=SRC, arg_types=[a_l1, b_l1, c_l1], compile_flags=flags + ["-DMM_MID"])

    a3 = [ObjectFifo(a_l2, name=f"A3_{r}", depth=2) for r in range(ROWS)]
    a1 = [a3[r].cons().split([0], obj_types=[a_l1], names=[f"A1_{r}"])[0] for r in range(ROWS)]
    # one less in the last two columns: the last memory tile keeps room for the doorbell's word (check_doorbell_word.py)
    b3 = [ObjectFifo(b_l2, name=f"B3_{c}", depth=B_DEPTH - (c >= COLS - 2)) for c in range(COLS)]
    b1 = [b3[c].cons().forward(obj_type=b_l1, name=f"B1_{c}") for c in range(COLS)]
    c3 = [ObjectFifo(c_l2, name=f"C3_{c}", depth=2) for c in range(COLS)]
    c1 = [c3[c].prod().join([m * n * r for r in range(ROWS)], obj_types=[c_l1] * ROWS, names=[f"C1_{c}_{r}" for r in range(ROWS)], depths=[2] * ROWS) for c in range(COLS)]

    rtp = [[Buffer(np.ndarray[(2,), np.dtype[np.int32]], name=f"rtp_{r}_{c}", use_write_rtp=True) for c in range(COLS)] for r in range(ROWS)]
    bar = [[WorkerRuntimeBarrier() for _ in range(COLS)] for _ in range(ROWS)]

    def core(in_a, in_b, out_c, first, mid, rtp, bar):
        bar.wait_for_value(1)
        n_mid = rtp[0]
        n_tiles = rtp[1]
        bar.release_with_value(1)
        for _ in range_(n_tiles):
            o = out_c.acquire(1)
            x = in_a.acquire(1)
            y = in_b.acquire(1)
            first(x, y, o)
            in_a.release(1)
            in_b.release(1)
            for _ in range_(n_mid):
                x = in_a.acquire(1)
                y = in_b.acquire(1)
                mid(x, y, o)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(ROWS, COLS, lambda r, c: Worker(core, [a1[r].cons(), b1[c].cons(), c1[c][r].prod(), first, mid, rtp[r][c], bar[r][c]], stack_size=0xD00))
    a_prods = [f.prod() for f in a3]
    b_prods = [f.prod() for f in b3]
    c_cons = [f.cons() for f in c3]
    nk, per_col, rbs = K // k, N // n // COLS, M // m // ROWS
    a_total, b_total, c_total = M * K // 8, N * K // 8, M * N
    a_row = m * K // 8
    b_col = (N // COLS) * K // 8
    c_rb = per_col * ROWS * m * n

    def sequence(A, B, C, a_hs, b_hs, c_hs):
        for r in range(ROWS):
            for c in range(COLS):
                rtp[r][c][0] = nk - 1
                rtp[r][c][1] = (M // m) * (N // n) // (ROWS * COLS)
                bar[r][c].set(1)
        for rb, tg in TaskGroup.pipelined(rbs, depth=4):
            for c in range(COLS):
                c_hs[c].drain(C, tap=lin(c_total, (c * rbs + rb) * c_rb, c_rb, 1024 if c_rb % 1024 == 0 else 64), wait=True, group=tg)
                if c < ROWS:
                    a_hs[c].fill(A, tap=lin(a_total, (rb * ROWS + c) * a_row, a_row, 64, repeat=per_col), group=tg)
                b_hs[c].fill(B, tap=lin(b_total, c * b_col, b_col, 256), group=tg)

    rt = Runtime(sequence, [A, B, C, a_prods, b_prods, c_cons])
    return with_doorbell(Program(iron.get_current_device(), rt, workers=[w for row in workers for w in row]).resolve_program())


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p, default_dev="npu2")
    for name, dflt in (("-M", 64), ("-K", 1024), ("-N", 4096), ("-m", 8), ("-k", 64), ("-n", 64)):
        p.add_argument(name, type=int, default=dflt)
    opts = p.parse_args()

    def kwargs(o):
        return dict(
            A=np.ndarray[(o.M * o.K // 8,), np.dtype[v8bfp16ebs8]],
            B=np.ndarray[(o.N * o.K // 8,), np.dtype[v8bfp16ebs8]],
            C=np.ndarray[(o.M * o.N,), np.dtype[np.float32]],
            M=o.M, K=o.K, N=o.N, m=o.m, k=o.k, n=o.n)

    run_design_cli(gemm, opts, compile_kwargs=kwargs, device=lambda o: device_from_args(o, n_cols=COLS))


if __name__ == "__main__":
    main()

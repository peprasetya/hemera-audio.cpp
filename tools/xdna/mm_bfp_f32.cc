// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Paulus Edwin Prasetya (hemera-audio.cpp, the AMD XDNA2 NPU port)
// bfp16ebs8 x bfp16ebs8 -> float32 matmul tile for AIE2P, for the whole-array design in gemm_bfp.py.
// Based on mlir-aie aie_kernels/linalg/mm_bfp.cc (Apache-2.0 WITH LLVM-exception).
//
// C[m x n] += A[m x k] * B[n x k]^T. The sums stay in float32 between K chunks instead of being
// squeezed back into 8-bit blocks after every chunk, which is what makes the result accurate.
//
// B arrives with each row pair interleaved, so one stream reads both rows with no seeks: for each pair
// (j, j+1) of 8-row groups, for each 8-column block x: subtile (j, x), subtile (j+1, x). A the same when
// m is 16 or more; when m is 8 it is just its subtiles in order. C is 8x8 blocks: block (z, j) at float
// (z * n/8 + j) * 64, row-major inside.
#include <aie_api/aie.hpp>

namespace {
constexpr unsigned rowA = DIM_M / 8, colA = DIM_K / 8, colB = DIM_N / 8;
static_assert(DIM_M % 8 == 0 && DIM_N % 16 == 0 && DIM_K % 16 == 0);
using acc_t = aie::accum<accfloat, 64>;
using stream_t = aie::block_vector_input_buffer_stream<bfp16ebs8, 64>;

// One row group of A (m = 8) against pairs of B's groups.
template <bool first>
__attribute__((always_inline)) inline void mm_single(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *pC) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  float *c0 = pC;
  for (unsigned j = 0; j < colB; j += 2) {
    stream_t sA(pA);
    stream_t sB(pB);
    sB.seek(j * colA);
    acc_t c00, c01;
    if constexpr (first) {
      c00 = aie::zeros<accfloat, 64>();
      c01 = aie::zeros<accfloat, 64>();
    } else {
      c00 = acc_t(aie::load_v<64>(c0));
      c01 = acc_t(aie::load_v<64>(c0 + 64));
    }
    for (unsigned i = 0; i < colA; ++i) {
      aie::block_vector<bfp16ebs8, 64> A0 = sA.pop(), B0 = sB.pop(), B1 = sB.pop();
      c00 = mac_8x8_8x8T(A0, B0, c00);
      c01 = mac_8x8_8x8T(A0, B1, c01);
    }
    aie::store_v(c0, c00.template to_vector<float>());
    aie::store_v(c0 + 64, c01.template to_vector<float>());
    c0 += 128;
  }
}

template <bool first>
__attribute__((always_inline)) inline void mm_paired(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *pC) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (unsigned z = 0; z < rowA; z += 2) {
    float *c0 = pC + (z * colB) * 64;
    float *c1 = pC + ((z + 1) * colB) * 64;
    for (unsigned j = 0; j < colB; j += 2) {
      stream_t sA(pA);
      sA.seek(z * colA);
      stream_t sB(pB);
      sB.seek(j * colA);
      acc_t c00, c01, c10, c11;
      if constexpr (first) {
        c00 = aie::zeros<accfloat, 64>();
        c01 = aie::zeros<accfloat, 64>();
        c10 = aie::zeros<accfloat, 64>();
        c11 = aie::zeros<accfloat, 64>();
      } else {
        c00 = acc_t(aie::load_v<64>(c0));
        c01 = acc_t(aie::load_v<64>(c0 + 64));
        c10 = acc_t(aie::load_v<64>(c1));
        c11 = acc_t(aie::load_v<64>(c1 + 64));
      }
      for (unsigned i = 0; i < colA; ++i) {
        aie::block_vector<bfp16ebs8, 64> A0 = sA.pop(), A1 = sA.pop(), B0 = sB.pop(), B1 = sB.pop();
        c00 = mac_8x8_8x8T(A0, B0, c00);
        c01 = mac_8x8_8x8T(A0, B1, c01);
        c10 = mac_8x8_8x8T(A1, B0, c10);
        c11 = mac_8x8_8x8T(A1, B1, c11);
      }
      aie::store_v(c0, c00.template to_vector<float>());
      aie::store_v(c0 + 64, c01.template to_vector<float>());
      aie::store_v(c1, c10.template to_vector<float>());
      aie::store_v(c1 + 64, c11.template to_vector<float>());
      c0 += 128;
      c1 += 128;
    }
  }
}

template <bool first>
__attribute__((always_inline)) inline void mm(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *pC) {
  if constexpr (rowA == 1)
    mm_single<first>(pA, pB, pC);
  else
    mm_paired<first>(pA, pB, pC);
}
} // namespace

extern "C" {
#ifdef MM_FIRST
void mm_first(bfp16ebs8 *pA, bfp16ebs8 *pB, float *pC) { mm<true>(pA, pB, pC); }
#endif
#ifdef MM_MID
void mm_mid(bfp16ebs8 *pA, bfp16ebs8 *pB, float *pC) { mm<false>(pA, pB, pC); }
#endif
}

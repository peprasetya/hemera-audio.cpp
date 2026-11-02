// Matrix multiplies on the AMD XDNA2 NPU (Strix, Strix Halo, Krackan), from inside the CPU backend.
// Set GGML_XDNA=1 to turn it on. See ggml-xdna.cpp.
#pragma once

#include "ggml.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Whether the NPU computes this matmul. The answer is the same on every thread of the team.
bool ggml_xdna_owns(const struct ggml_tensor * dst);

// Computes a matmul that ggml_xdna_owns said yes to.
void ggml_xdna_mul_mat(struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif

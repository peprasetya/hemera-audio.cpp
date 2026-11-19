// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Paulus Edwin Prasetya (hemera-audio.cpp, the AMD XDNA2 NPU port)
// Matrix multiplies on the AMD XDNA2 NPU (Strix, Strix Halo, Krackan), from inside the CPU backend.
// Set GGML_XDNA=1 to turn it on (--backend npu does). See ggml-xdna.cpp.
#pragma once

#include "ggml.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cgraph;
struct ggml_threadpool;

// Around a graph's computation, on the thread that starts it: prepares one NPU job for all the graph's matmuls that the
// NPU should do, and ends it.
void ggml_xdna_graph_begin(const struct ggml_cgraph * graph, int nth);
void ggml_xdna_graph_end(void);

// Whether this node is the graph's next NPU matmul. Same answer on every thread of the team.
bool ggml_xdna_planned(const struct ggml_tensor * dst);

// Computes it, on every thread of the team: the host's work is shared, the NPU's is started by thread 0.
void ggml_xdna_mul_mat(int ith, int nth, struct ggml_threadpool * threadpool, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif

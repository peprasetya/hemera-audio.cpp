// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Paulus Edwin Prasetya (hemera-audio.cpp, the AMD XDNA2 NPU port)
// The multiply for short K in ggml_compute_forward_mul_mat against the plain formula, with batches and broadcast.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);
    struct shape { int64_t K, N, M, b2, b3, a2, a3; };
    const shape shapes[] = {{11, 128, 11, 32, 1, 32, 1}, {11, 128, 352, 1, 1, 1, 1}, {3, 64, 5, 4, 2, 2, 1}, {31, 100, 7, 6, 1, 3, 1},
                            {1, 40, 9, 1, 1, 1, 1}, {16, 257, 4, 2, 2, 2, 2}, {11, 128, 11, 32, 1, 1, 1}};
    int bad = 0;
    for (const shape & s : shapes)
        for (int threads : {1, 2, 3, 8}) {
            ggml_init_params ip{ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.K, s.N, s.a2, s.a3);
            ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.K, s.M, s.b2, s.b3);
            ggml_tensor * c = ggml_mul_mat(ctx, a, b);
            ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, c);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
            ggml_gallocr_alloc_graph(al, g);
            auto fill = [&](ggml_tensor * t) { float * d = (float *) t->data; for (int64_t i = 0; i < ggml_nelements(t); i++) d[i] = val(rng); };
            fill(a);
            fill(b);
            ggml_backend_cpu_set_n_threads(cpu, threads);
            ggml_backend_graph_compute(cpu, g);
            const float * A = (const float *) a->data, * B = (const float *) b->data, * C = (const float *) c->data;
            double worst = 0;
            const int64_t r2 = s.b2 / s.a2, r3 = s.b3 / s.a3;
            for (int64_t i3 = 0; i3 < s.b3; i3++)
                for (int64_t i2 = 0; i2 < s.b2; i2++)
                    for (int64_t i = 0; i < s.M; i++)
                        for (int64_t j = 0; j < s.N; j++) {
                            double e = 0;
                            const float * row = A + ((i3 / r3 * s.a2 + i2 / r2) * s.N + j) * s.K;
                            const float * x = B + ((i3 * s.b2 + i2) * s.M + i) * s.K;
                            for (int64_t k = 0; k < s.K; k++) e += (double) row[k] * x[k];
                            const double got = C[((i3 * s.b2 + i2) * s.M + i) * s.N + j];
                            worst = std::max(worst, std::fabs(got - e));
                        }
            const bool ok = worst < 1e-5;
            if (!ok) bad++;
            printf("K=%lld N=%lld M=%lld batches %lldx%lld over %lldx%lld, %d threads: worst %.2e %s\n", (long long) s.K, (long long) s.N, (long long) s.M,
                   (long long) s.b2, (long long) s.b3, (long long) s.a2, (long long) s.a3, threads, worst, ok ? "ok" : "WRONG");
            ggml_gallocr_free(al);
            ggml_free(ctx);
        }
    printf(bad ? "%d WRONG\n" : "all right\n", bad);
    return bad != 0;
}

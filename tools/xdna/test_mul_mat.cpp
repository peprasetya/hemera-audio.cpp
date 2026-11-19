// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Paulus Edwin Prasetya (hemera-audio.cpp, the AMD XDNA2 NPU port)
// The NPU path against the exact product, over shapes that need padding, large batches and both kinds of
// weight, in mixed order. Run with GGML_XDNA=1 GGML_XDNA_FORCE=1. Exit code 0 when every shape is within
// tolerance. usage: test_mul_mat <rounds>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

struct shape { int64_t M, K, N; bool quant; };

int main(int argc, char ** argv) {
    const int rounds = argc > 1 ? atoi(argv[1]) : 2;
    const std::vector<shape> shapes = {
        {64, 1024, 4096, true}, {22, 1024, 4096, true}, {22, 4096, 1024, true}, {1, 2048, 6144, true},
        {5, 1024, 8192, true}, {29, 2048, 12288, true}, {65, 1024, 512, true}, {128, 2048, 1024, true},
        {300, 512, 512, true}, {500, 1024, 2048, true}, {1500, 512, 1024, true},
        {7, 96, 300, true}, {64, 160, 1000, true}, {100, 32, 513, true},
        {22, 100, 700, false}, {3, 777, 1025, false}, {64, 64, 512, false}, {40, 1000, 2000, false},
    };
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0, 1);
    std::vector<int> order;
    for (int r = 0; r < rounds; r++) for (size_t i = 0; i < shapes.size(); i++) order.push_back((int) i);
    std::shuffle(order.begin(), order.end(), rng);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 2);
    int bad = 0;
    for (int idx : order) {
        const shape s = shapes[idx];
        ggml_init_params ip{ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, s.quant ? GGML_TYPE_Q8_0 : GGML_TYPE_F32, s.K, s.N);
        ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.K, s.M);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_backend_buffer_t rest = ggml_backend_alloc_ctx_tensors(ctx, cpu);

        std::vector<float> W(s.K * s.N);
        for (auto & v : W) v = nd(rng);
        if (s.quant) {
            const auto * traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
            const size_t row = ggml_row_size(GGML_TYPE_Q8_0, s.K);
            for (int64_t j = 0; j < s.N; j++) traits->from_float(&W[j * s.K], (char *) w->data + j * row, s.K);
            std::vector<float> back(s.K);
            const auto * base = ggml_get_type_traits(GGML_TYPE_Q8_0);
            for (int64_t j = 0; j < s.N; j++) { base->to_float((char *) w->data + j * row, back.data(), s.K); std::copy(back.begin(), back.end(), &W[j * s.K]); }
        } else {
            memcpy(w->data, W.data(), W.size() * sizeof(float));
        }
        std::vector<float> X(s.K * s.M);
        for (auto & v : X) v = nd(rng);
        memcpy(x->data, X.data(), X.size() * sizeof(float));
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(cpu, g);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const float * out = (const float *) y->data;
        double se = 0, sr = 0;
        for (int64_t r = 0; r < s.M; r += std::max<int64_t>(1, s.M / 24))
            for (int64_t j = 0; j < s.N; j += std::max<int64_t>(1, s.N / 400)) {
                double e = 0;
                for (int64_t q = 0; q < s.K; q++) e += (double) X[r * s.K + q] * W[j * s.K + q];
                const double d = out[r * s.N + j] - e;
                se += d * d;
                sr += e * e;
            }
        const double err = std::sqrt(se / sr);
        const bool ok = err < 0.03;
        if (!ok) {
            bad++;
            // where it is wrong: per block of 32 rows, and per array column
            const auto where = [&](const char * name) {
                printf("  %s: by row block:", name);
                for (int64_t rb = 0; rb < std::min<int64_t>(16, (s.M + 31) / 32); rb++) {
                    double e2 = 0, r2 = 0;
                    for (int64_t r = rb * 32; r < std::min(s.M, (rb + 1) * 32); r += 3) for (int64_t j = 0; j < s.N; j += 17) {
                        double e = 0;
                        for (int64_t q = 0; q < s.K; q++) e += (double) X[r * s.K + q] * W[j * s.K + q];
                        e2 += (out[r * s.N + j] - e) * (out[r * s.N + j] - e); r2 += e * e;
                    }
                    printf(" %.3f", std::sqrt(e2 / r2));
                }
                printf("\n  %s: by array column:", name);
                for (int c = 0; c < 8; c++) {
                    double e2 = 0, r2 = 0;
                    for (int64_t r = 0; r < s.M; r += 7) for (int64_t j = c * 64; j < s.N; j += 8 * 64 + 13) {
                        double e = 0;
                        for (int64_t q = 0; q < s.K; q++) e += (double) X[r * s.K + q] * W[j * s.K + q];
                        e2 += (out[r * s.N + j] - e) * (out[r * s.N + j] - e); r2 += e * e;
                    }
                    printf(" %.3f", std::sqrt(e2 / r2));
                }
                printf("\n");
            };
            where("first run");
            ggml_backend_graph_compute(cpu, g);
            where("second run");
        }
        printf("%-6s M=%-5ld K=%-5ld N=%-6ld %8.3f ms  rms error %.5f  %s\n", s.quant ? "q8_0" : "f32", (long) s.M, (long) s.K, (long) s.N, ms, err, ok ? "ok" : "BAD");
        fflush(stdout);
        ggml_backend_buffer_free(rest);
        ggml_backend_buffer_free(weights);
        ggml_free(ctx);
    }
    // Several matmuls in one graph, as a model has them: one NPU job, with its doorbells. More than one job's worth too.
    for (int chain : {4, 24, 70}) {
        const int64_t M = 22, K0 = 1024;
        ggml_init_params ip{ggml_tensor_overhead() * (size_t) (chain * 2 + 8) + ggml_graph_overhead_custom(1024, false), nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        std::vector<ggml_tensor *> ws;
        std::vector<int64_t> Ks = {K0};
        for (int i = 0; i < chain; i++) {
            const int64_t N = i % 2 == 0 ? 2048 : 1024;
            ws.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, Ks.back(), N));
            Ks.push_back(N);
        }
        ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K0, M);
        ggml_tensor * y = x;
        for (int i = 0; i < chain; i++) y = ggml_mul_mat(ctx, ws[i], y);
        ggml_cgraph * g = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(g, y);
        ggml_backend_buffer_t rest = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        std::vector<std::vector<float>> Ws(chain);
        std::normal_distribution<float> small(0, 0.03f);
        for (int i = 0; i < chain; i++) {
            const int64_t K = ws[i]->ne[0], N = ws[i]->ne[1];
            std::vector<float> W(K * N);
            for (auto & v : W) v = small(rng);
            const size_t row = ggml_row_size(GGML_TYPE_Q8_0, K);
            const auto * traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
            for (int64_t j = 0; j < N; j++) traits->from_float(&W[j * K], (char *) ws[i]->data + j * row, K);
            const auto * base = ggml_get_type_traits(GGML_TYPE_Q8_0);
            std::vector<float> back(K);
            for (int64_t j = 0; j < N; j++) { base->to_float((char *) ws[i]->data + j * row, back.data(), K); std::copy(back.begin(), back.end(), &W[j * K]); }
            Ws[i] = std::move(W);
        }
        std::vector<float> X(K0 * M);
        for (auto & v : X) v = nd(rng);
        memcpy(x->data, X.data(), X.size() * sizeof(float));
        for (int rep = 0; rep < 3; rep++) {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(cpu, g);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rep == 0) printf("chain of %d: first run %.1f ms\n", chain, ms);
            if (rep != 2) continue;
            printf("chain of %d: %.3f ms a matmul\n", chain, ms / chain);
        }
        // the first rows and the output of the first two, against the exact product
        std::vector<float> first(2048);
        double se = 0, sr = 0;
        const float * out = (const float *) ggml_graph_node(g, 0)->data;
        for (int64_t r = 0; r < M; r += 7) for (int64_t j = 0; j < 2048; j += 13) {
            double e = 0;
            for (int64_t q = 0; q < K0; q++) e += (double) X[r * K0 + q] * Ws[0][j * K0 + q];
            const double d = out[r * 2048 + j] - e;
            se += d * d; sr += e * e;
        }
        const double err = std::sqrt(se / sr);
        printf("chain of %d: rms error of the first multiply %.5f %s\n", chain, err, err < 0.03 ? "ok" : "BAD");
        if (err >= 0.03) bad++;
        ggml_backend_buffer_free(rest);
        ggml_backend_buffer_free(weights);
        ggml_free(ctx);
    }
    ggml_backend_free(cpu);
    printf(bad ? "FAILED: %d shapes\n" : "all shapes within tolerance\n", bad);
    return bad != 0;
}

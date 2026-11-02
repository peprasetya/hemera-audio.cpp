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
        if (!ok) bad++;
        printf("%-6s M=%-5ld K=%-5ld N=%-6ld %8.3f ms  rms error %.5f  %s\n", s.quant ? "q8_0" : "f32", (long) s.M, (long) s.K, (long) s.N, ms, err, ok ? "ok" : "BAD");
        fflush(stdout);
        ggml_backend_buffer_free(rest);
        ggml_backend_buffer_free(weights);
        ggml_free(ctx);
    }
    ggml_backend_free(cpu);
    printf(bad ? "FAILED: %d shapes\n" : "all shapes within tolerance\n", bad);
    return bad != 0;
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Paulus Edwin Prasetya (hemera-audio.cpp, the AMD XDNA2 NPU port)
// Compares the host-side instruction generator with what the IRON compiler produced for the same shape.
// usage: check_insts <insts.bin> <M> <K> <N> ...
#include "ggml-xdna-insts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char ** argv) {
    int bad = 0;
    for (int i = 1; i + 3 < argc; i += 4) {
        std::ifstream f(argv[i], std::ios::binary);
        std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
        std::vector<uint32_t> want(b.size() / 4);
        memcpy(want.data(), b.data(), want.size() * 4);
        const auto got = ggml_xdna::make_insts(atoll(argv[i + 1]), atoll(argv[i + 2]), atoll(argv[i + 3]));
        const bool same = want == got;
        printf("%-44s %s (%zu words)\n", argv[i], same ? "same" : "DIFFERENT", got.size());
        if (!same) {
            bad++;
            for (size_t w = 0; w < std::min(want.size(), got.size()); w++)
                if (want[w] != got[w]) { printf("  first difference at word %zu: IRON %08x, here %08x\n", w, want[w], got[w]); break; }
            if (want.size() != got.size()) printf("  sizes: IRON %zu, here %zu\n", want.size(), got.size());
        }
    }
    return bad != 0;
}

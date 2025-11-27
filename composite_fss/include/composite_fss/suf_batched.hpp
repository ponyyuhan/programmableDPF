#pragma once

#include "suf.hpp"
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cfss {

struct BatchedSufDesc {
    SufDesc scalar_suf;                 // SUF template (without per-instance masks)
    std::vector<std::uint64_t> r_in;    // per-instance input masks
    std::vector<std::uint64_t> r_out;   // optional per-instance output masks
};

struct BatchedPdpfKey {
    std::vector<SufCompiled> per_instance;
};

inline BatchedPdpfKey suf2pdpf_batched(const BatchedSufDesc &desc, PdpfEngine &engine) {
    if (desc.r_in.empty()) throw std::runtime_error("suf2pdpf_batched: r_in must be non-empty");
    std::size_t batch = desc.r_in.size();
    if (!desc.r_out.empty() && desc.r_out.size() != batch) {
        throw std::runtime_error("suf2pdpf_batched: r_out size mismatch");
    }
    BatchedPdpfKey key;
    key.per_instance.resize(batch);
    for (std::size_t i = 0; i < batch; ++i) {
        SufDesc inst = desc.scalar_suf;
        inst.r_in = desc.r_in[i];
        inst.r_out = desc.r_out.empty() ? desc.scalar_suf.r_out : desc.r_out[i];
        key.per_instance[i] = compile_suf_to_pdpf(inst, engine);
    }
    return key;
}

inline void batched_eval_share(const BatchedPdpfKey &key,
                               int party,
                               std::size_t idx,
                               std::uint64_t masked_x,
                               PdpfEngine &engine,
                               std::vector<std::uint64_t> &out_words) {
    if (idx >= key.per_instance.size()) {
        out_words.clear();
        return;
    }
    const auto &compiled = key.per_instance[idx];
    std::size_t out_words_count = compiled.output_words ? compiled.output_words : 1;
    out_words.assign(out_words_count, 0);
    engine.eval_share(compiled.pdpf_program, party, masked_x, out_words);
}

inline void batched_eval_all(const BatchedPdpfKey &key,
                             int party,
                             const std::vector<std::uint64_t> &masked_xs,
                             PdpfEngine &engine,
                             std::vector<std::vector<std::uint64_t>> &out_words) {
    if (masked_xs.size() != key.per_instance.size()) {
        throw std::runtime_error("batched_eval_all: input size mismatch");
    }
    out_words.resize(masked_xs.size());
    for (std::size_t i = 0; i < masked_xs.size(); ++i) {
        batched_eval_share(key, party, i, masked_xs[i], engine, out_words[i]);
    }
}

} // namespace cfss


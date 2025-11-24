#pragma once

#include "suf.hpp"

namespace cfss {

// Evaluate a scalar SUF descriptor directly in the clear.
inline std::uint64_t eval_suf_scalar(const SufDesc &desc, std::uint64_t x) {
    if (desc.shape.domain_bits == 0) throw std::runtime_error("eval_suf_scalar: no domain");
    std::uint64_t mask = (desc.shape.domain_bits == 64) ? ~0ULL : ((1ULL << desc.shape.domain_bits) - 1ULL);
    std::uint64_t unmasked = 0;
    if (desc.shape.domain_bits == 64) {
        unmasked = x - desc.r_in;
    } else {
        std::uint64_t modulus = 1ULL << desc.shape.domain_bits;
        unmasked = (x + modulus - (desc.r_in & mask)) & mask;
    }
    std::size_t interval_idx = find_interval(desc, unmasked);
    if (interval_idx >= desc.polys.size()) return 0;
    if (desc.r_outputs == 0) return 0;
    const auto &p = desc.polys[interval_idx].polys[0];
    std::uint64_t val = eval_poly_mod(p, unmasked);
    RingConfig cfg = make_ring_config(desc.shape.domain_bits);
    return ring_add(cfg, val, desc.r_out & mask);
}

// Evaluate a vector SUF (stacked outputs) in the clear.
inline std::vector<std::uint64_t> eval_suf_vector(const std::vector<SufDesc> &scalars,
                                                  std::uint64_t x) {
    std::vector<std::uint64_t> out;
    out.reserve(scalars.size());
    for (const auto &s : scalars) {
        out.push_back(eval_suf_scalar(s, x));
    }
    return out;
}

// Stack multiple scalar SUFs (same domain/word bits) into a "multi" view.
struct StackedSuf {
    SufShape shape;
    std::vector<SufDesc> scalars;
};

inline StackedSuf stack_suf_outputs(const std::vector<SufDesc> &scalars) {
    if (scalars.empty()) throw std::runtime_error("stack_suf_outputs: empty");
    StackedSuf stacked;
    stacked.shape.domain_bits = scalars[0].shape.domain_bits;
    stacked.shape.word_bits = scalars[0].shape.word_bits;
    stacked.shape.num_words = static_cast<uint32_t>(scalars.size());
    stacked.scalars = scalars;
    return stacked;
}

inline std::vector<std::uint64_t> eval_suf_vector(const StackedSuf &stacked, std::uint64_t x) {
    return eval_suf_vector(stacked.scalars, x);
}

} // namespace cfss

// ================================================================
// File: src/pprf/pprf.cpp
// ================================================================

#include "pdpf/pprf/pprf.hpp"
#include <cmath>
#include <stdexcept>

namespace pdpf::pprf {

Pprf::Pprf(std::shared_ptr<prg::IPrg> prg)
    : prg_(std::move(prg)) {}

/**
 * Compute depth d = ceil(log2(M)).
 */
std::uint32_t Pprf::tree_depth(std::uint64_t M) const {
    if (M == 0) {
        throw std::invalid_argument("Pprf::tree_depth: M = 0");
    }
    std::uint32_t d = 0;
    std::uint64_t x = 1;
    while (x < M) {
        x <<= 1;
        ++d;
    }
    return d;
}

void Pprf::seed_to_children(const core::Seed &parent,
                            core::Seed &left,
                            core::Seed &right) const {
    prg_->expand(parent, left, right);
}

std::uint64_t Pprf::seed_to_uint64(const core::Seed &seed) const {
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < 8 && i < seed.size(); ++i) {
        v |= static_cast<std::uint64_t>(seed[i]) << (8 * i);
    }
    return v;
}

std::uint64_t Pprf::eval(const PprfKey &k, std::uint64_t x) const {
    if (x >= k.params.M) {
        throw std::out_of_range("Pprf::eval: x >= M");
    }
    if (k.params.N == 0) {
        throw std::invalid_argument("Pprf::eval: N = 0");
    }

    std::uint32_t d = tree_depth(k.params.M);
    core::Seed seed = k.root_seed;

    for (std::uint32_t lvl = 0; lvl < d; ++lvl) {
        core::Seed left{}, right{};
        seed_to_children(seed, left, right);
        std::uint32_t bit = (x >> (d - 1 - lvl)) & 1u;
        seed = (bit == 0) ? left : right;
    }

    return seed_to_uint64(seed) % k.params.N;
}

void Pprf::eval_all(const PprfKey &k,
                    std::vector<std::uint64_t> &out) const {
    if (k.params.N == 0) {
        throw std::invalid_argument("Pprf::eval_all: N = 0");
    }
    std::uint32_t d = tree_depth(k.params.M);
    std::vector<core::Seed> level;
    level.push_back(k.root_seed);

    for (std::uint32_t lvl = 0; lvl < d; ++lvl) {
        std::vector<core::Seed> next;
        next.reserve(level.size() * 2);
        for (const auto &node : level) {
            core::Seed l{}, r{};
            seed_to_children(node, l, r);
            next.push_back(l);
            next.push_back(r);
        }
        level.swap(next);
    }

    std::size_t leaves = level.size(); // = 2^d
    out.resize(static_cast<std::size_t>(k.params.M));
    for (std::size_t i = 0; i < out.size(); ++i) {
        // Only use first M leaves.
        out[i] = seed_to_uint64(level[i]) % k.params.N;
    }
}

PprfPuncturedKey Pprf::puncture(const PprfKey &k, std::uint64_t xp) const {
    if (xp >= k.params.M) {
        throw std::out_of_range("Pprf::puncture: xp >= M");
    }
    if (k.params.N == 0) {
        throw std::invalid_argument("Pprf::puncture: N = 0");
    }

    PprfPuncturedKey kp;
    kp.params = k.params;
    kp.xp = xp;

    std::uint32_t d = tree_depth(k.params.M);
    kp.co_path_seeds.resize(d);

    core::Seed current = k.root_seed;
    for (std::uint32_t lvl = 0; lvl < d; ++lvl) {
        core::Seed left{}, right{};
        seed_to_children(current, left, right);
        std::uint32_t bit = (xp >> (d - 1 - lvl)) & 1u;
        if (bit == 0) {
            kp.co_path_seeds[lvl] = right;
            current = left;
        } else {
            kp.co_path_seeds[lvl] = left;
            current = right;
        }
    }

    return kp;
}

std::uint64_t Pprf::punc_eval(const PprfPuncturedKey &kp,
                              std::uint64_t x) const {
    if (x >= kp.params.M) {
        throw std::out_of_range("Pprf::punc_eval: x >= M");
    }
    if (x == kp.xp) {
        return PUNCTURED_SENTINEL;
    }
    if (kp.params.N == 0) {
        throw std::invalid_argument("Pprf::punc_eval: N = 0");
    }

    std::uint32_t d = tree_depth(kp.params.M);
    // Find first level where x and xp differ.
    std::uint32_t diverge_lvl = d; // sentinel
    for (std::uint32_t lvl = 0; lvl < d; ++lvl) {
        std::uint32_t bit_x  = (x  >> (d - 1 - lvl)) & 1u;
        std::uint32_t bit_xp = (kp.xp >> (d - 1 - lvl)) & 1u;
        if (bit_x != bit_xp) {
            diverge_lvl = lvl;
            break;
        }
    }
    if (diverge_lvl == d) {
        // Should not happen because x != xp.
        return PUNCTURED_SENTINEL;
    }

    core::Seed seed = kp.co_path_seeds[diverge_lvl];
    for (std::uint32_t lvl = diverge_lvl + 1; lvl < d; ++lvl) {
        core::Seed left{}, right{};
        seed_to_children(seed, left, right);
        std::uint32_t bit = (x >> (d - 1 - lvl)) & 1u;
        seed = (bit == 0) ? left : right;
    }

    return seed_to_uint64(seed) % kp.params.N;
}

void Pprf::punc_eval_all(const PprfPuncturedKey &kp,
                         std::vector<std::uint64_t> &out) const {
    out.assign(kp.params.M, 0);
    for (std::uint64_t x = 0; x < kp.params.M; ++x) {
        out[x] = punc_eval(kp, x);
    }
}

} // namespace pdpf::pprf

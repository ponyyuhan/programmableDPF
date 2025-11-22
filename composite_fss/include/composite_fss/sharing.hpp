#pragma once

#include "ring.hpp"
#include <random>
#include <utility>

namespace cfss {

// Deterministic splitter used so each party, with the same inputs, obtains
// additive shares that reconstruct to `value` modulo the ring.
inline Share deterministic_share(int party,
                                 const Ring64 &ring,
                                 u64 value,
                                 u64 nonce) {
    std::mt19937_64 rng(nonce);
    u64 r = rng() & ring.modulus_mask;
    if (party == 0) {
        return Share{party, r};
    }
    return Share{party, ring.sub(value, r)};
}

struct MPCContext {
    Ring64 ring;
    std::mt19937_64 rng;

    explicit MPCContext(unsigned n_bits = 64, std::uint64_t seed = 0xC0FFEE)
        : ring(n_bits), rng(seed) {}

    // Sample a fresh random additive sharing of x.
    std::pair<Share, Share> share_value(u64 x) {
        u64 s0 = rng() & ring.modulus_mask;
        u64 s1 = ring.sub(x, s0);
        return {Share{0, s0}, Share{1, s1}};
    }

    u64 reconstruct(const Share &a0, const Share &a1) const {
        return ring.add(a0.v, a1.v);
    }

    Share add(const Share &a, const Share &b) const {
        return Share{a.party, ring.add(a.v, b.v)};
    }

    Share add_const(const Share &a, u64 c) const {
        return Share{a.party, ring.add(a.v, c)};
    }

    Share sub(const Share &a, const Share &b) const {
        return Share{a.party, ring.sub(a.v, b.v)};
    }
};

} // namespace cfss

#pragma once

#include "ring.hpp"
#include "arith.hpp"
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
        unsigned bits = (ring.modulus_mask == ~0ULL)
                            ? 64u
                            : static_cast<unsigned>(64 - __builtin_clzll(ring.modulus_mask));
        RingConfig cfg = make_ring_config(bits);
        return debug_open(cfg, a0, a1);
    }

    Share add(const Share &a, const Share &b) const {
        unsigned bits = (ring.modulus_mask == ~0ULL)
                            ? 64u
                            : static_cast<unsigned>(64 - __builtin_clzll(ring.modulus_mask));
        RingConfig cfg = make_ring_config(bits);
        return cfss::add(cfg, a, b);
    }

    Share add_const(const Share &a, u64 c) const {
        unsigned bits = (ring.modulus_mask == ~0ULL)
                            ? 64u
                            : static_cast<unsigned>(64 - __builtin_clzll(ring.modulus_mask));
        RingConfig cfg = make_ring_config(bits);
        return cfss::add_const(cfg, a, c);
    }

    Share sub(const Share &a, const Share &b) const {
        unsigned bits = (ring.modulus_mask == ~0ULL)
                            ? 64u
                            : static_cast<unsigned>(64 - __builtin_clzll(ring.modulus_mask));
        RingConfig cfg = make_ring_config(bits);
        return cfss::sub(cfg, a, b);
    }
};

} // namespace cfss

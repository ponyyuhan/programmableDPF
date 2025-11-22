#pragma once

#include "arith.hpp"
#include <random>

namespace cfss {

struct BeaverTriple {
    std::uint64_t a;
    std::uint64_t b;
    std::uint64_t c;
};

// Deterministic PRG-based triple pool; both parties seed with same value to
// obtain correlated triples without interaction.
class BeaverPool {
public:
    BeaverPool(const RingConfig &cfg, std::uint64_t seed, int party)
        : cfg_(cfg), party_(party), prng_(seed) {}

    BeaverTriple next_triple_public() {
        std::uniform_int_distribution<std::uint64_t> dist(0, cfg_.modulus_mask);
        std::uint64_t a = dist(prng_);
        std::uint64_t b = dist(prng_);
        std::uint64_t c = ring_mul(cfg_, a, b);
        return BeaverTriple{a, b, c};
    }

    // Return the share of triple (a,b,c) for this party.
    BeaverTriple share_triple(const BeaverTriple &pub) {
        std::uniform_int_distribution<std::uint64_t> dist(0, cfg_.modulus_mask);
        std::uint64_t a0 = dist(prng_);
        std::uint64_t b0 = dist(prng_);
        std::uint64_t c0 = dist(prng_);
        std::uint64_t a1 = ring_sub(cfg_, pub.a, a0);
        std::uint64_t b1 = ring_sub(cfg_, pub.b, b0);
        std::uint64_t c1 = ring_sub(cfg_, pub.c, c0);
        if (party_ == 0) return BeaverTriple{a0, b0, c0};
        return BeaverTriple{a1, b1, c1};
    }

private:
    RingConfig cfg_;
    int party_;
    std::mt19937_64 prng_;
};

struct MulLocal {
    std::uint64_t d;
    std::uint64_t e;
};

inline MulLocal mul_prepare(const RingConfig &cfg,
                            const Share &x,
                            const Share &y,
                            const BeaverTriple &triple) {
    MulLocal m;
    m.d = ring_sub(cfg, x.v, triple.a);
    m.e = ring_sub(cfg, y.v, triple.b);
    return m;
}

inline std::pair<std::uint64_t, std::uint64_t> mul_open(const RingConfig &cfg,
                                                        const MulLocal &m0,
                                                        const MulLocal &m1) {
    return {ring_add(cfg, m0.d, m1.d), ring_add(cfg, m0.e, m1.e)};
}

inline Share mul_finish(const RingConfig &cfg,
                        int party,
                        const BeaverTriple &triple,
                        std::uint64_t d_open,
                        std::uint64_t e_open) {
    std::uint64_t term = triple.c;
    term = ring_add(cfg, term, ring_mul(cfg, d_open, triple.b));
    term = ring_add(cfg, term, ring_mul(cfg, e_open, triple.a));
    if (party == 1) {
        term = ring_add(cfg, term, ring_mul(cfg, d_open, e_open));
    }
    return Share{party, term};
}

inline Share add(const RingConfig &cfg, const Share &a, const Share &b) {
    return Share{a.party, ring_add(cfg, a.v, b.v)};
}

inline Share sub(const RingConfig &cfg, const Share &a, const Share &b) {
    return Share{a.party, ring_sub(cfg, a.v, b.v)};
}

inline Share negate(const RingConfig &cfg, const Share &a) {
    return Share{a.party, ring_sub(cfg, 0, a.v)};
}

inline Share constant(const RingConfig &cfg, std::uint64_t v, int party) {
    return Share{party, v & cfg.modulus_mask};
}

} // namespace cfss

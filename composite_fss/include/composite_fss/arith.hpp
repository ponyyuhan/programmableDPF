#pragma once

#include "ring.hpp"
#include <cstdint>

namespace cfss {

// Share invariant: each party holds (party, value) and should never
// reconstruct except via explicit open() in tests.

struct RingConfig {
    std::uint32_t n_bits;
    std::uint64_t modulus_mask;
};

inline RingConfig make_ring_config(std::uint32_t n_bits) {
    RingConfig cfg{n_bits, (n_bits == 64) ? ~0ULL : ((1ULL << n_bits) - 1ULL)};
    return cfg;
}

inline std::uint64_t ring_add(const RingConfig &cfg, std::uint64_t a, std::uint64_t b) {
    return (a + b) & cfg.modulus_mask;
}

inline std::uint64_t ring_sub(const RingConfig &cfg, std::uint64_t a, std::uint64_t b) {
    return (a - b) & cfg.modulus_mask;
}

inline std::uint64_t ring_mul(const RingConfig &cfg, std::uint64_t a, std::uint64_t b) {
    return (a * b) & cfg.modulus_mask;
}

inline std::int64_t to_signed(const RingConfig &cfg, std::uint64_t x) {
    if (cfg.n_bits == 64) return static_cast<std::int64_t>(x);
    std::uint64_t sign_bit = 1ULL << (cfg.n_bits - 1);
    if (x & sign_bit) {
        std::uint64_t ext = ~((1ULL << cfg.n_bits) - 1ULL);
        return static_cast<std::int64_t>(x | ext);
    }
    return static_cast<std::int64_t>(x);
}

// Debug-only open used in tests.
inline std::uint64_t open(const RingConfig &cfg, const Share &a0, const Share &a1) {
    return ring_add(cfg, a0.v, a1.v);
}

} // namespace cfss

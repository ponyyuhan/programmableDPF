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

inline Share add(const RingConfig &cfg, const Share &a, const Share &b) {
    return Share{a.party_, ring_add(cfg, a.v_, b.v_)};
}

inline Share sub(const RingConfig &cfg, const Share &a, const Share &b) {
    return Share{a.party_, ring_sub(cfg, a.v_, b.v_)};
}

inline Share negate(const RingConfig &cfg, const Share &a) {
    return Share{a.party_, ring_sub(cfg, 0, a.v_)};
}

inline Share add_const(const RingConfig &cfg, const Share &a, std::uint64_t c) {
    return Share{a.party_, ring_add(cfg, a.v_, c)};
}

inline Share mul_const(const RingConfig &cfg, const Share &a, std::uint64_t c) {
    return Share{a.party_, ring_mul(cfg, a.v_, c)};
}

// TEST-ONLY: reconstruct a secret in clear.
#if COMPOSITE_FSS_INTERNAL
inline std::uint64_t debug_open(const RingConfig &cfg, const Share &a0, const Share &a1) {
    return ring_add(cfg, a0.value_internal(), a1.value_internal());
}

inline std::uint64_t open_share_pair(const RingConfig &cfg, const Share &a0, const Share &a1) {
    return debug_open(cfg, a0, a1);
}
#else
inline std::uint64_t debug_open(const RingConfig &, const Share &, const Share &) = delete;
inline std::uint64_t open_share_pair(const RingConfig &, const Share &, const Share &) = delete;
#endif

#ifdef COMPOSITE_FSS_ENABLE_OPEN
namespace testing {
// Test/simulation-only open. Do not enable in production builds.
inline std::uint64_t open(const RingConfig &cfg, const Share &a0, const Share &a1) {
    return ring_add(cfg, a0.value_internal(), a1.value_internal());
}
} // namespace testing
#endif

} // namespace cfss

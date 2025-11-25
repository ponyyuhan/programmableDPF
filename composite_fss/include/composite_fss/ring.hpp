#pragma once

#include <cstdint>
#include <cassert>

namespace cfss {

#ifndef COMPOSITE_FSS_INTERNAL
// Default to strict: no internal/raw access unless explicitly enabled.
#define COMPOSITE_FSS_INTERNAL 0
#endif

using u64 = std::uint64_t;

// Simple ring Z_{2^n} for n <= 64 backed by uint64_t with a bitmask.
struct Ring64 {
    u64 modulus_mask; // (1 << n) - 1, or all-ones for n=64.

    explicit Ring64(unsigned n_bits = 64)
        : modulus_mask(n_bits == 64 ? ~0ULL : ((1ULL << n_bits) - 1ULL)) {
        assert(n_bits > 0 && n_bits <= 64);
    }

    inline u64 add(u64 a, u64 b) const {
        return (a + b) & modulus_mask;
    }

    inline u64 sub(u64 a, u64 b) const {
        return (a - b) & modulus_mask;
    }

    inline u64 mul(u64 a, u64 b) const {
        return (a * b) & modulus_mask;
    }

    inline u64 neg(u64 a) const {
        return (-a) & modulus_mask;
    }

    inline u64 from_signed(std::int64_t x) const {
        return static_cast<u64>(x) & modulus_mask;
    }

    inline std::int64_t to_signed(u64 x) const {
        if (modulus_mask == ~0ULL) {
            return static_cast<std::int64_t>(x);
        }
        unsigned n_bits = 64 - __builtin_clzll(modulus_mask);
        u64 sign_bit = 1ULL << (n_bits - 1);
        if (x & sign_bit) {
            u64 ext = ~((1ULL << n_bits) - 1ULL);
            return static_cast<std::int64_t>(x | ext);
        }
        return static_cast<std::int64_t>(x);
    }
};

struct RingConfig;

struct Share {
private:
    int party_;
    u64 v_;

public:
    Share() : party_(-1), v_(0) {}
    Share(int party, u64 v) : party_(party), v_(v) {}

    int party() const { return party_; }

#if COMPOSITE_FSS_INTERNAL
    // Strictly internal/debug; do not use in protocol logic.
    u64 raw_value_unsafe() const { return v_; }
    u64 value_internal() const { return v_; }
#else
    // Deleted in strict builds; any accidental raw access will fail to compile.
    u64 raw_value_unsafe() const = delete;
    u64 value_internal() const = delete;
#endif

    // Internal-only accessor for protocol primitives; not part of the public API.
    friend inline u64 share_value(const Share &s);

    friend struct RingConfig;
    friend Share add(const struct RingConfig &, const Share &, const Share &);
    friend Share sub(const struct RingConfig &, const Share &, const Share &);
    friend Share negate(const struct RingConfig &, const Share &);
    friend Share add_const(const struct RingConfig &, const Share &, u64);
    friend Share mul_const(const struct RingConfig &, const Share &, u64);
};

} // namespace cfss

#pragma once

#include <cstdint>
#include <cassert>

namespace cfss {

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

struct Share {
    int party; // 0 or 1
    u64 v;
};

inline Share make_share(int party, u64 v) {
    return Share{party, v};
}

} // namespace cfss

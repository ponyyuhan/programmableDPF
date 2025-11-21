// ================================================================
// File: src/prg/prg.cpp
// ================================================================

#include "pdpf/prg/prg.hpp"
#include <cstring>
#include <random>
#include <array>

namespace pdpf::prg {

AesCtrPrg::AesCtrPrg(const core::Seed &master_key)
    : master_key_(master_key) {
    // In a production build, initialize AES key schedule / context using master_key_.
}

void AesCtrPrg::expand(const core::Seed &seed,
                       core::Seed &left,
                       core::Seed &right) const {
    // Deterministic fallback PRG: use std::mt19937_64 keyed by master_key_ XOR seed bytes.
    // This is *not* cryptographically secure and should be replaced with AES-CTR or ChaCha20.
    std::uint64_t acc = 0;
    for (auto b : master_key_) { acc = (acc << 5) ^ (acc >> 2) ^ b; }
    for (auto b : seed)       { acc = (acc << 5) ^ (acc >> 2) ^ b; }

    std::seed_seq seq{static_cast<unsigned int>(acc & 0xffffffffu),
                      static_cast<unsigned int>((acc >> 32) & 0xffffffffu)};
    std::mt19937_64 rng(seq);

    auto fill_seed = [&rng](core::Seed &dst) {
        for (std::size_t i = 0; i < dst.size(); ++i) {
            auto v = static_cast<std::uint8_t>(rng() & 0xFFu);
            dst[i] = v;
        }
    };

    fill_seed(left);
    fill_seed(right);
}

} // namespace pdpf::prg

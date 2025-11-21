// ================================================================
// File: include/pdpf/ldc/reed_muller_ldc.hpp
// ================================================================

#pragma once

#include "pdpf/core/types.hpp"
#include <cstdint>
#include <vector>

namespace pdpf::ldc {

/**
 * Parameters for Reed-Muller-style LDC used in Theorem 6. :contentReference[oaicite:19]{index=19}
 *
 * We encode z ∈ Z_p^N to C(z) ∈ Z_p^L with q-query decoding.
 */
struct LdcParams {
    std::uint64_t N = 0;     ///< original domain size
    std::uint64_t p = 0;     ///< prime modulus
    std::uint32_t sigma = 0; ///< independence parameter
    std::uint32_t r = 0;     ///< degree parameter
    std::uint32_t w = 0;     ///< number of variables
    std::uint64_t L = 0;     ///< codeword length
    std::uint64_t q = 0;     ///< number of queries for decoding
};

/**
 * Reed-Muller-like locally decodable code with additive decoding.
 *
 * API follows Lemma 2:
 *   - encode(z)   = C(z)
 *   - encode_unit(x) = C(e_x)
 *   - sample_indices(α) returns Δ ∈ [L]^q with σ-wise independence.
 */
class ReedMullerLdc {
public:
    explicit ReedMullerLdc(const LdcParams &params);

    const LdcParams& params() const { return params_; }

    /**
     * Encode z ∈ Z_p^N to codeword ∈ Z_p^L.
     * z and codeword are vectors of length N and L respectively.
     */
    void encode(const std::vector<std::int64_t> &z,
                std::vector<std::int64_t> &codeword) const;

    /**
     * Encode unit vector e_x (length N) to codeword ∈ Z_p^L.
     */
    void encode_unit(std::uint64_t x,
                     std::vector<std::int64_t> &codeword) const;

    /**
     * Randomized decoding map d: [N] → [L]^q.
     *
     * Returns Δ = (Δ_1,...,Δ_q) used for additive decoding:
     *   sum_{ℓ} C(z)_{Δ_ℓ} = z_α.
     */
    std::vector<std::uint64_t> sample_indices(std::uint64_t alpha) const;

private:
    LdcParams params_;
};

} // namespace pdpf::ldc

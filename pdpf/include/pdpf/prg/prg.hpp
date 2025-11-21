// ================================================================
// File: include/pdpf/prg/prg.hpp
// ================================================================

#pragma once

#include "pdpf/core/types.hpp"
#include <memory>

namespace pdpf::prg {

/**
 * Abstract length-doubling PRG:
 *   G: {0,1}^λ -> {0,1}^{2λ}
 * used both as:
 *  - Seed expander for GGM PPRF (Theorem 3). :contentReference[oaicite:6]{index=6}
 *  - To derive (s, k_PPRF) in PDPF.Gen1 (Figure 1). :contentReference[oaicite:7]{index=7}
 */
class IPrg {
public:
    virtual ~IPrg() = default;

    /**
     * Expand a λ-bit seed into 2λ bits, split as left || right.
     *
     * @param seed  input seed (λ bits)
     * @param left  output seed (λ bits)
     * @param right output seed (λ bits)
     */
    virtual void expand(const core::Seed &seed,
                        core::Seed &left,
                        core::Seed &right) const = 0;
};

/**
 * Example AES-CTR-based PRG.
 *
 * You can implement this using a crypto library (OpenSSL, libsodium, etc.).
 * The implementation should be constant-time.
 */
class AesCtrPrg : public IPrg {
public:
    explicit AesCtrPrg(const core::Seed &master_key);

    void expand(const core::Seed &seed,
                core::Seed &left,
                core::Seed &right) const override;

private:
    core::Seed master_key_;
};

} // namespace pdpf::prg

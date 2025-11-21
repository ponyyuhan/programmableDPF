// ================================================================
// File: include/pdpf/pdpf/pdpf_binary.hpp
// ================================================================

#pragma once

#include "pdpf/core/types.hpp"
#include "pdpf/core/group.hpp"
#include "pdpf/prg/prg.hpp"
#include "pdpf/pprf/pprf.hpp"
#include <memory>
#include <vector>

namespace pdpf::pdpf {

/**
 * Parameters for small-domain binary PDPF (Theorem 4). :contentReference[oaicite:10]{index=10}
 *
 * M is the PPRF input domain size and controls ε ≈ sqrt((N+1)/M).
 */
struct PdpfParams {
    core::SecurityParams sec;
    std::uint64_t        M = 0;
};

/**
 * Offline key k0 = (k*, N, ...)
 *  - k_star is uniform λ-bit seed.
 */
struct OfflineKey {
    core::Seed  k_star{};
    PdpfParams  params{};
};

/**
 * Online key k1 = (kp, s, params).
 *  - kp: punctured PPRF key
 *  - s: shift value (G(k*), Figure 1)
 */
struct OnlineKey {
    pprf::PprfPuncturedKey kp;
    core::Seed             s{};
    PdpfParams             params{};
};

/**
 * Small-domain PDPF over G = Z with payload set G' = {0,1}. :contentReference[oaicite:11]{index=11}
 *
 * API corresponds to:
 *  - Gen0  → gen_offline
 *  - Gen1  → gen_online
 *  - EvalAll0 → eval_all_offline
 *  - EvalAll1 → eval_all_online
 */
class PdpfBinary {
public:
    explicit PdpfBinary(std::shared_ptr<prg::IPrg> prg);

    /**
     * Offline generation Gen0(1^λ, N, Gˆ, Gˆ') → k0.
     * Computes suitable M from sec.epsilon and sec.domain_size_N.
     */
    OfflineKey gen_offline(const core::SecurityParams &sec) const;

    /**
     * Online generation Gen1(k0, α, β) → k1.
     * β ∈ {0,1}.
     *
     * Implements Figure 1:
     *  - Expand k* → (s, k_PPRF)
     *  - Use PPRF over [M] → [N+1]
     *  - For β=1: choose ℓ s.t. Eval + s == α
     *  - For β=0: choose ℓ s.t. Eval + s == N+1
     *  - Puncture at ℓ.
     */
    OnlineKey gen_online(const OfflineKey &k0,
                         std::uint64_t alpha,
                         std::uint8_t beta);

    /**
     * EvalAll0(k0) → Y^(0)[1..N] ∈ Z^N.
     *
     * Y[x] = count of ℓ where PPRF.Eval(k_PPRF, ℓ) + s = x.
     */
    void eval_all_offline(const OfflineKey &k0,
                          std::vector<core::GroupZ::Value> &Y) const;

    /**
     * EvalAll1(k1) → Y^(1)[1..N] ∈ Z^N.
     *
     * Y[x] = - count of ℓ where PPRF.PuncEval(kp, ℓ) + s = x.
     */
    void eval_all_online(const OnlineKey &k1,
                         std::vector<core::GroupZ::Value> &Y) const;

private:
    std::shared_ptr<prg::IPrg> prg_;

    /**
     * Compute M as function of N and ε.
     * Paper suggests roughly M ≈ 0.318·(N+1)/ε^2 empirically. :contentReference[oaicite:12]{index=12}
     */
    std::uint64_t choose_M(const core::SecurityParams &sec) const;

    std::uint64_t seed_to_uint64(const core::Seed &s) const;
};

} // namespace pdpf::pdpf

// ================================================================
// File: include/pdpf/pdpf/pdpf_amplified.hpp
// ================================================================

#pragma once

#include "pdpf/core/types.hpp"
#include "pdpf/ldc/reed_muller_ldc.hpp"
#include "pdpf/pdpf/pdpf_binary.hpp"
#include "pdpf/pdpf/pdpf_group.hpp"
#include "pdpf/prg/prg.hpp"
#include <memory>
#include <vector>

namespace pdpf::pdpf {

/**
 * Offline key for amplified PDPF (Theorem 6). :contentReference[oaicite:21]{index=21}
 *
 * We assume output group is Z_p (for prime p) and we amplify
 * a base PDPF with 1/poly leakage to negligible error.
 */
struct AmplifiedOfflineKey {
    core::Seed        master_seed{}; ///< k* for deriving q base offline keys
    core::SecurityParams sec{};
    ldc::LdcParams    ldc_params{};
    std::uint64_t     prime_p = 0;
};

/**
 * Online key for amplified PDPF:
 *  - Contains q inner OnlineKey's for base PDPF over domain L.
 */
struct AmplifiedOnlineKey {
    std::vector<PdpfGroupOnlineKey> inner_keys; ///< length q
    core::SecurityParams   sec{};
    ldc::LdcParams         ldc_params{};
    std::uint64_t          prime_p = 0;
    std::vector<std::uint64_t> deltas; ///< sampled Δ indices for LDC decoding
};

/**
 * Amplified PDPF (Theorem 6, Figure 2). :contentReference[oaicite:22]{index=22}
 *
 * Construction idea:
 *   - Use LDC C,d with additive decoding.
 *   - For each ℓ in [q], run base PDPF on index Δ_ℓ.
 *   - Aggregate using C(e_x) and base EvalAll.
 */
class PdpfAmplified {
public:
    PdpfAmplified(std::shared_ptr<prg::IPrg> prg,
                  const ldc::LdcParams &ldc_params,
                  std::uint64_t prime_p);

    AmplifiedOfflineKey gen_offline(const core::SecurityParams &sec);

    AmplifiedOnlineKey gen_online(const AmplifiedOfflineKey &k0,
                                  std::uint64_t alpha,
                                  std::int64_t beta);

    /**
     * Eval0(x): share for input x ∈ [N] using amplified construction.
     *
     * Returns y ∈ Z_p.
     */
    std::int64_t eval_offline(const AmplifiedOfflineKey &k0,
                              const AmplifiedOnlineKey &k1,
                              std::uint64_t x) const;

    /**
     * Eval1(x): share for input x ∈ [N] using amplified construction.
     *
     * Returns y ∈ Z_p.
     */
    std::int64_t eval_online(const AmplifiedOfflineKey &k0,
                             const AmplifiedOnlineKey &k1,
                             std::uint64_t x) const;

    /**
     * Optional: full-domain evaluation, if you want EvalAll.
     */
    void eval_all_offline(const AmplifiedOfflineKey &k0,
                          const AmplifiedOnlineKey &k1,
                          std::vector<std::int64_t> &Y0) const;

    void eval_all_online(const AmplifiedOfflineKey &k0,
                         const AmplifiedOnlineKey &k1,
                         std::vector<std::int64_t> &Y1) const;

private:
    std::shared_ptr<prg::IPrg> prg_;
    PdpfGroup                  group_pdpf_;   ///< base PDPF over Z_p on domain L
    ldc::ReedMullerLdc         ldc_;
    std::uint64_t              prime_p_;

    core::Seed derive_inner_seed(const core::Seed &master,
                                 std::uint64_t index) const;

    std::int64_t mod_p(std::int64_t x) const;
};

} // namespace pdpf::pdpf

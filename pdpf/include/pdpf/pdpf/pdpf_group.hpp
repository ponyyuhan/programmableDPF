// ================================================================
// File: include/pdpf/pdpf/pdpf_group.hpp
// ================================================================

#pragma once

#include "pdpf/core/types.hpp"
#include "pdpf/core/group.hpp"
#include "pdpf/pdpf/pdpf_binary.hpp"
#include <memory>
#include <vector>

namespace pdpf::pdpf {

/**
 * Offline key for PDPF over general group G and allowed subset G'.
 * 
 * We implement Theorem 5 via bit-decomposition + group decomposition: :contentReference[oaicite:16]{index=16}
 *  - Represent β ∈ G' as vector of bits across components.
 *  - Maintain one PdpfBinary offline key per bit.
 */
struct PdpfGroupOfflineKey {
    std::vector<OfflineKey> bit_offline_keys; // one per bit of representation
    core::GroupDescriptor   group_desc;       // describes G
    core::SecurityParams    sec;             // same for all bits
};

/**
 * Online key for PDPF over G.
 */
struct PdpfGroupOnlineKey {
    std::vector<OnlineKey>  bit_online_keys;
    core::GroupDescriptor   group_desc;
    core::SecurityParams    sec;
};

/**
 * PDPF over arbitrary finite abelian group G with allowed payload subset G'.
 * This wraps multiple binary PDPFs.
 */
class PdpfGroup {
public:
    explicit PdpfGroup(std::shared_ptr<prg::IPrg> prg);

    /**
     * Offline generation for group-valued PDPF.
     *
     * @param sec       security params (N, λ, ε)
     * @param group     group descriptor G (Z_{q1} × ... × Z_{qℓ})
     * @param payload_bits number of bits needed to encode elements of G' (|G'| ≤ 2^{payload_bits}).
     */
    PdpfGroupOfflineKey gen_offline(const core::SecurityParams &sec,
                                    const core::GroupDescriptor &group,
                                    std::size_t payload_bits) const;

    /**
     * Encode payload β ∈ G into payload_bits binary components.
     */
    std::vector<std::uint8_t> encode_payload(const core::GroupDescriptor &group,
                                             const core::GroupElement &beta,
                                             std::size_t payload_bits) const;

    /**
     * Decode β from bit-wise shares:
     *  - For each x, we reconstruct β(x) by combining reconstruction from binary PDPFs.
     */
    core::GroupElement decode_payload(const core::GroupDescriptor &group,
                                      const std::vector<std::int64_t> &bit_values,
                                      std::size_t payload_bits) const;

    /**
     * Online generation for group PDPF:
     *  - Given α, β ∈ G', produce group-valued online key.
     */
    PdpfGroupOnlineKey gen_online(const PdpfGroupOfflineKey &k0,
                                  std::uint64_t alpha,
                                  const core::GroupElement &beta);

    /**
     * EvalAll0: offline share for all x ∈ [N], group-valued.
     *  - Internally, uses multiple PdpfBinary::eval_all_offline.
     */
    void eval_all_offline(const PdpfGroupOfflineKey &k0,
                          std::vector<core::GroupElement> &Y) const;

    /**
     * EvalAll1: online share for all x ∈ [N], group-valued.
     */
    void eval_all_online(const PdpfGroupOnlineKey &k1,
                         std::vector<core::GroupElement> &Y) const;

    // Convenience point evaluations (used by composite_fss PdpfEngineFull).
    core::GroupElement eval_point_offline(const PdpfGroupOfflineKey &k0,
                                          std::uint64_t x) const;

    core::GroupElement eval_point_online(const PdpfGroupOnlineKey &k1,
                                         std::uint64_t x) const;

private:
    std::shared_ptr<prg::IPrg> prg_;
    PdpfBinary                 base_pdpf_; // reused for all bits

    std::size_t infer_payload_bits(const core::GroupDescriptor &group) const;
};

} // namespace pdpf::pdpf

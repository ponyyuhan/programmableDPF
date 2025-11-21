// ================================================================
// File: src/pdpf/pdpf_group.cpp
// ================================================================

#include "pdpf/pdpf/pdpf_group.hpp"
#include <stdexcept>

namespace pdpf::pdpf {

PdpfGroup::PdpfGroup(std::shared_ptr<prg::IPrg> prg)
    : prg_(std::move(prg)),
      base_pdpf_(prg_) {}

std::size_t PdpfGroup::infer_payload_bits(const core::GroupDescriptor &group) const {
    if (group.moduli.empty()) {
        // G = Z, choose some upper bound externally.
        throw std::runtime_error("PdpfGroup::infer_payload_bits: ambiguous for Z");
    }
    std::size_t bits = 0;
    for (auto q : group.moduli) {
        if (q == 0) {
            throw std::runtime_error("PdpfGroup::infer_payload_bits: infinite component present");
        }
        std::size_t b = 0;
        std::uint64_t x = q - 1;
        while (x > 0) { x >>= 1; ++b; }
        bits += b;
    }
    return bits;
}

PdpfGroupOfflineKey PdpfGroup::gen_offline(const core::SecurityParams &sec,
                                           const core::GroupDescriptor &group,
                                           std::size_t payload_bits) {
    PdpfGroupOfflineKey k0;
    k0.group_desc = group;
    k0.sec        = sec;

    if (payload_bits == 0) {
        payload_bits = infer_payload_bits(group);
    }

    k0.bit_offline_keys.clear();
    k0.bit_offline_keys.reserve(payload_bits);
    for (std::size_t i = 0; i < payload_bits; ++i) {
        k0.bit_offline_keys.push_back(base_pdpf_.gen_offline(sec));
    }
    return k0;
}

std::vector<std::uint8_t> PdpfGroup::encode_payload(const core::GroupDescriptor &group,
                                                    const core::GroupElement &beta,
                                                    std::size_t payload_bits) const {
    if (payload_bits == 0) {
        throw std::invalid_argument("encode_payload: payload_bits = 0");
    }
    if (group.arity() == 0 || beta.size() == 0) {
        throw std::invalid_argument("encode_payload: empty group or payload");
    }
    std::uint64_t modulus = group.moduli.empty() ? 0 : group.moduli[0];
    std::int64_t value = beta[0];
    if (modulus > 0) {
        value %= static_cast<std::int64_t>(modulus);
        if (value < 0) value += static_cast<std::int64_t>(modulus);
    }

    std::vector<std::uint8_t> bits(payload_bits, 0);
    for (std::size_t i = 0; i < payload_bits; ++i) {
        bits[i] = static_cast<std::uint8_t>((value >> i) & 0x1);
    }
    return bits;
}

core::GroupElement PdpfGroup::decode_payload(const core::GroupDescriptor &group,
                                             const std::vector<std::int64_t> &bit_values,
                                             std::size_t payload_bits) const {
    if (payload_bits == 0) {
        throw std::invalid_argument("decode_payload: payload_bits = 0");
    }
    std::int64_t value = 0;
    for (std::size_t i = 0; i < payload_bits && i < bit_values.size() && i < 63; ++i) {
        // bit_values can be negative for online share; propagate sign.
        value += bit_values[i] * static_cast<std::int64_t>(1ULL << i);
    }

    core::GroupElement out = core::group_zero(group);
    if (!group.moduli.empty() && group.moduli[0] > 0) {
        std::int64_t mod = static_cast<std::int64_t>(group.moduli[0]);
        value %= mod;
        if (value < 0) value += mod;
    }
    if (out.empty()) {
        out.push_back(value);
    } else {
        out[0] = value;
    }
    return out;
}

PdpfGroupOnlineKey PdpfGroup::gen_online(const PdpfGroupOfflineKey &k0,
                                         std::uint64_t alpha,
                                         const core::GroupElement &beta) {
    const std::size_t payload_bits = k0.bit_offline_keys.size();
    auto bits = encode_payload(k0.group_desc, beta, payload_bits);
    if (bits.size() != payload_bits) {
        throw std::runtime_error("PdpfGroup::gen_online: bit encoding size mismatch");
    }

    PdpfGroupOnlineKey k1;
    k1.group_desc = k0.group_desc;
    k1.sec        = k0.sec;
    k1.bit_online_keys.reserve(payload_bits);

    for (std::size_t i = 0; i < payload_bits; ++i) {
        std::uint8_t b = bits[i] & 1u;
        const OfflineKey &bit_off = k0.bit_offline_keys[i];
        k1.bit_online_keys.push_back(base_pdpf_.gen_online(bit_off, alpha, b));
    }

    return k1;
}

void PdpfGroup::eval_all_offline(const PdpfGroupOfflineKey &k0,
                                 std::vector<core::GroupElement> &Y) const {
    const std::uint64_t N = k0.sec.domain_size_N;
    const std::size_t payload_bits = k0.bit_offline_keys.size();

    // Y_bit[i][x] = share for bit i at position x.
    std::vector<std::vector<core::GroupZ::Value>> Y_bits(payload_bits);

    for (std::size_t i = 0; i < payload_bits; ++i) {
        base_pdpf_.eval_all_offline(k0.bit_offline_keys[i], Y_bits[i]);
    }

    Y.assign(N, core::group_zero(k0.group_desc));

    for (std::uint64_t x = 0; x < N; ++x) {
        std::vector<std::int64_t> bit_vals(payload_bits, 0);
        for (std::size_t i = 0; i < payload_bits; ++i) {
            bit_vals[i] = Y_bits[i][x];
        }
        Y[x] = decode_payload(k0.group_desc, bit_vals, payload_bits);
    }
}

void PdpfGroup::eval_all_online(const PdpfGroupOnlineKey &k1,
                                std::vector<core::GroupElement> &Y) const {
    const std::uint64_t N = k1.sec.domain_size_N;
    const std::size_t payload_bits = k1.bit_online_keys.size();

    std::vector<std::vector<core::GroupZ::Value>> Y_bits(payload_bits);

    for (std::size_t i = 0; i < payload_bits; ++i) {
        base_pdpf_.eval_all_online(k1.bit_online_keys[i], Y_bits[i]);
    }

    Y.assign(N, core::group_zero(k1.group_desc));

    for (std::uint64_t x = 0; x < N; ++x) {
        std::vector<std::int64_t> bit_vals(payload_bits, 0);
        for (std::size_t i = 0; i < payload_bits; ++i) {
            bit_vals[i] = Y_bits[i][x];
        }
        Y[x] = decode_payload(k1.group_desc, bit_vals, payload_bits);
    }
}

} // namespace pdpf::pdpf

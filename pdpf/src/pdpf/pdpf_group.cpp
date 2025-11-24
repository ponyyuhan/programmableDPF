// ================================================================
// File: src/pdpf/pdpf_group.cpp
// ================================================================

#include "pdpf/pdpf/pdpf_group.hpp"
#include <stdexcept>

namespace pdpf::pdpf {

PdpfGroup::PdpfGroup(std::shared_ptr<prg::IPrg> prg)
    : prg_(std::move(prg)),
      base_pdpf_(prg_) {}

namespace {

struct BitPosition {
    std::size_t component;
    std::size_t bit_index;
};

static std::vector<BitPosition> build_bit_layout(const core::GroupDescriptor &group) {
    if (group.moduli.empty()) {
        throw std::runtime_error("group descriptor is empty");
    }
    std::vector<BitPosition> layout;
    for (std::size_t i = 0; i < group.moduli.size(); ++i) {
        auto q = group.moduli[i];
        if (q == 0) {
            throw std::runtime_error("infinite group component not supported");
        }
        std::size_t bits = 0;
        std::uint64_t x = q - 1;
        while (x > 0) { x >>= 1; ++bits; }
        for (std::size_t b = 0; b < bits; ++b) {
            layout.push_back(BitPosition{i, b});
        }
    }
    return layout;
}

} // namespace

std::size_t PdpfGroup::infer_payload_bits(const core::GroupDescriptor &group) const {
    auto layout = build_bit_layout(group);
    return layout.size();
}

PdpfGroupOfflineKey PdpfGroup::gen_offline(const core::SecurityParams &sec,
                                           const core::GroupDescriptor &group,
                                           std::size_t payload_bits) const {
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
    auto layout = build_bit_layout(group);
    if (payload_bits == 0) payload_bits = layout.size();
    if (layout.size() != payload_bits) {
        throw std::runtime_error("encode_payload: payload_bits mismatch");
    }
    if (beta.size() != group.arity()) {
        throw std::invalid_argument("encode_payload: beta size mismatch");
    }

    std::vector<std::uint8_t> bits(payload_bits, 0);
    for (std::size_t k = 0; k < payload_bits; ++k) {
        auto pos = layout[k];
        std::uint64_t q = group.moduli[pos.component];
        std::int64_t v = beta[pos.component];
        v %= static_cast<std::int64_t>(q);
        if (v < 0) v += static_cast<std::int64_t>(q);
        bits[k] = static_cast<std::uint8_t>((static_cast<std::uint64_t>(v) >> pos.bit_index) & 1ULL);
    }
    return bits;
}

core::GroupElement PdpfGroup::decode_payload(const core::GroupDescriptor &group,
                                             const std::vector<std::int64_t> &bit_values,
                                             std::size_t payload_bits) const {
    auto layout = build_bit_layout(group);
    if (payload_bits == 0) payload_bits = layout.size();
    if (layout.size() != payload_bits || bit_values.size() != payload_bits) {
        throw std::runtime_error("decode_payload: payload_bits mismatch");
    }

    core::GroupElement out = core::group_zero(group);
    out.resize(group.arity(), 0);

    for (std::size_t k = 0; k < payload_bits; ++k) {
        auto pos = layout[k];
        std::int64_t contrib = bit_values[k] * (static_cast<std::int64_t>(1ULL << pos.bit_index));
        auto q = group.moduli[pos.component];
        if (q == 0) {
            out[pos.component] += contrib;
        } else {
            std::int64_t mod = static_cast<std::int64_t>(q);
            std::int64_t v = out[pos.component] + contrib;
            v %= mod;
            if (v < 0) v += mod;
            out[pos.component] = v;
        }
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

core::GroupElement PdpfGroup::eval_point_offline(const PdpfGroupOfflineKey &k0,
                                                 std::uint64_t x) const {
    const std::size_t payload_bits = k0.bit_offline_keys.size();
    std::vector<std::int64_t> bit_vals(payload_bits, 0);
    for (std::size_t i = 0; i < payload_bits; ++i) {
        std::vector<core::GroupZ::Value> Ybit;
        base_pdpf_.eval_all_offline(k0.bit_offline_keys[i], Ybit);
        if (x < Ybit.size()) {
            bit_vals[i] = Ybit[static_cast<std::size_t>(x)];
        }
    }
    return decode_payload(k0.group_desc, bit_vals, payload_bits);
}

core::GroupElement PdpfGroup::eval_point_online(const PdpfGroupOnlineKey &k1,
                                                std::uint64_t x) const {
    const std::size_t payload_bits = k1.bit_online_keys.size();
    std::vector<std::int64_t> bit_vals(payload_bits, 0);
    for (std::size_t i = 0; i < payload_bits; ++i) {
        std::vector<core::GroupZ::Value> Ybit;
        base_pdpf_.eval_all_online(k1.bit_online_keys[i], Ybit);
        if (x < Ybit.size()) {
            bit_vals[i] = Ybit[static_cast<std::size_t>(x)];
        }
    }
    return decode_payload(k1.group_desc, bit_vals, payload_bits);
}

} // namespace pdpf::pdpf

// ================================================================
// File: src/pdpf/pdpf/pdpf_lut.cpp
// ================================================================

#include "pdpf/pdpf/pdpf_lut.hpp"

namespace pdpf::pdpf {

PdpfLut::PdpfLut(std::shared_ptr<prg::IPrg> prg)
    : group_pdpf_(std::make_shared<PdpfGroup>(std::move(prg))) {}

PdpfLutOfflineKey PdpfLut::gen_offline(const core::SecurityParams &sec,
                                       const core::GroupDescriptor &group,
                                       const std::vector<std::uint64_t> &table,
                                       std::size_t in_bits,
                                       std::size_t out_bits) const {
    PdpfLutOfflineKey k0;
    k0.sec = sec;
    k0.group = group;
    k0.in_bits = in_bits;
    k0.out_bits = out_bits;
    for (std::size_t i = 0; i < table.size(); ++i) {
        std::uint64_t v = table[i] & ((out_bits >= 64) ? ~0ULL : ((1ULL << out_bits) - 1ULL));
        if (v == 0) continue;
        auto off = group_pdpf_->gen_offline(sec, group, 0);
        core::GroupElement beta = {static_cast<std::int64_t>(v % group.moduli[0])};
        auto on = group_pdpf_->gen_online(off, i, beta);
        k0.indices.push_back(i);
        k0.entries.push_back(std::move(off));
    }
    return k0;
}

PdpfLutOnlineKey PdpfLut::gen_online(const PdpfLutOfflineKey &k0) const {
    PdpfLutOnlineKey k1;
    k1.sec = k0.sec;
    k1.group = k0.group;
    k1.in_bits = k0.in_bits;
    k1.out_bits = k0.out_bits;
    k1.indices = k0.indices;
    for (std::size_t i = 0; i < k0.entries.size(); ++i) {
        auto on = group_pdpf_->gen_online(k0.entries[i], k0.indices[i],
                                          core::GroupElement{1}); // beta ignored
        k1.entries.push_back(std::move(on));
    }
    return k1;
}

std::int64_t PdpfLut::eval_point_offline(const PdpfLutOfflineKey &k0,
                                         std::uint64_t x) const {
    std::int64_t acc = 0;
    for (std::size_t i = 0; i < k0.entries.size(); ++i) {
        if (x >= k0.sec.domain_size_N) break;
        auto ge = group_pdpf_->eval_point_offline(k0.entries[i], x);
        acc += ge[0];
    }
    return acc;
}

std::int64_t PdpfLut::eval_point_online(const PdpfLutOnlineKey &k1,
                                        std::uint64_t x) const {
    std::int64_t acc = 0;
    for (std::size_t i = 0; i < k1.entries.size(); ++i) {
        if (x >= k1.sec.domain_size_N) break;
        auto ge = group_pdpf_->eval_point_online(k1.entries[i], x);
        acc += ge[0];
    }
    return acc;
}

} // namespace pdpf::pdpf

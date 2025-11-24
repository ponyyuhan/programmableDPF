// ================================================================
// File: include/pdpf/pdpf/pdpf_lut.hpp
// A naive multipoint/LUT PDPF built from sums of group PDPFs.
// ================================================================

#pragma once

#include "pdpf/core/types.hpp"
#include "pdpf/core/group.hpp"
#include "pdpf/pdpf/pdpf_group.hpp"
#include <vector>
#include <memory>

namespace pdpf::pdpf {

struct PdpfLutOfflineKey {
    core::SecurityParams sec;
    core::GroupDescriptor group;
    std::size_t in_bits;
    std::size_t out_bits;
    // One group PDPF offline key per non-zero entry.
    std::vector<std::size_t> indices;
    std::vector<PdpfGroupOfflineKey> entries;
};

struct PdpfLutOnlineKey {
    core::SecurityParams sec;
    core::GroupDescriptor group;
    std::size_t in_bits;
    std::size_t out_bits;
    std::vector<std::size_t> indices;
    std::vector<PdpfGroupOnlineKey> entries;
};

class PdpfLut {
public:
    explicit PdpfLut(std::shared_ptr<prg::IPrg> prg);

    // Build keys for table: table[x] in Z_mod
    PdpfLutOfflineKey gen_offline(const core::SecurityParams &sec,
                                  const core::GroupDescriptor &group,
                                  const std::vector<std::uint64_t> &table,
                                  std::size_t in_bits,
                                  std::size_t out_bits) const;

    PdpfLutOnlineKey gen_online(const PdpfLutOfflineKey &k0) const;

    std::int64_t eval_point_offline(const PdpfLutOfflineKey &k0,
                                    std::uint64_t x) const;

    std::int64_t eval_point_online(const PdpfLutOnlineKey &k1,
                                   std::uint64_t x) const;

private:
    std::shared_ptr<PdpfGroup> group_pdpf_;
};

} // namespace pdpf::pdpf

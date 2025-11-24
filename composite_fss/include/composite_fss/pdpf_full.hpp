#pragma once

#include "pdpf.hpp"
#include "pdpf_adapter.hpp"

namespace cfss {

// Placeholder "full" engine that currently delegates to the adapter.
// This allows swapping to a true pdpf-backed implementation later without
// touching gate code.
class PdpfEngineFull : public PdpfEngine {
public:
    PdpfEngineFull(std::size_t cap_bits = 32, std::uint64_t seed = 0xA5A5)
        : adapter_(cap_bits, seed) {}

    PdpfProgramId make_lut_program(const LutProgramDesc &desc,
                                   const std::vector<std::uint64_t> &table_flat) override {
        return adapter_.make_lut_program(desc, table_flat);
    }

    PdpfProgramId make_cmp_program(const CmpProgramDesc &desc,
                                   const std::vector<std::uint64_t> &table_flat) override {
        return adapter_.make_cmp_program(desc, table_flat);
    }

    void eval_share(PdpfProgramId program,
                    int party,
                    std::uint64_t masked_x,
                    std::vector<std::uint64_t> &out_words) const override {
        adapter_.eval_share(program, party, masked_x, out_words);
    }

    void eval_share_batch(PdpfProgramId program,
                          int party,
                          const std::vector<std::uint64_t> &masked_xs,
                          std::vector<std::vector<std::uint64_t>> &out_batch) const override {
        adapter_.eval_share_batch(program, party, masked_xs, out_batch);
    }

    LutProgramDesc lookup_lut_desc(PdpfProgramId program) const override {
        return adapter_.lookup_lut_desc(program);
    }

private:
    PdpfEngineAdapter adapter_;
};

} // namespace cfss

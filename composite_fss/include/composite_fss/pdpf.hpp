#pragma once

#include <cstdint>
#include <vector>

namespace cfss {

using PdpfProgramId = std::uint32_t;

struct LutProgramDesc {
    unsigned domain_bits = 0;   // m_tau in the formalization
    unsigned output_words = 0;  // number of u64 words per party
};

struct CmpProgramDesc {
    unsigned domain_bits = 0;
};

// Note: the concrete backend may still be naive (table-based), but the API is
// multi-output and descriptor-driven so gates can pre-allocate outputs.
class PdpfEngine {
public:
    virtual ~PdpfEngine() = default;

    // Build a LUT-backed PDPF program. The table is provided in row-major
    // flattened form: table[word + output_words * idx] corresponds to entry
    // idx for output word "word".
    virtual PdpfProgramId make_lut_program(const LutProgramDesc &desc,
                                           const std::vector<std::uint64_t> &table_flat) = 0;

    virtual PdpfProgramId make_cmp_program(const CmpProgramDesc &desc,
                                           const std::vector<std::uint64_t> &table_flat) = 0;

    // Evaluate a PDPF program on masked_x and write output_words entries into
    // out_words. The size of out_words must match the program descriptor.
    virtual void eval_share(PdpfProgramId program,
                            int party,
                            std::uint64_t masked_x,
                            std::vector<std::uint64_t> &out_words) const = 0;

    // Optional batched evaluation; default implementation loops.
    virtual void eval_share_batch(PdpfProgramId program,
                                  int party,
                                  const std::vector<std::uint64_t> &masked_xs,
                                  std::vector<std::vector<std::uint64_t>> &out_batch) const {
        out_batch.resize(masked_xs.size());
        for (std::size_t i = 0; i < masked_xs.size(); ++i) {
            eval_share(program, party, masked_xs[i], out_batch[i]);
        }
    }

    virtual LutProgramDesc lookup_lut_desc(PdpfProgramId program) const = 0;
};

} // namespace cfss

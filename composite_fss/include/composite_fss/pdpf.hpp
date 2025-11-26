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

    // Evaluate a batch of masked inputs for a single program. The caller must
    // provide an output buffer of at least n_inputs * output_words entries.
    virtual void eval_share_batch(PdpfProgramId program,
                                  int party,
                                  const std::uint64_t *masked_xs,
                                  std::size_t n_inputs,
                                  std::uint64_t *flat_out) const {
        if (masked_xs == nullptr || flat_out == nullptr || n_inputs == 0) return;
        auto desc = lookup_lut_desc(program);
        std::size_t out_words = desc.output_words ? desc.output_words : 1;
        std::vector<std::uint64_t> tmp(out_words);
        for (std::size_t i = 0; i < n_inputs; ++i) {
            eval_share(program, party, masked_xs[i], tmp);
            for (std::size_t w = 0; w < out_words; ++w) {
                flat_out[i * out_words + w] = tmp[w];
            }
        }
    }

    // Optional batched evaluation; default implementation loops.
    virtual void eval_share_batch(PdpfProgramId program,
                                  int party,
                                  const std::vector<std::uint64_t> &masked_xs,
                                  std::vector<std::vector<std::uint64_t>> &out_batch) const {
        if (masked_xs.empty()) return;
        auto desc = lookup_lut_desc(program);
        std::size_t out_words = desc.output_words ? desc.output_words : 1;
        out_batch.assign(masked_xs.size(), std::vector<std::uint64_t>(out_words, 0));
        std::vector<std::uint64_t> flat(masked_xs.size() * out_words);
        eval_share_batch(program, party, masked_xs.data(), masked_xs.size(), flat.data());
        for (std::size_t i = 0; i < masked_xs.size(); ++i) {
            for (std::size_t w = 0; w < out_words; ++w) {
                out_batch[i][w] = flat[i * out_words + w];
            }
        }
    }

    // Flattened batch evaluation: writes outputs consecutively into flat_out.
    // flat_out will be resized to masked_xs.size() * output_words.
    virtual void eval_share_batch(PdpfProgramId program,
                                  int party,
                                  const std::vector<std::uint64_t> &masked_xs,
                                  std::vector<std::uint64_t> &flat_out) const {
        auto desc = lookup_lut_desc(program);
        std::size_t out_words = desc.output_words ? desc.output_words : 1;
        flat_out.assign(masked_xs.size() * out_words, 0);
        if (!masked_xs.empty()) {
            eval_share_batch(program, party, masked_xs.data(), masked_xs.size(), flat_out.data());
        }
    }

    virtual LutProgramDesc lookup_lut_desc(PdpfProgramId program) const = 0;
};

} // namespace cfss

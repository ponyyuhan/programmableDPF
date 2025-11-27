#pragma once

#include <cstdint>
#include <stdexcept>
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

// Predicate primitives used by the structured SUF backend.
struct PredicateSpec {
    enum class Kind {
        LT_CONST,
        LT_MOD,
        MSB,
        MSB_SHIFT,
        INTERVAL_INDEX,
    };
    Kind kind = Kind::LT_CONST;
    std::uint64_t param = 0; // β for LT_CONST, γ for LT_MOD, shift for MSB_SHIFT
    unsigned f = 0;          // number of low bits for LT_MOD
};

struct PredProgramDesc {
    unsigned domain_bits = 0;
    unsigned num_preds = 0;
    std::uint64_t r_in = 0; // mask to unmask \hat{x} = x + r_in
};

inline std::uint64_t mask_for_bits(unsigned n_bits) {
    return (n_bits >= 64) ? ~0ULL : ((1ULL << n_bits) - 1ULL);
}

inline std::uint64_t unmask_input(std::uint64_t masked_x, unsigned n_bits, std::uint64_t r_in) {
    if (n_bits >= 64) {
        return masked_x - r_in;
    }
    std::uint64_t mask = mask_for_bits(n_bits);
    std::uint64_t modulus = (1ULL << n_bits);
    return (masked_x + modulus - (r_in & mask)) & mask;
}

inline bool eval_primitive_pred(const PredicateSpec &p,
                                std::uint64_t masked_x,
                                unsigned n_bits,
                                std::uint64_t r_in) {
    std::uint64_t x = unmask_input(masked_x, n_bits, r_in);
    switch (p.kind) {
        case PredicateSpec::Kind::LT_CONST:
            return x < p.param;
        case PredicateSpec::Kind::LT_MOD: {
            std::uint64_t m = (p.f >= 64) ? ~0ULL : ((1ULL << p.f) - 1ULL);
            return (x & m) < p.param;
        }
        case PredicateSpec::Kind::MSB:
            return (x >> (n_bits - 1)) & 1ULL;
        case PredicateSpec::Kind::MSB_SHIFT: {
            std::uint64_t y = x + p.param;
            return (y >> (n_bits - 1)) & 1ULL;
        }
        case PredicateSpec::Kind::INTERVAL_INDEX:
            return false;
    }
    return false;
}

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

    // Structured predicate backend (default LUT fallback).
    virtual PdpfProgramId make_predicate_program(const PredProgramDesc &desc,
                                                 const std::vector<PredicateSpec> &specs) {
        return make_predicate_program_via_lut(desc, specs);
    }

    virtual void eval_predicate_share(PdpfProgramId program,
                                      int party,
                                      std::uint64_t masked_x,
                                      std::vector<std::uint64_t> &out_words) const {
        eval_share(program, party, masked_x, out_words);
    }

    virtual LutProgramDesc lookup_lut_desc(PdpfProgramId program) const = 0;

protected:
    PdpfProgramId make_predicate_program_via_lut(const PredProgramDesc &desc,
                                                 const std::vector<PredicateSpec> &specs) {
        if (desc.num_preds == 0) {
            throw std::runtime_error("make_predicate_program: no predicates provided");
        }
        if (desc.domain_bits >= 31) {
            throw std::runtime_error("make_predicate_program: LUT fallback only supports domain_bits <= 30");
        }
        std::size_t domain_size = 1ULL << desc.domain_bits;
        unsigned bool_words = (desc.num_preds + 63) / 64;
        std::vector<std::uint64_t> table_flat(domain_size * bool_words, 0);

        for (std::size_t x = 0; x < domain_size; ++x) {
            for (unsigned p = 0; p < desc.num_preds; ++p) {
                bool bit = eval_primitive_pred(specs[p],
                                               static_cast<std::uint64_t>(x),
                                               desc.domain_bits,
                                               desc.r_in);
                if (!bit) continue;
                std::size_t base = x * bool_words;
                std::size_t word_idx = p / 64;
                std::size_t bit_off = p % 64;
                table_flat[base + word_idx] |= (1ULL << bit_off);
            }
        }

        LutProgramDesc lut_desc;
        lut_desc.domain_bits = desc.domain_bits;
        lut_desc.output_words = bool_words;
        return make_lut_program(lut_desc, table_flat);
    }
};

} // namespace cfss

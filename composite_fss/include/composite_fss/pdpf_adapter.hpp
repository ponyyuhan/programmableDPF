#pragma once

#include "pdpf.hpp"
#include <functional>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace cfss {

class PdpfEngineAdapter : public PdpfEngine {
public:
    PdpfEngineAdapter(std::size_t n_bits_cap = 32, std::uint64_t seed = 0xA5A5)
        : n_bits_cap_(n_bits_cap), seed_(seed) {}

    PdpfProgramId make_lut_program(const LutProgramDesc &desc,
                                   const std::vector<std::uint64_t> &table_flat) override {
        if (desc.domain_bits > n_bits_cap_) {
            throw std::runtime_error("adapter: in_bits too large");
        }
        std::size_t domain_size = (desc.domain_bits == 64)
                                      ? (table_flat.size() / (desc.output_words ? desc.output_words : 1))
                                      : (1ULL << desc.domain_bits);
        if (table_flat.size() != domain_size * desc.output_words) {
            throw std::runtime_error("adapter: table size mismatch");
        }
        ProgramData data;
        data.desc = desc;
        data.table = table_flat;
        PdpfProgramId id = static_cast<PdpfProgramId>(programs_.size());
        programs_.push_back(std::move(data));
        return id;
    }

    PdpfProgramId make_cmp_program(const CmpProgramDesc &desc,
                                   const std::vector<std::uint64_t> &table_flat) override {
        LutProgramDesc lut_desc{desc.domain_bits, 1};
        return make_lut_program(lut_desc, table_flat);
    }

    void eval_share(PdpfProgramId program,
                    int party,
                    std::uint64_t x,
                    std::vector<std::uint64_t> &out_words) const override {
        if (program >= programs_.size()) {
            throw std::runtime_error("adapter: invalid program id");
        }
        const auto &data = programs_[program];
        if (out_words.size() != data.desc.output_words) {
            out_words.resize(data.desc.output_words);
        }
        std::uint64_t mask = (data.desc.domain_bits == 64) ? ~0ULL : ((1ULL << data.desc.domain_bits) - 1ULL);
        std::size_t idx = static_cast<std::size_t>(x & mask);
        if (data.desc.domain_bits < 64) {
            std::size_t dom = 1ULL << data.desc.domain_bits;
            if (idx >= dom) {
                idx %= dom;
            }
        } else if (idx >= (data.table.size() / data.desc.output_words)) {
            idx %= (data.table.size() / data.desc.output_words);
        }
        std::uint64_t share_mask = mask;
        for (unsigned w = 0; w < data.desc.output_words; ++w) {
            std::uint64_t val = data.table[idx * data.desc.output_words + w];
            std::uint64_t val_masked = val & share_mask;
            std::uint64_t r = prg(idx, w) & share_mask;
            if (party == 0) {
                out_words[w] = r;
            } else {
                out_words[w] = (val_masked - r) & share_mask;
            }
        }
    }

    void eval_share_batch(PdpfProgramId program,
                          int party,
                          const std::vector<std::uint64_t> &masked_xs,
                          std::vector<std::vector<std::uint64_t>> &out_batch) const override {
        out_batch.resize(masked_xs.size());
        for (std::size_t i = 0; i < masked_xs.size(); ++i) {
            eval_share(program, party, masked_xs[i], out_batch[i]);
        }
    }

    LutProgramDesc lookup_lut_desc(PdpfProgramId program) const override {
        if (program >= programs_.size()) return LutProgramDesc{};
        return programs_[program].desc;
    }

private:
    struct ProgramData {
        LutProgramDesc desc;
        std::vector<std::uint64_t> table;
    };

    std::size_t n_bits_cap_;
    std::uint64_t seed_;
    std::vector<ProgramData> programs_;

    std::uint64_t prg(std::size_t idx, unsigned word) const {
        std::uint64_t s = seed_ ^ static_cast<std::uint64_t>(idx * 0x9E3779B97F4A7C15ULL + word * 0xD1B54A32D192ED03ULL);
        std::mt19937_64 gen(s);
        return gen();
    }
};

} // namespace cfss

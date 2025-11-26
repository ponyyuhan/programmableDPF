#pragma once

#include "pdpf.hpp"
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <unordered_map>

#ifdef COMPOSITE_FSS_CUDA
#include "../../src/gpu/pdpf_engine_cuda.hpp"
#endif

namespace cfss {

enum class EngineBackend {
    CPU,
#ifdef COMPOSITE_FSS_CUDA
    CUDA,
#endif
};

inline EngineBackend select_backend_from_env() {
#ifdef COMPOSITE_FSS_CUDA
    const char *env = std::getenv("COMPOSITE_FSS_USE_GPU");
    if (env && std::string(env) == "1") return EngineBackend::CUDA;
#endif
    return EngineBackend::CPU;
}

class PdpfEngineAdapter : public PdpfEngine {
public:
    PdpfEngineAdapter(std::size_t n_bits_cap = 32,
                      std::uint64_t seed = 0xA5A5,
                      EngineBackend backend = EngineBackend::CPU)
        : n_bits_cap_(n_bits_cap),
          seed_(seed),
          backend_(backend) {
#ifndef COMPOSITE_FSS_CUDA
        backend_ = EngineBackend::CPU;
#else
        if (backend_ == EngineBackend::CUDA) {
            cuda_engine_ = std::make_unique<PdpfEngineCuda>(static_cast<unsigned>(n_bits_cap_), seed_);
        }
#endif
    }

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
#ifdef COMPOSITE_FSS_CUDA
        if (backend_ == EngineBackend::CUDA && cuda_engine_) {
            PdpfProgramId gpu_id = cuda_engine_->make_lut_program(desc, table_flat,
                                                                  static_cast<PdpfProgramId>(programs_.size()));
            (void)gpu_id;
        }
#endif
        PdpfProgramId id = static_cast<PdpfProgramId>(programs_.size());
        programs_.push_back(std::move(data));
        return id;
    }

    PdpfProgramId make_cmp_program(const CmpProgramDesc &desc,
                                   const std::vector<std::uint64_t> &table_flat) override {
        LutProgramDesc lut_desc{desc.domain_bits, 1};
        return make_lut_program(lut_desc, table_flat);
    }

    std::size_t program_bytes(PdpfProgramId program) const {
        if (program >= programs_.size()) return 0;
        return programs_[program].table.size() * sizeof(std::uint64_t);
    }

    void eval_share(PdpfProgramId program,
                    int party,
                    std::uint64_t x,
                    std::vector<std::uint64_t> &out_words) const override {
#ifdef COMPOSITE_FSS_CUDA
        if (backend_ == EngineBackend::CUDA && cuda_engine_) {
            cuda_engine_->eval_share(program, party, x, out_words);
            return;
        }
#endif
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
        std::uint64_t share_mask = (n_bits_cap_ >= 64)
                                       ? std::numeric_limits<std::uint64_t>::max()
                                       : ((1ULL << n_bits_cap_) - 1ULL);
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
                          const std::uint64_t *masked_xs,
                          std::size_t n_inputs,
                          std::uint64_t *flat_out) const override {
        if (masked_xs == nullptr || flat_out == nullptr || n_inputs == 0) return;
#ifdef COMPOSITE_FSS_CUDA
        if (backend_ == EngineBackend::CUDA && cuda_engine_) {
            cuda_engine_->eval_share_batch(program, party, masked_xs, n_inputs, flat_out);
            return;
        }
#endif
        if (program >= programs_.size()) {
            throw std::runtime_error("adapter: invalid program id");
        }
        const auto &data = programs_[program];
        std::size_t out_words = data.desc.output_words ? data.desc.output_words : 1;
        std::vector<std::uint64_t> tmp(out_words);
        for (std::size_t i = 0; i < n_inputs; ++i) {
            eval_share(program, party, masked_xs[i], tmp);
            for (std::size_t w = 0; w < out_words; ++w) {
                flat_out[i * out_words + w] = tmp[w];
            }
        }
    }

    void eval_share_batch(PdpfProgramId program,
                          int party,
                          const std::vector<std::uint64_t> &masked_xs,
                          std::vector<std::vector<std::uint64_t>> &out_batch) const override {
        out_batch.resize(masked_xs.size());
        if (masked_xs.empty()) return;
        LutProgramDesc desc = lookup_lut_desc(program);
        std::size_t out_words = desc.output_words ? desc.output_words : 1;
        std::vector<std::uint64_t> flat(masked_xs.size() * out_words);
        eval_share_batch(program, party, masked_xs.data(), masked_xs.size(), flat.data());
        for (std::size_t i = 0; i < masked_xs.size(); ++i) {
            out_batch[i].resize(out_words);
            for (std::size_t w = 0; w < out_words; ++w) {
                out_batch[i][w] = flat[i * out_words + w];
            }
        }
    }

    void eval_share_batch(PdpfProgramId program,
                          int party,
                          const std::vector<std::uint64_t> &masked_xs,
                          std::vector<std::uint64_t> &flat_out) const override {
        LutProgramDesc desc = lookup_lut_desc(program);
        std::size_t out_words = desc.output_words ? desc.output_words : 1;
        flat_out.assign(masked_xs.size() * out_words, 0);
        if (!masked_xs.empty()) {
            eval_share_batch(program, party, masked_xs.data(), masked_xs.size(), flat_out.data());
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
    EngineBackend backend_ = EngineBackend::CPU;
#ifdef COMPOSITE_FSS_CUDA
    std::unique_ptr<PdpfEngineCuda> cuda_engine_;
#endif

    std::uint64_t prg(std::size_t idx, unsigned word) const {
        std::uint64_t s = seed_ ^ static_cast<std::uint64_t>(idx * 0x9E3779B97F4A7C15ULL + word * 0xD1B54A32D192ED03ULL);
        std::mt19937_64 gen(s);
        return gen();
    }
};

} // namespace cfss

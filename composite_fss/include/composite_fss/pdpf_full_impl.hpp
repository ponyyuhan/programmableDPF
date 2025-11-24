#pragma once

#include "pdpf_full.hpp"
#include <pdpf/pdpf/pdpf_group.hpp>
#include <pdpf/pdpf/pdpf_lut.hpp>
#include <pdpf/prg/prg.hpp>
#include <pdpf/core/types.hpp>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <functional>

namespace cfss {

// PdpfProgram impl holding a pdpf_lut offline/online key pair per output word.
struct PdpfLutImpl {
    std::vector<pdpf::pdpf::PdpfLutOfflineKey> off;
    std::vector<pdpf::pdpf::PdpfLutOnlineKey> on;
};

struct PdpfProgramEntry {
    LutProgramDesc desc;
    std::shared_ptr<PdpfLutImpl> impl;
};

struct ProgramCacheKey {
    unsigned domain_bits = 0;
    unsigned output_words = 0;
    std::size_t table_hash = 0;

    bool operator==(const ProgramCacheKey &o) const {
        return domain_bits == o.domain_bits &&
               output_words == o.output_words &&
               table_hash == o.table_hash;
    }
};

struct ProgramCacheKeyHash {
    std::size_t operator()(const ProgramCacheKey &k) const {
        std::size_t h = std::hash<unsigned>{}(k.domain_bits);
        h ^= (std::hash<unsigned>{}(k.output_words) << 1);
        h ^= (k.table_hash << 1);
        return h;
    }
};

struct PdpfStats {
    std::size_t lut_gen = 0;
    std::size_t cmp_gen = 0;
    std::size_t evals = 0;
    std::size_t output_words = 0;
};

class PdpfEngineFullImpl : public PdpfEngineFull {
public:
    PdpfEngineFullImpl(std::size_t cap_bits = 24)
        : PdpfEngineFull(cap_bits, 0xA5A5) {
        pdpf::core::Seed master{};
        pdpf::core::RandomDevice rng;
        rng.random_seed(master);
        prg_ = std::make_shared<pdpf::prg::AesCtrPrg>(master);
        lut_engine_ = std::make_shared<pdpf::pdpf::PdpfLut>(prg_);
    }

    PdpfProgramId make_lut_program(const LutProgramDesc &desc,
                                   const std::vector<std::uint64_t> &table_flat) override {
        if (desc.output_words == 0) {
            throw std::runtime_error("PdpfEngineFullImpl: output_words must be > 0");
        }
        std::size_t domain_size = (desc.domain_bits == 64)
                                      ? (table_flat.size() / (desc.output_words ? desc.output_words : 1))
                                      : (1ULL << desc.domain_bits);
        if (table_flat.size() != domain_size * desc.output_words) {
            throw std::runtime_error("PdpfEngineFullImpl: table_flat size mismatch");
        }

        // Hash the payload (mask-insensitive cache key).
        std::size_t h = 0;
        for (auto v : table_flat) {
            h ^= std::hash<std::uint64_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        ProgramCacheKey cache_key{desc.domain_bits, desc.output_words, h};

        PdpfProgramEntry entry;
        entry.desc = desc;

        auto cache_it = cache_.find(cache_key);
        if (cache_it != cache_.end()) {
            entry.impl = cache_it->second;
        } else {
            stats_.lut_gen++;
        pdpf::core::SecurityParams sec;
        sec.lambda_bits = 128;
        sec.domain_size_N = domain_size;
        sec.epsilon = 0.25;

            entry.impl = std::make_shared<PdpfLutImpl>();
            entry.impl->off.resize(desc.output_words);
            entry.impl->on.resize(desc.output_words);
            for (unsigned w = 0; w < desc.output_words; ++w) {
                std::vector<std::uint64_t> table_word(domain_size);
                for (std::size_t idx = 0; idx < domain_size; ++idx) {
                    table_word[idx] = table_flat[idx * desc.output_words + w];
                }
                pdpf::core::GroupDescriptor group{{~0ULL}};
                entry.impl->off[w] = lut_engine_->gen_offline(sec, group, table_word, desc.domain_bits, 64);
                entry.impl->on[w] = lut_engine_->gen_online(entry.impl->off[w]);
            }
            cache_.emplace(cache_key, entry.impl);
        }
        PdpfProgramId id = static_cast<PdpfProgramId>(programs_.size());
        programs_.push_back(std::move(entry));
        return id;
    }

    PdpfProgramId make_cmp_program(const CmpProgramDesc &desc,
                                   const std::vector<std::uint64_t> &table_flat) override {
        // Comparison programs are treated as LUTs with output_words=1.
        LutProgramDesc lut_desc{desc.domain_bits, 1};
        stats_.cmp_gen++;
        return make_lut_program(lut_desc, table_flat);
    }

    void eval_share(PdpfProgramId program,
                    int party,
                    std::uint64_t masked_x,
                    std::vector<std::uint64_t> &out_words) const override {
        if (program >= programs_.size()) {
            throw std::runtime_error("PdpfEngineFullImpl: invalid program id");
        }
        const auto &entry = programs_[program];
        if (out_words.size() != entry.desc.output_words) {
            out_words.resize(entry.desc.output_words);
        }
        for (unsigned w = 0; w < entry.desc.output_words; ++w) {
            std::int64_t acc = 0;
            if (party == 0) {
                acc = lut_engine_->eval_point_offline(entry.impl->off[w], masked_x);
            } else {
                acc = lut_engine_->eval_point_online(entry.impl->on[w], masked_x);
            }
            stats_.output_words += entry.desc.output_words;
            out_words[w] = static_cast<std::uint64_t>(acc);
        }
        stats_.evals++;
    }

    void eval_share_batch(PdpfProgramId program,
                          int party,
                          const std::vector<std::uint64_t> &masked_xs,
                          std::vector<std::vector<std::uint64_t>> &out_batch) const override {
        out_batch.resize(masked_xs.size());
        if (program >= programs_.size()) {
            throw std::runtime_error("PdpfEngineFullImpl: invalid program id");
        }
        const auto &entry = programs_[program];
        for (std::size_t i = 0; i < masked_xs.size(); ++i) {
            if (out_batch[i].size() != entry.desc.output_words) {
                out_batch[i].resize(entry.desc.output_words);
            }
        }
        // Hoist per-program state; fallback to per-point for now.
        for (std::size_t i = 0; i < masked_xs.size(); ++i) {
            for (unsigned w = 0; w < entry.desc.output_words; ++w) {
                std::int64_t acc = 0;
                if (party == 0) {
                    acc = lut_engine_->eval_point_offline(entry.impl->off[w], masked_xs[i]);
                } else {
                    acc = lut_engine_->eval_point_online(entry.impl->on[w], masked_xs[i]);
                }
                stats_.output_words += entry.desc.output_words;
                out_batch[i][w] = static_cast<std::uint64_t>(acc);
            }
            stats_.evals++;
        }
    }

    LutProgramDesc lookup_lut_desc(PdpfProgramId program) const override {
        if (program >= programs_.size()) {
            return LutProgramDesc{};
        }
        return programs_[program].desc;
    }

    const PdpfStats &stats() const { return stats_; }
    void reset_stats() { stats_ = PdpfStats{}; }

private:
    std::shared_ptr<pdpf::prg::AesCtrPrg> prg_;
    std::shared_ptr<pdpf::pdpf::PdpfLut> lut_engine_;
    std::vector<PdpfProgramEntry> programs_;
    mutable PdpfStats stats_;
    std::unordered_map<ProgramCacheKey, std::shared_ptr<PdpfLutImpl>, ProgramCacheKeyHash> cache_;
};

} // namespace cfss

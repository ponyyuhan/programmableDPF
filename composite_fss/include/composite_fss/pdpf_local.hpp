#pragma once

#include "pdpf.hpp"
#include "sharing.hpp"
#include <unordered_map>
#include <functional>
#include <random>
#include <cassert>

namespace cfss {

// Local in-memory simulator: stores the true function and hands out additive
// shares deterministically derived from (key.id, x).
class LocalPdpfEngine : public PdpfEngine {
public:
    using Func = std::function<std::vector<u64>(u64)>;

    explicit LocalPdpfEngine(Ring64 ring = Ring64(64))
        : ring_(ring), next_id_(0) {}

    std::pair<PdpfKey, PdpfKey> progGen(const ProgramDesc &desc) override {
        std::size_t id = next_id_++;
        Func f = build_function(desc);
        programs_[id] = std::move(f);
        return {PdpfKey{id}, PdpfKey{id}};
    }

    std::vector<u64> eval(int party, const PdpfKey &key, u64 x) const override {
        auto it = programs_.find(key.id);
        assert(it != programs_.end());
        const Func &f = it->second;
        std::vector<u64> y = f(x);
        std::vector<u64> shares(y.size());
        for (std::size_t i = 0; i < y.size(); ++i) {
            std::uint64_t seed = static_cast<std::uint64_t>(key.id * 0x9e3779b97f4a7c15ULL ^ (x + i));
            std::mt19937_64 rng(seed);
            u64 r = rng() & ring_.modulus_mask;
            shares[i] = (party == 0) ? r : ring_.sub(y[i], r);
        }
        return shares;
    }

private:
    Ring64 ring_;
    std::size_t next_id_;
    std::unordered_map<std::size_t, Func> programs_;

    Func build_function(const ProgramDesc &d) {
        if (auto cmp = std::get_if<CompareDesc>(&d)) {
            CompareDesc cd = *cmp;
            return [cd](u64 x) -> std::vector<u64> {
                std::vector<u64> out(cd.thresholds.size());
                for (std::size_t i = 0; i < cd.thresholds.size(); ++i) {
                    out[i] = (x < cd.thresholds[i]) ? 1ULL : 0ULL;
                }
                return out;
            };
        }
        if (auto lut = std::get_if<LUTDesc>(&d)) {
            LUTDesc ld = *lut;
            return [ld](u64 x) -> std::vector<u64> {
                u64 mask = (ld.input_bits == 64) ? ~0ULL : ((1ULL << ld.input_bits) - 1ULL);
                u64 idx = x & mask;
                if (idx >= ld.table.size()) idx %= static_cast<u64>(ld.table.size());
                u64 out = ld.table[static_cast<std::size_t>(idx)] & ((ld.output_bits == 64) ? ~0ULL : ((1ULL << ld.output_bits) - 1ULL));
                return {out};
            };
        }
        auto mp = std::get<MultiPointDesc>(d);
        return [mp, this](u64 x) -> std::vector<u64> {
            std::vector<u64> out(mp.vector_len, 0ULL);
            for (std::size_t i = 0; i < mp.alphas.size(); ++i) {
                if (x == mp.alphas[i]) {
                    for (unsigned j = 0; j < mp.vector_len; ++j) {
                        out[j] = ring_.add(out[j], mp.betas[i][j]);
                    }
                }
            }
            return out;
        };
    }
};

} // namespace cfss

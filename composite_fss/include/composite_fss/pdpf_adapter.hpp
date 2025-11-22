#pragma once

#include "pdpf.hpp"
#include "sharing.hpp"
#include "ring.hpp"
#include "pdpf/pdpf/pdpf_group.hpp"
#include "pdpf/prg/prg.hpp"
#include "pdpf/core/types.hpp"
#include <memory>
#include <unordered_map>
#include <stdexcept>

namespace cfss {

// Adapter that derives deterministic additive shares using the project PDPF
// PRG (AesCtrPrg). It is functional (correctness) rather than compact, and
// only intended for small domains (e.g., LUTs or tiny n_bits comparisons).
class PdpfEngineAdapter : public PdpfEngine {
public:
    explicit PdpfEngineAdapter(unsigned n_bits_cap = 16)
        : n_bits_cap_(n_bits_cap) {
        pdpf::core::RandomDevice rng;
        rng.random_seed(master_seed_);
        prg_ = std::make_shared<pdpf::prg::AesCtrPrg>(master_seed_);
    }

    std::pair<PdpfKey, PdpfKey> progGen(const ProgramDesc &desc) override {
        std::size_t domain = domain_size(desc);
        if (domain == 0 || domain > (1ULL << n_bits_cap_)) {
            throw std::runtime_error("PdpfEngineAdapter: domain too large for adapter");
        }

        // Build function table explicitly.
        auto f = build_function(desc);
        std::vector<std::vector<u64>> values(domain);
        unsigned out_len = 0;
        for (std::size_t x = 0; x < domain; ++x) {
            values[x] = f(static_cast<u64>(x));
            if (out_len == 0) out_len = static_cast<unsigned>(values[x].size());
        }
        Ring64 ring(64);

        // Derive deterministic share0 via AesCtrPrg keyed by master_seed_,
        // counter = x * out_len + j; share1 = val - share0.
        std::vector<std::vector<u64>> shares0(domain, std::vector<u64>(out_len, 0));
        std::vector<std::vector<u64>> shares1(domain, std::vector<u64>(out_len, 0));

        for (std::size_t x = 0; x < domain; ++x) {
            for (unsigned j = 0; j < out_len; ++j) {
                u64 ctr = static_cast<u64>(x * out_len + j);
                u64 s0 = derive_share0(ctr);
                u64 v = (j < values[x].size()) ? values[x][j] : 0;
                shares0[x][j] = s0 & ring.modulus_mask;
                shares1[x][j] = ring.sub(v, shares0[x][j]);
            }
        }

        std::size_t id = next_id_++;
        programs_[id] = Program{
            .domain = domain,
            .out_len = out_len,
            .shares0 = std::move(shares0),
            .shares1 = std::move(shares1),
        };
        return {PdpfKey{id}, PdpfKey{id}};
    }

    std::vector<u64> eval(int party, const PdpfKey &key, u64 x) const override {
        auto it = programs_.find(key.id);
        if (it == programs_.end()) {
            throw std::runtime_error("PdpfEngineAdapter: unknown key id");
        }
        const Program &p = it->second;
        u64 idx = x % p.domain;
        if (party == 0) {
            return p.shares0[static_cast<std::size_t>(idx)];
        }
        return p.shares1[static_cast<std::size_t>(idx)];
    }

private:
    struct Program {
        std::size_t domain;
        unsigned out_len;
        std::vector<std::vector<u64>> shares0;
        std::vector<std::vector<u64>> shares1;
    };

    unsigned n_bits_cap_;
    pdpf::core::Seed master_seed_{};
    std::shared_ptr<pdpf::prg::AesCtrPrg> prg_;
    std::unordered_map<std::size_t, Program> programs_;
    std::size_t next_id_{0};

    static std::size_t domain_size(const ProgramDesc &d) {
        if (auto cmp = std::get_if<CompareDesc>(&d)) {
            if (cmp->n_bits >= 64) return 0;
            return 1ULL << cmp->n_bits;
        }
        if (auto lut = std::get_if<LUTDesc>(&d)) {
            return lut->table.size();
        }
        auto mp = std::get<MultiPointDesc>(d);
        if (mp.n_bits >= 64) return 0;
        return 1ULL << mp.n_bits;
    }

    // Local evaluator reused from the simulator to produce clear outputs.
    static std::function<std::vector<u64>(u64)> build_function(const ProgramDesc &d) {
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
                u64 idx = x % static_cast<u64>(ld.table.size());
                u64 mask = (ld.output_bits == 64) ? ~0ULL : ((1ULL << ld.output_bits) - 1ULL);
                return {ld.table[static_cast<std::size_t>(idx)] & mask};
            };
        }
        auto mp = std::get<MultiPointDesc>(d);
        return [mp](u64 x) -> std::vector<u64> {
            std::vector<u64> out(mp.vector_len, 0ULL);
            for (std::size_t i = 0; i < mp.alphas.size(); ++i) {
                if (x == mp.alphas[i]) {
                    for (unsigned j = 0; j < mp.vector_len; ++j) {
                        out[j] += mp.betas[i][j];
                    }
                }
            }
            return out;
        };
    }

    // Use AES-CTR PRG to derive a 64-bit pseudorandom share from master_seed_
    // and a counter.
    u64 derive_share0(u64 counter) const {
        pdpf::core::Seed s = master_seed_;
        for (int i = 0; i < 8; ++i) {
            s[15 - i] ^= static_cast<std::uint8_t>((counter >> (8 * i)) & 0xFF);
        }
        pdpf::core::Seed left{}, right{};
        prg_->expand(s, left, right);
        u64 v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<u64>(left[i]) << (8 * i);
        }
        return v;
    }
};

} // namespace cfss

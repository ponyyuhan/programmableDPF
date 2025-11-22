#pragma once

#include "../pdpf_adapter.hpp"
#include "../arith.hpp"
#include "../sharing.hpp"
#include <random>
#include <vector>
#include <cmath>

namespace cfss {

struct NExpGateParams {
    unsigned n_bits;
    unsigned f; // fractional bits
};

struct NExpGateKey {
    std::uint64_t r_in;
    std::uint64_t r_out;
    PdpfKey lut_key;
};

struct NExpGateKeyPair {
    NExpGateKey k0;
    NExpGateKey k1;
};

inline NExpGateKeyPair gen_nexp_gate(const NExpGateParams &params,
                                     PdpfEngine &engine,
                                     std::mt19937_64 &rng) {
    RingConfig cfg = make_ring_config(params.n_bits);
    std::uniform_int_distribution<std::uint64_t> dist(0, cfg.modulus_mask);
    std::uint64_t r_in = 0;
    std::uint64_t r_out = 0;

    std::size_t size = 1ULL << params.n_bits;
    LUTDesc desc;
    desc.input_bits = params.n_bits;
    desc.output_bits = params.n_bits;
    desc.table.resize(size);

    double scale = static_cast<double>(1ULL << params.f);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        std::uint64_t x = ring_sub(cfg, static_cast<std::uint64_t>(x_hat), r_in);
        double xr = static_cast<double>(static_cast<std::int64_t>(x)) / scale;
        double val = std::exp(-xr);
        std::int64_t fp = static_cast<std::int64_t>(std::llround(val * scale));
        desc.table[x_hat] = ring_add(cfg, static_cast<std::uint64_t>(fp), r_out);
    }

    auto [k0, k1] = engine.progGen(desc);
    NExpGateKeyPair pair;
    pair.k0 = NExpGateKey{r_in, r_out, k0};
    pair.k1 = NExpGateKey{r_in, r_out, k1};
    return pair;
}

inline Share nexpgate_eval(int party,
                           const NExpGateKey &key,
                           std::uint64_t x_hat,
                           PdpfEngine &engine) {
    auto out = engine.eval(party, key.lut_key, x_hat);
    return Share{party, out.empty() ? 0 : out[0]};
}

} // namespace cfss

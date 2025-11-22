#pragma once

#include "../pdpf_adapter.hpp"
#include "../arith.hpp"
#include "../sharing.hpp"
#include <random>
#include <vector>
#include <cmath>

namespace cfss {

struct RecSqrtGateParams {
    unsigned n_bits;
    unsigned f_in;
    unsigned f_out;
    unsigned domain_bits;
};

struct RecSqrtGateKey {
    std::uint64_t r_in;
    std::uint64_t r_out;
    PdpfKey lut_key;
};

struct RecSqrtGateKeyPair {
    RecSqrtGateKey k0;
    RecSqrtGateKey k1;
};

inline RecSqrtGateKeyPair gen_recsqrt_gate(const RecSqrtGateParams &params,
                                           PdpfEngine &engine,
                                           std::mt19937_64 &rng) {
    RingConfig cfg = make_ring_config(params.n_bits);
    std::uniform_int_distribution<std::uint64_t> dist(0, cfg.modulus_mask);
    std::uint64_t r_in = dist(rng);
    std::uint64_t r_out = dist(rng);

    std::size_t size = 1ULL << params.domain_bits;
    LUTDesc desc;
    desc.input_bits = params.domain_bits;
    desc.output_bits = params.n_bits;
    desc.table.resize(size);

    double scale_out = static_cast<double>(1ULL << params.f_out);
    double scale_in = static_cast<double>(1ULL << params.f_in);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        std::uint64_t x = ring_sub(cfg, static_cast<std::uint64_t>(x_hat), r_in);
        if (x == 0) x = 1; // avoid division by zero
        double xr = static_cast<double>(x) / scale_in;
        double val = 1.0 / std::sqrt(xr);
        std::int64_t fp = static_cast<std::int64_t>(std::llround(val * scale_out));
        desc.table[x_hat] = ring_add(cfg, static_cast<std::uint64_t>(fp), r_out);
    }

    auto [k0, k1] = engine.progGen(desc);
    RecSqrtGateKeyPair pair;
    pair.k0 = RecSqrtGateKey{r_in, r_out, k0};
    pair.k1 = RecSqrtGateKey{r_in, r_out, k1};
    return pair;
}

inline Share recsqrt_eval(int party,
                          const RecSqrtGateKey &key,
                          std::uint64_t x_hat,
                          PdpfEngine &engine) {
    auto out = engine.eval(party, key.lut_key, x_hat);
    return Share{party, out.empty() ? 0 : out[0]};
}

} // namespace cfss

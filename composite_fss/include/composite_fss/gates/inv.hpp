#pragma once

#include "../pdpf_adapter.hpp"
#include "../arith.hpp"
#include "../sharing.hpp"
#include <random>
#include <vector>

namespace cfss {

struct InvGateParams {
    unsigned n_bits;
    unsigned f;      // fractional bits
    unsigned max_d;  // maximum denominator (inclusive)
};

struct InvGateKey {
    std::uint64_t r_in;
    std::uint64_t r_out;
    PdpfKey lut_key;
};

struct InvGateKeyPair {
    InvGateKey k0;
    InvGateKey k1;
};

inline InvGateKeyPair gen_inv_gate(const InvGateParams &params,
                                   PdpfEngine &engine,
                                   std::mt19937_64 &rng) {
    RingConfig cfg = make_ring_config(params.n_bits);
    std::uniform_int_distribution<std::uint64_t> dist(0, cfg.modulus_mask);
    std::uint64_t r_in = 0;
    std::uint64_t r_out = 0;

    unsigned domain_bits = 0;
    std::uint32_t bound = params.max_d + 1;
    while ((1u << domain_bits) < bound) ++domain_bits;
    std::size_t size = 1ULL << domain_bits;

    LUTDesc desc;
    desc.input_bits = domain_bits;
    desc.output_bits = params.n_bits;
    desc.table.resize(size);

    double scale = static_cast<double>(1ULL << params.f);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        std::uint64_t x = ring_sub(cfg, static_cast<std::uint64_t>(x_hat), r_in);
        std::uint64_t den = (x == 0) ? 1 : (x % bound);
        double val = 1.0 / static_cast<double>(den);
        std::int64_t fp = static_cast<std::int64_t>(std::llround(val * scale));
        desc.table[x_hat] = ring_add(cfg, static_cast<std::uint64_t>(fp), r_out);
    }

    auto [k0, k1] = engine.progGen(desc);
    InvGateKeyPair pair;
    pair.k0 = InvGateKey{r_in, r_out, k0};
    pair.k1 = InvGateKey{r_in, r_out, k1};
    return pair;
}

inline Share invgate_eval(int party,
                          const InvGateKey &key,
                          std::uint64_t x_hat,
                          PdpfEngine &engine) {
    auto out = engine.eval(party, key.lut_key, x_hat);
    return Share{party, out.empty() ? 0 : out[0]};
}

} // namespace cfss

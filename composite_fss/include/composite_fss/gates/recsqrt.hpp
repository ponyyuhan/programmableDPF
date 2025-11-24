#pragma once

#include "../pdpf_adapter.hpp"
#include "../arith.hpp"
#include "../sharing.hpp"
#include "../suf.hpp"
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
    PdpfProgramId prog;
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
    std::vector<std::uint64_t> table(size);

    double scale_out = static_cast<double>(1ULL << params.f_out);
    double scale_in = static_cast<double>(1ULL << params.f_in);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        std::uint64_t x = ring_sub(cfg, static_cast<std::uint64_t>(x_hat), r_in);
        if (x == 0) x = 1; // avoid division by zero
        double xr = static_cast<double>(x) / scale_in;
        double val = 1.0 / std::sqrt(xr);
        std::int64_t fp = static_cast<std::int64_t>(std::llround(val * scale_out));
        table[x_hat] = ring_add(cfg, static_cast<std::uint64_t>(fp), r_out);
    }

    auto suf = table_to_suf(params.domain_bits, 1, table);
    auto compiled = compile_suf_to_pdpf(suf, engine);
    RecSqrtGateKeyPair pair;
    pair.k0 = RecSqrtGateKey{r_in, r_out, compiled.pdpf_program};
    pair.k1 = RecSqrtGateKey{r_in, r_out, compiled.pdpf_program};
    return pair;
}

inline Share recsqrt_eval(int party,
                          const RecSqrtGateKey &key,
                          std::uint64_t x_hat,
                          PdpfEngine &engine) {
    std::vector<std::uint64_t> out(1);
    engine.eval_share(key.prog, party, x_hat, out);
    return Share{party, out[0]};
}

} // namespace cfss

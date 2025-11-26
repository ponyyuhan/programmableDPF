#pragma once

#include "../pdpf_adapter.hpp"
#include "../arith.hpp"
#include "../sharing.hpp"
#include "../suf.hpp"
#include <random>
#include <vector>
#include <cmath>

namespace cfss {

struct NExpGateParams {
    unsigned n_bits;
    unsigned f; // fractional bits
};

struct NExpGateKey {
    std::uint64_t r_in;   // sum mask
    std::uint64_t r_out;  // sum mask
    Share r_in_share;
    Share r_out_share;
    PdpfProgramId prog;
};

struct NExpGateKeyPair {
    NExpGateKey k0;
    NExpGateKey k1;
};

inline NExpGateKeyPair gen_nexp_gate(const NExpGateParams &params,
                                     PdpfEngine &engine,
                                     std::mt19937_64 &rng) {
    Ring64 ring(params.n_bits);
    RingConfig cfg = make_ring_config(params.n_bits);
    std::uniform_int_distribution<std::uint64_t> dist(0, cfg.modulus_mask);
    std::uint64_t r_in = dist(rng);
    std::uint64_t r_out = dist(rng);
    std::uint64_t r_in_share0 = dist(rng);
    std::uint64_t r_out_share0 = dist(rng);
    std::uint64_t r_in_share1 = ring_sub(cfg, r_in, r_in_share0);
    std::uint64_t r_out_share1 = ring_sub(cfg, r_out, r_out_share0);

    std::size_t size = 1ULL << params.n_bits;
    std::vector<std::uint64_t> table(size);

    double scale = static_cast<double>(1ULL << params.f);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        std::uint64_t x = ring_sub(cfg, static_cast<std::uint64_t>(x_hat), r_in);
        double xr = static_cast<double>(ring.to_signed(x)) / scale;
        double val = std::exp(-xr);
        std::int64_t fp = static_cast<std::int64_t>(std::llround(val * scale));
        table[x_hat] = ring_add(cfg, static_cast<std::uint64_t>(fp), r_out);
    }

    auto suf = table_to_suf(params.n_bits, 1, table);
    auto compiled = compile_suf_to_pdpf(suf, engine);
    NExpGateKeyPair pair;
    pair.k0 = NExpGateKey{r_in, r_out, Share{0, r_in_share0}, Share{0, r_out_share0}, compiled.pdpf_program};
    pair.k1 = NExpGateKey{r_in, r_out, Share{1, r_in_share1}, Share{1, r_out_share1}, compiled.pdpf_program};
    return pair;
}

inline Share nexpgate_eval(int party,
                           const NExpGateKey &key,
                           std::uint64_t x_hat,
                           PdpfEngine &engine) {
    std::vector<std::uint64_t> out(1);
    engine.eval_share(key.prog, party, x_hat, out);
    return Share{party, out[0]};
}

// Evaluate nExp on a secret share without opening it unmasked.
inline std::pair<Share, Share> nexpgate_eval_from_share_pair(const RingConfig &cfg,
                                                             const NExpGateKey &k0,
                                                             const NExpGateKey &k1,
                                                             const Share &z0,
                                                             const Share &z1,
                                                             PdpfEngine &engine) {
    // Mask input
    Share zhat0 = add(cfg, z0, k0.r_in_share);
    Share zhat1 = add(cfg, z1, k1.r_in_share);
    std::uint64_t hat = ring_add(cfg, share_value(zhat0), share_value(zhat1));

    // Eval PDPF on masked input
    auto y0 = nexpgate_eval(0, k0, hat, engine);
    auto y1 = nexpgate_eval(1, k1, hat, engine);

    // Remove output mask
    y0 = sub(cfg, y0, k0.r_out_share);
    y1 = sub(cfg, y1, k1.r_out_share);
    return {y0, y1};
}

// Batched nExp on secret shares; outputs vectors of the same length.
inline std::pair<std::vector<Share>, std::vector<Share>>
nexpgate_eval_batch_from_share_pair(const RingConfig &cfg,
                                    const NExpGateKey &k0,
                                    const NExpGateKey &k1,
                                    const std::vector<Share> &z0,
                                    const std::vector<Share> &z1,
                                    PdpfEngine &engine) {
    std::size_t n = z0.size();
    std::vector<Share> out0(n), out1(n);
    if (z1.size() != n) return {out0, out1};

    std::vector<std::uint64_t> hats(n);
    for (std::size_t i = 0; i < n; ++i) {
        Share zhat0 = add(cfg, z0[i], k0.r_in_share);
        Share zhat1 = add(cfg, z1[i], k1.r_in_share);
        hats[i] = ring_add(cfg, share_value(zhat0), share_value(zhat1));
    }

    LutProgramDesc desc0 = engine.lookup_lut_desc(k0.prog);
    std::size_t out_words = desc0.output_words ? desc0.output_words : 1;
    std::vector<std::uint64_t> flat_out0(n * out_words, 0), flat_out1(n * out_words, 0);
    engine.eval_share_batch(k0.prog, 0, hats.data(), n, flat_out0.data());
    engine.eval_share_batch(k1.prog, 1, hats.data(), n, flat_out1.data());

    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t y0_raw = flat_out0[i * out_words];
        std::uint64_t y1_raw = flat_out1[i * out_words];
        Share y0{0, y0_raw};
        Share y1{1, y1_raw};
        y0 = sub(cfg, y0, k0.r_out_share);
        y1 = sub(cfg, y1, k1.r_out_share);
        out0[i] = y0;
        out1[i] = y1;
    }
    return {out0, out1};
}

} // namespace cfss

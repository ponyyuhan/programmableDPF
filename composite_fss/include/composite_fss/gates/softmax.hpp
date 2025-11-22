#pragma once

#include "../arith.hpp"
#include "../beaver.hpp"
#include "../sharing.hpp"
#include "nexp.hpp"
#include "inv.hpp"
#include <vector>
#include <random>

namespace cfss {

struct SoftmaxParams {
    unsigned n_bits;
    unsigned f;
    std::size_t vec_len;
};

struct SoftmaxKey {
    SoftmaxParams params;
    NExpGateKey nexp_key;
    InvGateKey inv_key;
};

struct SoftmaxKeyPair {
    SoftmaxKey k0;
    SoftmaxKey k1;
};

inline SoftmaxKeyPair softmax_keygen(const SoftmaxParams &params,
                                     PdpfEngine &engine,
                                     std::mt19937_64 &rng) {
    SoftmaxKeyPair kp;
    kp.k0.params = kp.k1.params = params;

    NExpGateParams np{params.n_bits, params.f};
    auto nkeys = gen_nexp_gate(np, engine, rng);
    kp.k0.nexp_key = nkeys.k0;
    kp.k1.nexp_key = nkeys.k1;

    InvGateParams ip;
    ip.n_bits = params.n_bits;
    ip.f = params.f;
    ip.max_d = static_cast<unsigned>(params.vec_len * 2); // loose upper bound
    auto ikeys = gen_inv_gate(ip, engine, rng);
    kp.k0.inv_key = ikeys.k0;
    kp.k1.inv_key = ikeys.k1;
    return kp;
}

struct SoftmaxEvalShare {
    std::vector<Share> y;
};

// Oblivious softmax: expects input shares x (same length for both parties).
inline std::pair<SoftmaxEvalShare, SoftmaxEvalShare>
softmax_eval_pair(PdpfEngine &engine,
                  const SoftmaxKeyPair &keys,
                  const RingConfig &cfg,
                  const std::vector<Share> &x0,
                  const std::vector<Share> &x1,
                  BeaverPool &pool0,
                  BeaverPool &pool1) {
    const auto &k0 = keys.k0;
    std::size_t k = k0.params.vec_len;
    SoftmaxEvalShare out0, out1;
    out0.y.resize(k);
    out1.y.resize(k);

    // Reconstruct max via opens (for this iteration, keep open explicit).
    std::int64_t m = 0;
    {
        std::vector<std::uint64_t> opens(k);
        for (std::size_t i = 0; i < k; ++i) {
            opens[i] = ring_add(cfg, x0[i].v, x1[i].v);
        }
        m = to_signed(cfg, opens[0]);
        for (std::size_t i = 1; i < k; ++i) {
            std::int64_t v = to_signed(cfg, opens[i]);
            if (v > m) m = v;
        }
    }

    // Open inputs to compute delta = x - max (still functional, not oblivious).
    std::vector<std::uint64_t> opens(k);
    for (std::size_t i = 0; i < k; ++i) opens[i] = ring_add(cfg, x0[i].v, x1[i].v);
    std::int64_t mx = to_signed(cfg, opens[0]);
    for (std::size_t i = 1; i < k; ++i) {
        std::int64_t v = to_signed(cfg, opens[i]);
        if (v > mx) mx = v;
    }

    // nExp on (x - mx) via PDPF LUT (public delta input).
    std::vector<Share> exp0(k), exp1(k);
    for (std::size_t i = 0; i < k; ++i) {
        std::int64_t delta = to_signed(cfg, opens[i]) - mx;
        std::uint64_t delta_u = static_cast<std::uint64_t>(delta) & cfg.modulus_mask;
        exp0[i] = nexpgate_eval(0, k0.nexp_key, delta_u, engine);
        exp1[i] = nexpgate_eval(1, keys.k1.nexp_key, delta_u, engine);
    }

    // Sum exp shares and open denominator.
    std::uint64_t denom_open = 0;
    for (std::size_t i = 0; i < k; ++i) {
        denom_open = ring_add(cfg, denom_open, ring_add(cfg, exp0[i].v, exp1[i].v));
    }
    if (denom_open == 0) denom_open = 1;
    auto inv0 = invgate_eval(0, k0.inv_key, denom_open, engine);
    auto inv1 = invgate_eval(1, keys.k1.inv_key, denom_open, engine);

    // y_i = exp_i * inv using Beaver.
    for (std::size_t i = 0; i < k; ++i) {
        auto t_pub = pool0.next_triple_public();
        auto t0 = pool0.share_triple(t_pub);
        auto t1 = pool1.share_triple(t_pub);
        auto m0 = mul_prepare(cfg, exp0[i], inv0, t0);
        auto m1 = mul_prepare(cfg, exp1[i], inv1, t1);
        auto [d_open, e_open] = mul_open(cfg, m0, m1);
        auto z0 = mul_finish(cfg, 0, t0, d_open, e_open);
        auto z1 = mul_finish(cfg, 1, t1, d_open, e_open);
        // scale down by f
        z0.v >>= k0.params.f;
        z1.v >>= k0.params.f;
        out0.y[i] = z0;
        out1.y[i] = z1;
    }

    return {out0, out1};
}

} // namespace cfss

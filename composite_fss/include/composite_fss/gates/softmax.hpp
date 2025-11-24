#pragma once

#include "../arith.hpp"
#include "../beaver.hpp"
#include "../sharing.hpp"
#include "../wire.hpp"
#include "nexp.hpp"
#include "inv.hpp"
#include "drelu.hpp"
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
    // One DReLU per comparison step in the linear scan.
    std::vector<DreluKey> drelu_keys;
    // One nExp key per element.
    std::vector<NExpGateKey> nexp_keys;
    // One Inv key for the denominator.
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

    // DReLU masks for k-1 comparisons.
    kp.k0.drelu_keys.resize(params.vec_len - 1);
    kp.k1.drelu_keys.resize(params.vec_len - 1);
    for (std::size_t i = 0; i + 1 < params.vec_len; ++i) {
        auto dkeys = drelu_gen(params.n_bits, engine, rng);
        kp.k0.drelu_keys[i] = dkeys.k0;
        kp.k1.drelu_keys[i] = dkeys.k1;
    }

    // nExp per element.
    kp.k0.nexp_keys.resize(params.vec_len);
    kp.k1.nexp_keys.resize(params.vec_len);
    NExpGateParams np{params.n_bits, params.f};
    for (std::size_t i = 0; i < params.vec_len; ++i) {
        auto nkeys = gen_nexp_gate(np, engine, rng);
        kp.k0.nexp_keys[i] = nkeys.k0;
        kp.k1.nexp_keys[i] = nkeys.k1;
    }

    // Single Inv key for the denominator.
    InvGateParams ip{params.n_bits, params.f, static_cast<unsigned>(params.vec_len * 2)};
    auto ikeys = gen_inv_gate(ip, engine, rng);
    kp.k0.inv_key = ikeys.k0;
    kp.k1.inv_key = ikeys.k1;
    return kp;
}

struct SoftmaxEvalShare {
    std::vector<Share> y;
};

// Fully oblivious softmax on masked wires (single-process two-party simulation).
inline std::pair<SoftmaxEvalShare, SoftmaxEvalShare>
softmax_eval_pair(PdpfEngine &engine,
                  const SoftmaxKeyPair &keys,
                  const RingConfig &cfg,
                  const std::vector<MaskedWire> &x0,
                  const std::vector<MaskedWire> &x1,
                  BeaverPool &pool0,
                  BeaverPool &pool1) {
    const auto &k0 = keys.k0;
    const auto &k1 = keys.k1;
    std::size_t k = k0.params.vec_len;
    SoftmaxEvalShare out0, out1;
    out0.y.resize(k);
    out1.y.resize(k);

    // Start from secret shares of logits.
    std::vector<Share> cur0(k), cur1(k);
    for (std::size_t i = 0; i < k; ++i) {
        cur0[i] = x0[i].x;
        cur1[i] = x1[i].x;
    }

    // Secure max via DReLU + Beaver select.
    Share x_max0 = cur0[0];
    Share x_max1 = cur1[0];
    for (std::size_t i = 1; i < k; ++i) {
        Share diff0 = sub(cfg, cur0[i], x_max0);
        Share diff1 = sub(cfg, cur1[i], x_max1);
        // Mask diff
        Share masked0 = add(cfg, diff0, k0.drelu_keys[i - 1].r_in);
        Share masked1 = add(cfg, diff1, k1.drelu_keys[i - 1].r_in);
        std::uint64_t hat_diff = open_share_pair(cfg, masked0, masked1);
        MaskedWire diff_wire0{hat_diff, diff0, k0.drelu_keys[i - 1].r_in};
        MaskedWire diff_wire1{hat_diff, diff1, k1.drelu_keys[i - 1].r_in};
        Share bit0 = drelu_eval(k0.drelu_keys[i - 1], diff_wire0, engine, 0);
        Share bit1 = drelu_eval(k1.drelu_keys[i - 1], diff_wire1, engine, 1);
        Share delta0 = sub(cfg, cur0[i], x_max0);
        Share delta1 = sub(cfg, cur1[i], x_max1);
        Share prod0 = beaver_mul(pool0, cfg, bit0, delta0);
        Share prod1 = beaver_mul(pool1, cfg, bit1, delta1);
        x_max0 = add(cfg, x_max0, prod0);
        x_max1 = add(cfg, x_max1, prod1);
    }

    // z_i = x_max - x_i
    std::vector<Share> z0(k), z1(k);
    for (std::size_t i = 0; i < k; ++i) {
        z0[i] = sub(cfg, x_max0, cur0[i]);
        z1[i] = sub(cfg, x_max1, cur1[i]);
    }

    // exp_i = exp(-(x_i - x_max)) using masked nExp (per element).
    std::vector<Share> exp0(k), exp1(k);
    for (std::size_t i = 0; i < k; ++i) {
        auto [e0, e1] = nexpgate_eval_from_share_pair(cfg, k0.nexp_keys[i], k1.nexp_keys[i],
                                                      z0[i], z1[i], engine);
        exp0[i] = e0;
        exp1[i] = e1;
    }

    // denom = sum exp_i
    Share denom0 = exp0[0];
    Share denom1 = exp1[0];
    for (std::size_t i = 1; i < k; ++i) {
        denom0 = add(cfg, denom0, exp0[i]);
        denom1 = add(cfg, denom1, exp1[i]);
    }

    // inv_denom = 1/denom using masked inv.
    auto [inv0, inv1] = invgate_eval_from_share_pair(cfg, k0.inv_key, k1.inv_key, denom0, denom1, engine);

    // Normalize with Beaver multiplication and fixed-point truncation.
    for (std::size_t i = 0; i < k; ++i) {
        Share prod0 = beaver_mul(pool0, cfg, exp0[i], inv0);
        Share prod1 = beaver_mul(pool1, cfg, exp1[i], inv1);
        // Fixed-point truncation by f bits (logical shift).
        prod0 = Share{prod0.party(), prod0.value_internal() >> k0.params.f};
        prod1 = Share{prod1.party(), prod1.value_internal() >> k0.params.f};
        out0.y[i] = prod0;
        out1.y[i] = prod1;
    }

    return {out0, out1};
}

} // namespace cfss

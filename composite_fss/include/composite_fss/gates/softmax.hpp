#pragma once

#include "../arith.hpp"
#include "../beaver.hpp"
#include "../sharing.hpp"
#include "../wire.hpp"
#include "nexp.hpp"
#include "inv.hpp"
#include "drelu.hpp"
#include "trunc.hpp"
#include <vector>
#include <random>
#include <iostream>

namespace cfss {

struct SoftmaxParams {
    unsigned n_bits;
    unsigned f;
    std::size_t vec_len;
};

struct SoftmaxKey {
    SoftmaxParams params;
    unsigned inv_f = 0; // fractional bits of the inverse output
    unsigned trunc_shift = 0;
    // One DReLU per comparison step in the linear scan.
    std::vector<DreluKey> drelu_keys;
    // Shared nExp kernel + per-element masks.
    NExpKernel nexp_kernel;
    std::vector<NExpInstanceKey> nexp_instances;
    // One Inv key for the denominator.
    InvGateKey inv_key;
    // Truncation for normalization.
    LRSKey trunc_key;
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
    // Choose a safe fractional precision for the inverse: f + inv_f <= n_bits - 1.
    unsigned inv_f = std::min<unsigned>(params.n_bits - 2, params.f + 6);
    kp.k0.inv_f = kp.k1.inv_f = inv_f;
    unsigned trunc_shift = (inv_f > params.f) ? (inv_f - params.f) : 0;
    kp.k0.trunc_shift = kp.k1.trunc_shift = trunc_shift;

    // DReLU masks for k-1 comparisons.
    std::size_t cmp_count = (params.vec_len > 0) ? (params.vec_len - 1) : 0;
    kp.k0.drelu_keys.resize(cmp_count);
    kp.k1.drelu_keys.resize(cmp_count);
    if (cmp_count > 0) {
        auto dkeys = drelu_gen(params.n_bits, engine, rng);
        kp.k0.drelu_keys.assign(cmp_count, dkeys.k0);
        kp.k1.drelu_keys.assign(cmp_count, dkeys.k1);
    }

    // Shared nExp kernel with per-element masks.
    NExpGateParams np{params.n_bits, params.f};
    kp.k0.nexp_kernel = gen_nexp_kernel(np, engine);
    kp.k1.nexp_kernel = kp.k0.nexp_kernel;
    kp.k0.nexp_instances.resize(params.vec_len);
    kp.k1.nexp_instances.resize(params.vec_len);
    for (std::size_t i = 0; i < params.vec_len; ++i) {
        auto inst_pair = gen_nexp_instance(params.n_bits, rng);
        kp.k0.nexp_instances[i] = inst_pair.k0;
        kp.k1.nexp_instances[i] = inst_pair.k1;
    }

    // Single Inv key for the denominator.
    InvGateParams ip{
        params.n_bits,
        kp.k0.inv_f,
        static_cast<unsigned>(params.vec_len * (1u << params.f))
    };
    auto ikeys = gen_inv_gate(ip, engine, rng);
    kp.k0.inv_key = ikeys.k0;
    kp.k1.inv_key = ikeys.k1;

    // Truncation key for normalization by 2^{inv_f}.
    MPCContext dealer_ctx(params.n_bits, rng());
    auto trunc_keys = lrs_gen(params.n_bits, trunc_shift, engine, dealer_ctx);
    kp.k0.trunc_key = trunc_keys.k0;
    kp.k1.trunc_key = trunc_keys.k1;
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
    if (k == 0) return {out0, out1};

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
        std::uint64_t hat_diff = ring_add(cfg, share_value(masked0), share_value(masked1));
        MaskedWire diff_wire0{hat_diff, diff0, k0.drelu_keys[i - 1].r_in};
        MaskedWire diff_wire1{hat_diff, diff1, k1.drelu_keys[i - 1].r_in};
        Share bit0 = drelu_eval(k0.drelu_keys[i - 1], diff_wire0, engine, 0);
        Share bit1 = drelu_eval(k1.drelu_keys[i - 1], diff_wire1, engine, 1);
        Share delta0 = sub(cfg, cur0[i], x_max0);
        Share delta1 = sub(cfg, cur1[i], x_max1);
        auto prod_pair = beaver_mul_pair_from_pools(cfg, pool0, pool1, bit0, bit1, delta0, delta1);
        x_max0 = add(cfg, x_max0, prod_pair.first);
        x_max1 = add(cfg, x_max1, prod_pair.second);
    }

    // z_i = x_max - x_i
    std::vector<Share> z0(k), z1(k);
    for (std::size_t i = 0; i < k; ++i) {
        z0[i] = sub(cfg, x_max0, cur0[i]);
        z1[i] = sub(cfg, x_max1, cur1[i]);
    }

    // exp_i = exp(-(x_i - x_max)) using shared kernel + per-element masks (batched).
    auto exp_pair = nexpgate_eval_batch_from_instances(cfg, k0.nexp_kernel,
                                                       k0.nexp_instances,
                                                       k1.nexp_instances,
                                                       z0, z1, engine);
    std::vector<Share> exp0 = std::move(exp_pair.first);
    std::vector<Share> exp1 = std::move(exp_pair.second);

    // denom = sum exp_i (shares)
    Share denom0 = exp0[0];
    Share denom1 = exp1[0];
    for (std::size_t i = 1; i < k; ++i) {
        denom0 = add(cfg, denom0, exp0[i]);
        denom1 = add(cfg, denom1, exp1[i]);
    }

    // Add tiny epsilon (1 ULP in Q_f) to avoid zero.
    denom0 = add_const(cfg, denom0, 1ULL);

    // Inverse: Q_inv_f
    auto inv_pair = invgate_eval_from_share_pair(cfg, k0.inv_key, k1.inv_key, denom0, denom1, engine);
    Share inv0 = inv_pair.first;
    Share inv1 = inv_pair.second;

    // Normalize with Beaver mul + LRS truncation.
    std::vector<Share> prod0(k), prod1(k);
    for (std::size_t i = 0; i < k; ++i) {
        auto prod_pair = beaver_mul_pair_from_pools(cfg, pool0, pool1, exp0[i], exp1[i], inv0, inv1);
        prod0[i] = prod_pair.first;
        prod1[i] = prod_pair.second;
    }
    auto norm_batch = lrs_eval_batch_from_share_pair(cfg, k0.trunc_key, k1.trunc_key,
                                                     prod0, prod1, engine);
    out0.y = std::move(norm_batch.first);
    out1.y = std::move(norm_batch.second);

    return {out0, out1};
}

} // namespace cfss

#pragma once

#include "../arith.hpp"
#include "../beaver.hpp"
#include "../gates/gelu.hpp"
#include "../gates/trunc.hpp"
#include "../sharing.hpp"
#include <random>
#include <vector>

namespace cfss {

enum class FusedActivationKind { None, GeLU, SiLU };

struct FusedLayerParams {
    unsigned n_bits = 0;
    unsigned f_w = 0;    // weight fractional bits
    unsigned f_act = 0;  // activation fractional bits
    std::size_t in_dim = 0;
    std::size_t out_dim = 0;
    FusedActivationKind act = FusedActivationKind::None;
};

struct FusedLayerKey {
    FusedLayerParams params;
    LRSKey trunc_key;
    std::vector<ActivationKey> act_keys0;
    std::vector<ActivationKey> act_keys1;
    std::uint64_t beaver_seed = 0;
};

struct FusedLayerKeyPair {
    FusedLayerKey k0;
    FusedLayerKey k1;
};

inline std::pair<Share, Share> fused_trunc_activation_eval(const RingConfig &cfg,
                                                           const LRSKey &t0,
                                                           const LRSKey &t1,
                                                           const ActivationKey *a0,
                                                           const ActivationKey *a1,
                                                           BeaverPool &pool0,
                                                           BeaverPool &pool1,
                                                           const Share &x0,
                                                           const Share &x1,
                                                           PdpfEngine &engine) {
    auto trunc_pair = lrs_eval_from_share_pair(cfg, t0, t1, x0, x1, engine);
    if (a0 == nullptr || a1 == nullptr) {
        return trunc_pair;
    }
    std::uint64_t x_hat = ring_add(cfg,
                                   ring_add(cfg, share_value(trunc_pair.first), share_value(trunc_pair.second)),
                                   a0->r_in);
    auto eval0 = activation_eval_main(0, *a0, x_hat, engine);
    auto eval1 = activation_eval_main(1, *a1, x_hat, engine);
    Share y0 = activation_finish(cfg, *a0, eval0, pool0);
    Share y1 = activation_finish(cfg, *a1, eval1, pool1);
    return {y0, y1};
}

inline FusedLayerKeyPair fused_layer_keygen(const FusedLayerParams &p,
                                            PdpfEngine &engine,
                                            std::mt19937_64 &rng) {
    FusedLayerKeyPair pair;
    pair.k0.params = pair.k1.params = p;
    MPCContext dealer_ctx(p.n_bits, rng());
    auto trunc_pair = lrs_gen(p.n_bits, p.f_w, engine, dealer_ctx);
    pair.k0.trunc_key = trunc_pair.k0;
    pair.k1.trunc_key = trunc_pair.k1;

    pair.k0.beaver_seed = rng();
    pair.k1.beaver_seed = pair.k0.beaver_seed;

    if (p.act != FusedActivationKind::None) {
        ActivationParams ap;
        ap.kind = (p.act == FusedActivationKind::GeLU) ? ActivationKind::GeLU : ActivationKind::SiLU;
        ap.n_bits = p.n_bits;
        ap.f = p.f_act;
        ap.lut_bits = 8;
        ap.clip = 3.0;
        auto akp = activation_gen(ap, engine, dealer_ctx);
        pair.k0.act_keys0.assign(p.out_dim, akp.k0);
        pair.k0.act_keys1.assign(p.out_dim, akp.k1);
        pair.k1.act_keys0.assign(p.out_dim, akp.k0);
        pair.k1.act_keys1.assign(p.out_dim, akp.k1);
    }
    return pair;
}

inline std::pair<std::vector<Share>, std::vector<Share>>
fused_layer_eval_pair(const FusedLayerKeyPair &keys,
                      const RingConfig &cfg,
                      const std::vector<Share> &x0,
                      const std::vector<Share> &x1,
                      const std::vector<std::vector<Share>> &W0,
                      const std::vector<std::vector<Share>> &W1,
                      const std::vector<Share> &b0,
                      const std::vector<Share> &b1,
                      PdpfEngine &engine) {
    const auto &k0 = keys.k0;
    const auto &k1 = keys.k1;
    std::size_t out_dim = k0.params.out_dim;
    std::size_t in_dim = k0.params.in_dim;
    std::vector<Share> y0(out_dim), y1(out_dim);
    if (x0.size() != in_dim || x1.size() != in_dim) return {y0, y1};

    BeaverPool pool0(cfg, k0.beaver_seed, 0);
    BeaverPool pool1(cfg, k1.beaver_seed, 1);

    for (std::size_t o = 0; o < out_dim; ++o) {
        Share acc0 = (o < b0.size()) ? b0[o] : constant(cfg, 0, 0);
        Share acc1 = (o < b1.size()) ? b1[o] : constant(cfg, 0, 1);
        for (std::size_t i = 0; i < in_dim; ++i) {
            Share w0 = (o < W0.size() && i < W0[o].size()) ? W0[o][i] : constant(cfg, 0, 0);
            Share w1 = (o < W1.size() && i < W1[o].size()) ? W1[o][i] : constant(cfg, 0, 1);
            auto prod = beaver_mul_pair_from_pools(cfg, pool0, pool1, w0, w1, x0[i], x1[i]);
            acc0 = add(cfg, acc0, prod.first);
            acc1 = add(cfg, acc1, prod.second);
        }
        const ActivationKey *a0 = (k0.params.act != FusedActivationKind::None && o < k0.act_keys0.size())
                                      ? &k0.act_keys0[o]
                                      : nullptr;
        const ActivationKey *a1 = (k1.params.act != FusedActivationKind::None && o < k1.act_keys1.size())
                                      ? &k1.act_keys1[o]
                                      : nullptr;
        auto out_pair = fused_trunc_activation_eval(cfg, k0.trunc_key, k1.trunc_key, a0, a1, pool0, pool1, acc0, acc1, engine);
        y0[o] = out_pair.first;
        y1[o] = out_pair.second;
    }
    return {y0, y1};
}

} // namespace cfss

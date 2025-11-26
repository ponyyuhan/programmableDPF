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
        pair.k0.act_keys0.resize(p.out_dim);
        pair.k0.act_keys1.resize(p.out_dim);
        pair.k1.act_keys0.resize(p.out_dim);
        pair.k1.act_keys1.resize(p.out_dim);
        for (std::size_t i = 0; i < p.out_dim; ++i) {
            auto akp = activation_gen(ap, engine, dealer_ctx);
            pair.k0.act_keys0[i] = akp.k0;
            pair.k0.act_keys1[i] = akp.k1;
            pair.k1.act_keys0[i] = akp.k0;
            pair.k1.act_keys1[i] = akp.k1;
        }
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
        auto trunc_pair = lrs_eval_from_share_pair(cfg, k0.trunc_key, k1.trunc_key, acc0, acc1, engine);
        Share act0 = trunc_pair.first;
        Share act1 = trunc_pair.second;

        if (k0.params.act != FusedActivationKind::None && o < k0.act_keys0.size()) {
            auto a0 = activation_eval_main(0, k0.act_keys0[o], ring_add(cfg, ring_add(cfg, share_value(act0), share_value(act1)), k0.act_keys0[o].r_in), engine);
            auto a1 = activation_eval_main(1, k1.act_keys1[o], ring_add(cfg, ring_add(cfg, share_value(act0), share_value(act1)), k1.act_keys1[o].r_in), engine);
            auto out_pair = activation_finish(cfg, k0.act_keys0[o], a0, pool0);
            auto out_pair_b = activation_finish(cfg, k1.act_keys1[o], a1, pool1);
            y0[o] = out_pair;
            y1[o] = out_pair_b;
        } else {
            y0[o] = act0;
            y1[o] = act1;
        }
    }
    return {y0, y1};
}

} // namespace cfss


#pragma once

#include "../arith.hpp"
#include "../beaver.hpp"
#include "../sharing.hpp"
#include "../gates/recip.hpp"
#include "../gates/trunc.hpp"
#include <vector>
#include <cmath>
#include <random>

namespace cfss {

struct NormParams {
    unsigned n_bits = 0;
    unsigned f = 0;          // fractional bits of inputs/outputs
    std::size_t dim = 0;     // vector length
    bool rms = false;        // true: RMSNorm (no mean subtraction)
    double eps = 1e-5;       // epsilon added to variance
};

struct NormKey {
    NormParams params;
    RecipKey inv_dim_key;    // 1 / dim in Q_f
    RecipKey inv_sqrt_key;   // 1 / sqrt(var) in Q_f (expects input in Q_{2f})
    LRSKey trunc_f;          // shift by f to drop one fixed-point scale
};

struct NormKeyPair {
    NormKey k0;
    NormKey k1;
};

inline NormKeyPair norm_keygen(const NormParams &p,
                               PdpfEngine &engine,
                               std::mt19937_64 &rng) {
    NormKeyPair kp;
    kp.k0.params = kp.k1.params = p;

    // Recip for 1/dim (input is integer, output Q_f).
    RecipParams rp_dim{p.n_bits, 0, p.f,
                       static_cast<unsigned>(p.dim > 0 ? p.dim : 1),
                       false};
    auto dim_keys = gen_recip_gate(rp_dim, engine, rng);

    // Recip for 1/sqrt(var). Input variance is roughly Q_{2f}, but cap to keep LUT small.
    unsigned f_in_var = static_cast<unsigned>(std::min<std::size_t>(p.f * 2, (p.n_bits > 3 ? p.n_bits - 3 : p.n_bits)));
    unsigned max_var = 256; // conservative cap to keep LUT size manageable in the adapter
    RecipParams rp_var{p.n_bits, f_in_var, p.f, max_var, true};
    auto var_keys = gen_recip_gate(rp_var, engine, rng);

    // Truncation by f to drop one fixed-point scale after multiplications.
    MPCContext dealer_ctx(p.n_bits, rng());
    auto trunc_pair = lrs_gen(p.n_bits, p.f, engine, dealer_ctx);

    kp.k0.inv_dim_key = dim_keys.k0;
    kp.k1.inv_dim_key = dim_keys.k1;
    kp.k0.inv_sqrt_key = var_keys.k0;
    kp.k1.inv_sqrt_key = var_keys.k1;
    kp.k0.trunc_f = trunc_pair.k0;
    kp.k1.trunc_f = trunc_pair.k1;
    return kp;
}

inline std::pair<std::vector<Share>, std::vector<Share>>
norm_eval_pair(const NormKeyPair &keys,
               const RingConfig &cfg,
               const std::vector<Share> &x0,
               const std::vector<Share> &x1,
               PdpfEngine &engine,
               BeaverPool &pool0,
               BeaverPool &pool1) {
    const auto &k0 = keys.k0;
    const auto &k1 = keys.k1;
    std::size_t dim = k0.params.dim;
    std::vector<Share> y0(dim), y1(dim);
    if (x0.size() != dim || x1.size() != dim) {
        return {y0, y1};
    }

    // Sum inputs.
    Share sum0 = x0[0];
    Share sum1 = x1[0];
    for (std::size_t i = 1; i < dim; ++i) {
        sum0 = add(cfg, sum0, x0[i]);
        sum1 = add(cfg, sum1, x1[i]);
    }

    // inv_dim in Q_f.
    auto inv_dim_pair = recip_eval_from_share_pair(cfg, k0.inv_dim_key, k1.inv_dim_key,
                                                   sum0, sum1, engine, pool0, pool1);

    // mean = sum * inv_dim (Q_f); shift down by f to return to Q_f.
    Share mean0 = constant(cfg, 0, 0);
    Share mean1 = constant(cfg, 0, 1);
    if (!k0.params.rms) {
        auto mean_prod = beaver_mul_pair_from_pools(cfg, pool0, pool1,
                                                    sum0, sum1,
                                                    inv_dim_pair.first, inv_dim_pair.second);
        auto mean_trunc = lrs_eval_from_share_pair(cfg, k0.trunc_f, k1.trunc_f,
                                                   mean_prod.first, mean_prod.second, engine);
        mean0 = mean_trunc.first;
        mean1 = mean_trunc.second;
    }

    // Variance computation.
    Share var0 = constant(cfg, 0, 0);
    Share var1 = constant(cfg, 0, 1);
    for (std::size_t i = 0; i < dim; ++i) {
        Share d0 = sub(cfg, x0[i], mean0);
        Share d1 = sub(cfg, x1[i], mean1);
        auto sq = beaver_mul_pair_from_pools(cfg, pool0, pool1, d0, d1, d0, d1);
        var0 = add(cfg, var0, sq.first);
        var1 = add(cfg, var1, sq.second);
    }

    // Average variance: (sum squares / dim).
    auto var_scaled = beaver_mul_pair_from_pools(cfg, pool0, pool1,
                                                 var0, var1,
                                                 inv_dim_pair.first, inv_dim_pair.second);
    auto var_avg = lrs_eval_from_share_pair(cfg, k0.trunc_f, k1.trunc_f,
                                            var_scaled.first, var_scaled.second, engine);

    // Add epsilon in Q_{2f} (var_avg currently Q_{2f}).
    if (k0.params.eps > 0.0) {
        double eps_scale = std::ldexp(k0.params.eps, static_cast<int>(k0.params.f * 2));
        std::int64_t eps_fp = static_cast<std::int64_t>(std::llround(eps_scale));
        var_avg.first = add_const(cfg, var_avg.first, static_cast<u64>(eps_fp));
        // party1 add zero (already implicit)
    }

    // inv_std in Q_f.
    auto inv_std_pair = recip_eval_from_share_pair(cfg, k0.inv_sqrt_key, k1.inv_sqrt_key,
                                                   var_avg.first, var_avg.second,
                                                   engine, pool0, pool1);

    // Normalize each coordinate: (x - mean) * inv_std, then truncate by f.
    std::vector<Share> prod0(dim), prod1(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        Share d0 = sub(cfg, x0[i], mean0);
        Share d1 = sub(cfg, x1[i], mean1);
        auto prod = beaver_mul_pair_from_pools(cfg, pool0, pool1,
                                               d0, d1,
                                               inv_std_pair.first, inv_std_pair.second);
        prod0[i] = prod.first;
        prod1[i] = prod.second;
    }
    auto norm_batch = lrs_eval_batch_from_share_pair(cfg, k0.trunc_f, k1.trunc_f,
                                                     prod0, prod1, engine);
    y0 = std::move(norm_batch.first);
    y1 = std::move(norm_batch.second);
    return {y0, y1};
}

} // namespace cfss

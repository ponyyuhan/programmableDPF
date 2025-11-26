#pragma once

#include "../pdpf_adapter.hpp"
#include "../arith.hpp"
#include "../beaver.hpp"
#include "../sharing.hpp"
#include "../suf.hpp"
#include "./gez.hpp"
#include "./trunc.hpp"
#include <limits>
#include <cmath>
#include <random>
#include <vector>

namespace cfss {

struct RecipParams {
    unsigned n_bits;     // ring bitwidth of inputs/outputs
    unsigned f_in;       // fractional bits in the input
    unsigned f_out;      // fractional bits in the output
    unsigned max_d;      // maximum |x| in real units before masking (Q_{f_in})
    bool rsqrt = false;  // optional future extension
};

struct RecipKey {
    RecipParams params;
    InvGateKey inv_key;
    unsigned shift_down = 0;
    // Bound applied to |x| before inversion (in full fixed-point units) and after any pre-shift.
    std::uint64_t bound_raw = 0;
    std::uint64_t bound_scaled = 0;
    GEZKey sign_gez;
    GEZKey clamp_gez;
    std::optional<LRSKey> pre_lrs;
};

struct RecipKeyPair {
    RecipKey k0;
    RecipKey k1;
};

inline std::pair<Share, Share> beaver_select_pair_from_pools(const RingConfig &cfg,
                                                             BeaverPool &pool0,
                                                             BeaverPool &pool1,
                                                             const Share &bit0,
                                                             const Share &bit1,
                                                             const Share &a0,
                                                             const Share &a1,
                                                             const Share &b0,
                                                             const Share &b1) {
    Share diff0 = sub(cfg, a0, b0);
    Share diff1 = sub(cfg, a1, b1);
    auto prod_pair = beaver_mul_pair_from_pools(cfg, pool0, pool1, bit0, bit1, diff0, diff1);
    return {add(cfg, b0, prod_pair.first), add(cfg, b1, prod_pair.second)};
}

inline RecipKeyPair gen_recip_gate(const RecipParams &params,
                                   PdpfEngine &engine,
                                   std::mt19937_64 &rng) {
    // Downscale the input if f_in + f_out would overflow the ring.
    int shift_down = 0;
    const int headroom = 2; // leave at least two bits to reduce wraparound risk
    if (static_cast<int>(params.f_in + params.f_out) > static_cast<int>(params.n_bits) - headroom) {
        shift_down = static_cast<int>(params.f_in + params.f_out - (params.n_bits - headroom));
    }
    if (shift_down > static_cast<int>(params.f_in)) {
        shift_down = static_cast<int>(params.f_in);
    }
    if (shift_down < 0) shift_down = 0;
    unsigned shift_u = static_cast<unsigned>(shift_down);

    // Effective fractional bits after any pre-shift.
    unsigned f_in_eff = (params.f_in > shift_u) ? (params.f_in - shift_u) : 0;

    auto saturating_shift = [](std::uint64_t v, unsigned shift, std::uint64_t max_val) {
        if (shift >= 63) return max_val;
        if (v > (max_val >> shift)) return max_val;
        return (v << shift);
    };

    std::uint64_t ring_max = (params.n_bits == 64) ? ~0ULL : ((1ULL << params.n_bits) - 1ULL);
    std::uint64_t max_pos = (params.n_bits == 64) ? (std::numeric_limits<std::uint64_t>::max() >> 1)
                                                  : ((1ULL << (params.n_bits - 1)) - 1ULL);

    std::uint64_t bound_raw = saturating_shift(static_cast<std::uint64_t>(params.max_d),
                                               params.f_in, max_pos);
    std::uint64_t bound_scaled = saturating_shift(static_cast<std::uint64_t>(params.max_d),
                                                  f_in_eff, max_pos);
    if (bound_scaled == 0) bound_scaled = 1;

    // The Inv gate outputs in Q_{f_out + f_in - shift_down}, so we later truncate by f_in.
    unsigned inv_f = params.f_out + f_in_eff;
    if (inv_f > params.n_bits - headroom) {
        inv_f = params.n_bits - headroom;
    }
    if (bound_scaled == 0) bound_scaled = 1;
    std::uint64_t bound_inv = std::min<std::uint64_t>(
        bound_scaled, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max() - 1));
    InvGateParams ip{params.n_bits, inv_f, static_cast<unsigned>(bound_inv)};

    auto inv_pair = gen_inv_gate(ip, engine, rng);
    MPCContext dealer_ctx(params.n_bits, rng());

    // Sign and clamp gates share randomness across parties.
    GEZParams gp{params.n_bits};
    auto sign_keys = gez_gen(gp, engine, dealer_ctx);
    auto clamp_keys = gez_gen(gp, engine, dealer_ctx);

    // Optional pre-shift of the input.
    std::optional<LRSKeyPair> pre_lrs_pair;
    if (shift_u > 0) {
        pre_lrs_pair = lrs_gen(params.n_bits, shift_u, engine, dealer_ctx);
    }

    RecipKey k0;
    k0.params = params;
    k0.inv_key = inv_pair.k0;
    k0.shift_down = shift_u;
    k0.bound_raw = bound_raw;
    k0.bound_scaled = bound_scaled;
    k0.sign_gez = sign_keys.k0;
    k0.clamp_gez = clamp_keys.k0;
    if (pre_lrs_pair.has_value()) k0.pre_lrs = pre_lrs_pair->k0;

    RecipKey k1 = k0;
    k1.inv_key = inv_pair.k1;
    if (pre_lrs_pair.has_value()) {
        k1.pre_lrs = pre_lrs_pair->k1;
    } else {
        k1.pre_lrs.reset();
    }
    return RecipKeyPair{k0, k1};
}

inline std::pair<Share, Share> recip_eval_from_share_pair(const RingConfig &cfg,
                                                          const RecipKey &k0,
                                                          const RecipKey &k1,
                                                          const Share &x0,
                                                          const Share &x1,
                                                          PdpfEngine &engine,
                                                          BeaverPool &pool0,
                                                          BeaverPool &pool1) {
    // Sign bit: 1 if x >= 0.
    std::uint64_t x_hat = ring_add(cfg, ring_add(cfg, share_value(x0), share_value(x1)), k0.sign_gez.r_in);
    MPCContext sign_ctx0(cfg.n_bits, 0xC011AB5E);
    MPCContext sign_ctx1(cfg.n_bits, 0xC011AB5F);
    Share sign0 = gez_eval(0, k0.sign_gez, x_hat, engine, sign_ctx0);
    Share sign1 = gez_eval(1, k1.sign_gez, x_hat, engine, sign_ctx1);

    // Absolute value via Beaver select: sign ? x : -x.
    Share neg0 = negate(cfg, x0);
    Share neg1 = negate(cfg, x1);
    auto abs_pair = beaver_select_pair_from_pools(cfg, pool0, pool1, sign0, sign1,
                                                  x0, x1, neg0, neg1);
    Share abs0 = abs_pair.first;
    Share abs1 = abs_pair.second;

    // Clamp bit: 1 if |x| <= bound_raw. If bound_raw spans the full positive range,
    // skip the GEZ and treat clamp as always true to avoid signed wraparound.
    std::uint64_t max_pos = (cfg.n_bits == 64) ? (std::numeric_limits<std::uint64_t>::max() >> 1)
                                               : ((1ULL << (cfg.n_bits - 1)) - 1ULL);
    Share clamp0, clamp1;
    if (k0.bound_raw >= max_pos) {
        clamp0 = constant(cfg, 1, 0);
        clamp1 = constant(cfg, 0, 1);
    } else {
        Share bound0 = constant(cfg, k0.bound_raw, 0);
        Share bound1 = constant(cfg, 0, 1);
        Share diff0 = sub(cfg, bound0, abs0);
        Share diff1 = sub(cfg, bound1, abs1);
        std::uint64_t diff_hat = ring_add(cfg, ring_add(cfg, share_value(diff0), share_value(diff1)), k0.clamp_gez.r_in);
        MPCContext clamp_ctx0(cfg.n_bits, 0xC0C1C2C3);
        MPCContext clamp_ctx1(cfg.n_bits, 0xD0D1D2D3);
        clamp0 = gez_eval(0, k0.clamp_gez, diff_hat, engine, clamp_ctx0);
        clamp1 = gez_eval(1, k1.clamp_gez, diff_hat, engine, clamp_ctx1);
    }

    // Optional logical shift of the input to fit the inversion domain.
    Share inv_in0 = abs0;
    Share inv_in1 = abs1;
    if (k0.shift_down > 0 && k0.pre_lrs.has_value() && k1.pre_lrs.has_value()) {
        auto scaled_pair = lrs_eval_from_share_pair(cfg, *k0.pre_lrs, *k1.pre_lrs, abs0, abs1, engine);
        inv_in0 = scaled_pair.first;
        inv_in1 = scaled_pair.second;
    }

    // Inversion on the scaled magnitude.
    auto inv_pair = invgate_eval_from_share_pair(cfg, k0.inv_key, k1.inv_key, inv_in0, inv_in1, engine);

    // Clamp: if |x| > bound, return 0.
    Share zero0 = constant(cfg, 0, 0);
    Share zero1 = constant(cfg, 0, 1);
    auto clamped_pair = beaver_select_pair_from_pools(cfg, pool0, pool1, clamp0, clamp1,
                                                      inv_pair.first, inv_pair.second,
                                                      zero0, zero1);

    // Restore sign: sign ? (+) : (-).
    Share neg_res0 = negate(cfg, clamped_pair.first);
    Share neg_res1 = negate(cfg, clamped_pair.second);
    auto signed_pair = beaver_select_pair_from_pools(cfg, pool0, pool1, sign0, sign1,
                                                     clamped_pair.first, clamped_pair.second,
                                                     neg_res0, neg_res1);
    return signed_pair;
}

} // namespace cfss

#pragma once

#include "../pdpf_adapter.hpp"
#include "../arith.hpp"
#include "../sharing.hpp"
#include "../suf.hpp"
#include <cmath>
#include <random>
#include <vector>

namespace cfss {

struct RecipParams {
    unsigned n_bits;     // ring bitwidth of inputs/outputs
    unsigned f_in;       // fractional bits in the input
    unsigned f_out;      // fractional bits in the output
    unsigned max_d;      // maximum input magnitude (raw, before masking)
    bool rsqrt = false;  // optional future extension
};

struct RecipKey {
    RecipParams params;
    InvGateKey inv_key;
    unsigned shift_down = 0;
    std::optional<LRSKey> pre_lrs;
};

struct RecipKeyPair {
    RecipKey k0;
    RecipKey k1;
};

inline RecipKeyPair gen_recip_gate(const RecipParams &params,
                                   PdpfEngine &engine,
                                   std::mt19937_64 &rng) {
    // To stay within the output ring, downscale the input if needed.
    // shift_down reduces the effective fractional bits to fit f_in+f_out under n_bits.
    int shift_down = 0;
    if (params.f_in + params.f_out > params.n_bits - 2) {
        shift_down = static_cast<int>(params.f_in + params.f_out - (params.n_bits - 2));
    }
    unsigned f_in_eff = static_cast<unsigned>(params.f_in > static_cast<unsigned>(shift_down)
                                              ? params.f_in - shift_down
                                              : 0);
    // Bound expressed in the reduced fixed-point units.
    std::uint64_t bound_fp = static_cast<std::uint64_t>(params.max_d) << f_in_eff;
    if (bound_fp >= (1ULL << params.n_bits)) {
        bound_fp = (1ULL << params.n_bits) - 1;
    }
    InvGateParams ip{params.n_bits, params.f_out,
                     static_cast<unsigned>(bound_fp)};
    auto inv_pair = gen_inv_gate(ip, engine, rng);
    RecipKeyPair pair;
    RecipKey k;
    k.params = params;
    k.inv_key = inv_pair.k0;
    k.shift_down = static_cast<unsigned>(shift_down > 0 ? shift_down : 0);
    if (shift_down > 0) {
        MPCContext dealer_ctx(params.n_bits, rng());
        auto lrs_keys = lrs_gen(params.n_bits, static_cast<unsigned>(shift_down), engine, dealer_ctx);
        k.pre_lrs = lrs_keys.k0;
        RecipKey k1 = k;
        k1.inv_key = inv_pair.k1;
        k1.pre_lrs = lrs_keys.k1;
        pair.k0 = k;
        pair.k1 = k1;
    } else {
        RecipKey k1 = k;
        k1.inv_key = inv_pair.k1;
        pair.k0 = k;
        pair.k1 = k1;
    }
    return pair;
}

inline std::pair<Share, Share> recip_eval_from_share_pair(const RingConfig &cfg,
                                                          const RecipKey &k0,
                                                          const RecipKey &k1,
                                                          const Share &x0,
                                                          const Share &x1,
                                                          PdpfEngine &engine) {
    // If f_in was downscaled at keygen time, pre-scale the input shares accordingly.
    Share x0_scaled = x0;
    Share x1_scaled = x1;
    if (k0.shift_down > 0 && k0.pre_lrs.has_value()) {
        auto scaled_pair = lrs_eval_from_share_pair(cfg, *k0.pre_lrs, *k1.pre_lrs, x0, x1, engine);
        x0_scaled = scaled_pair.first;
        x1_scaled = scaled_pair.second;
    }
    return invgate_eval_from_share_pair(cfg, k0.inv_key, k1.inv_key, x0_scaled, x1_scaled, engine);
}

} // namespace cfss

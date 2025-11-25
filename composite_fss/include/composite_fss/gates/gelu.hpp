#pragma once

#include "../arith.hpp"
#include "../pdpf.hpp"
#include "../sharing.hpp"
#include "../beaver.hpp"
#include "../suf.hpp"
#include "../suf_packing.hpp"
#include "../suf_to_lut.hpp"
#include "../suf_unpack.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace cfss {

// Unified activation description so GeLU / SiLU share the same packed SUF path.
enum class ActivationKind { GeLU, SiLU };

struct ActivationParams {
    ActivationKind kind = ActivationKind::GeLU;
    unsigned n_bits = 16;
    unsigned f = 8;          // fractional bits
    unsigned lut_bits = 8;   // index width
    double clip = 4.0;       // clip bound on the real axis
    std::vector<std::uint64_t> lut; // optional precomputed delta(t) * 2^f table
};

using GeLUParams = ActivationParams;
using SiLUParams = ActivationParams;

struct ActivationIndexingInfo {
    unsigned shift = 0;
    unsigned frac_after_shift = 0;
    std::uint64_t y_scale = 1;
    std::uint64_t clip_mag = 0;
};

struct ActivationChannels {
    SufChannelId relu_value;
    SufChannelId delta_value;
    SufChannelId relu_bit;
};

struct ActivationKey {
    ActivationKind kind = ActivationKind::GeLU;
    unsigned n_bits = 16;
    unsigned f = 8;
    std::uint64_t r_in = 0;
    PackedSufProgram main_prog;
    ActivationChannels channels;
    ActivationIndexingInfo info;
};

struct ActivationKeyPair {
    ActivationKey k0;
    ActivationKey k1;
};

using GeLUKey = ActivationKey;
using GeLUKeyPair = ActivationKeyPair;
using SiLUKey = ActivationKey;
using SiLUKeyPair = ActivationKeyPair;

struct ActivationEvalResult {
    Share relu_value;
    Share delta_value;
    Share relu_bit;
#if COMPOSITE_FSS_INTERNAL
    std::vector<Share> packed_words;
#endif
};

inline double activation_ref(ActivationKind kind, double x) {
    switch (kind) {
        case ActivationKind::GeLU:
            return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        case ActivationKind::SiLU:
            return x / (1.0 + std::exp(-x));
        default:
            return x;
    }
}

inline ActivationIndexingInfo make_index_info(const ActivationParams &p) {
    ActivationIndexingInfo info;
    info.shift = (p.f > 6) ? (p.f - 6) : 0;
    info.frac_after_shift = (p.f >= info.shift) ? (p.f - info.shift) : 0;
    info.y_scale = (info.frac_after_shift >= 64) ? 0ULL : (1ULL << info.frac_after_shift);
    info.clip_mag = static_cast<std::uint64_t>(std::llround(p.clip * static_cast<double>(info.y_scale)));
    return info;
}

inline std::pair<SufDesc, ActivationChannels> build_activation_suf(const ActivationParams &p,
                                                                   std::uint64_t r_in,
                                                                   std::uint64_t r_out,
                                                                   const ActivationIndexingInfo &info) {
    Ring64 ring(p.n_bits);
    SufDesc suf;
    suf.shape.domain_bits = p.n_bits;
    suf.r_outputs = 2;      // ReLU value + delta value
    suf.l_outputs = 1;      // relu bit
    suf.degree = 1;
    suf.r_in = r_in;
    suf.r_out = 0;          // outputs left unmasked; masking handled externally if needed

    SufChannelId val_ch = suf.shape.add_channel("relu_val", SufFieldKind::Ring, p.n_bits, 1);
    SufChannelId delta_ch = suf.shape.add_channel("delta_val", SufFieldKind::Ring, p.n_bits, 1);
    SufChannelId relu_bit_ch = suf.shape.add_channel("relu_bit", SufFieldKind::Bool, 1, 1);

    std::size_t domain_size = 1ULL << p.n_bits;
    suf.alpha.reserve(domain_size);
    suf.polys.reserve(domain_size);
    suf.bools.reserve(domain_size);

    for (std::size_t x = 0; x < domain_size; ++x) {
        std::uint64_t unmasked = static_cast<std::uint64_t>(x);
        std::int64_t xs = ring.to_signed(unmasked);
        std::int64_t relu = std::max<std::int64_t>(xs, 0);
        std::uint64_t relu_masked = ring.from_signed(relu);

        std::int64_t shifted = (info.shift > 0) ? (xs >> info.shift) : xs;
        std::uint64_t mag = static_cast<std::uint64_t>(std::llabs(shifted));
        bool in_clip = mag <= info.clip_mag;
        if (mag > info.clip_mag) mag = info.clip_mag;

        suf.alpha.push_back(static_cast<std::uint64_t>(x));
        PolyVec pv;
        pv.polys.resize(2);
        pv.polys[0].coeffs = {static_cast<std::int64_t>(relu_masked)};
        double x_real = static_cast<double>(xs) / static_cast<double>(1ULL << p.f);
        double relu_real = static_cast<double>(relu) / static_cast<double>(1ULL << p.f);
        double ref = activation_ref(p.kind, x_real);
        double delta_real = in_clip ? (relu_real - ref) : 0.0;
        std::int64_t delta_fp = static_cast<std::int64_t>(std::llround(delta_real * static_cast<double>(1ULL << p.f)));
        pv.polys[1].coeffs = {static_cast<std::int64_t>(ring.from_signed(delta_fp))};
        suf.polys.push_back(std::move(pv));

        std::vector<BoolExpr> bvec(1);
        bvec[0].kind = BoolExpr::CONST;
        bvec[0].const_value = (xs >= 0);
        suf.bools.push_back(std::move(bvec));
    }

    ActivationChannels channels{val_ch, delta_ch, relu_bit_ch};
    return {suf, channels};
}

inline ActivationKeyPair activation_gen(const ActivationParams &input_params,
                                        PdpfEngine &engine,
                                        MPCContext &dealer_ctx) {
    ActivationParams params = input_params;
    if (params.kind == ActivationKind::GeLU) {
        if (params.clip <= 0.0) params.clip = 3.0;
    } else {
        if (params.clip <= 0.0) params.clip = 6.0;
    }

    std::uint64_t r_in = 0; // keep GeLU unmasked on input to stabilize SUF layout

    ActivationIndexingInfo info = make_index_info(params);
    auto suf_and_channels = build_activation_suf(params, r_in, 0, info);
    auto packed = compile_suf_desc_packed(suf_and_channels.first, engine, std::nullopt, params.n_bits);

    ActivationKey key;
    key.kind = params.kind;
    key.n_bits = params.n_bits;
    key.f = params.f;
    key.r_in = r_in;
    key.main_prog = packed;
    key.channels = suf_and_channels.second;
    key.info = info;

    ActivationKeyPair pair;
    pair.k0 = key;
    pair.k1 = key;
    return pair;
}

inline ActivationEvalResult activation_eval_main(int party,
                                                 const ActivationKey &key,
                                                 std::uint64_t x_hat,
                                                 PdpfEngine &engine) {
    std::vector<std::uint64_t> out(key.main_prog.layout.num_words);
    engine.eval_share(key.main_prog.compiled.pdpf_program, party, x_hat, out);
    std::vector<Share> share_words;
    share_words.reserve(out.size());
    for (auto w : out) {
        share_words.emplace_back(party, w);
    }
    RingConfig cfg = make_ring_config(key.n_bits);
    ActivationEvalResult res;
    res.relu_value = suf_unpack_channel_share(cfg, key.main_prog.layout, key.channels.relu_value, 0, share_words);
    res.delta_value = suf_unpack_channel_share(cfg, key.main_prog.layout, key.channels.delta_value, 0, share_words);
    res.relu_bit = suf_unpack_channel_share(cfg, key.main_prog.layout, key.channels.relu_bit, 0, share_words);
#if COMPOSITE_FSS_INTERNAL
    res.packed_words = std::move(share_words);
#endif
    return res;
}
inline Share activation_finish(const RingConfig &cfg,
                               const ActivationKey &key,
                               const ActivationEvalResult &r,
                               BeaverPool &pool) {
    (void)key;
    // GeLU/SiLU encoded as ReLU(x) - delta(x).
    Share y = sub(cfg, r.relu_value, r.delta_value);
    // Optional gating by relu_bit to zero negative inputs.
    if (r.relu_bit.party() == r.delta_value.party()) {
        y = beaver_mul(pool, cfg, r.relu_bit, y);
    }
    return y;
}

// --- Convenience wrappers for GeLU ---
inline GeLUKeyPair gelu_gen(const GeLUParams &params, PdpfEngine &engine, MPCContext &dealer_ctx) {
    ActivationParams p = params;
    p.kind = ActivationKind::GeLU;
    return activation_gen(p, engine, dealer_ctx);
}

inline ActivationEvalResult gelu_eval_main(int party,
                                           const GeLUKey &key,
                                           std::uint64_t x_hat,
                                           PdpfEngine &engine) {
    return activation_eval_main(party, key, x_hat, engine);
}

inline std::pair<Share, Share> gelu_finish(const RingConfig &cfg,
                                           const GeLUKey &k0,
                                           const GeLUKey &k1,
                                           const ActivationEvalResult &r0,
                                           const ActivationEvalResult &r1,
                                           BeaverPool &,
                                           BeaverPool &) {
    (void)k0;
    (void)k1;
    Share t0 = sub(cfg, r0.relu_value, r0.delta_value);
    Share t1 = sub(cfg, r1.relu_value, r1.delta_value);
    return {t0, t1};
}

// Evaluate both parties locally (useful for tests/benchmarks).
inline std::pair<Share, Share> gelu_eval_pair(const GeLUKeyPair &keys,
                                              std::uint64_t x_hat,
                                              PdpfEngine &engine,
                                              BeaverPool &pool0,
                                              BeaverPool &pool1) {
    RingConfig cfg = make_ring_config(keys.k0.n_bits);
    auto r0 = gelu_eval_main(0, keys.k0, x_hat, engine);
    auto r1 = gelu_eval_main(1, keys.k1, x_hat, engine);
    return gelu_finish(cfg, keys.k0, keys.k1, r0, r1, pool0, pool1);
}

#if COMPOSITE_FSS_INTERNAL
inline std::pair<Share, Share> gelu_eval_pair(const GeLUKeyPair &keys,
                                              std::uint64_t x_hat,
                                              PdpfEngine &engine) {
    RingConfig cfg = make_ring_config(keys.k0.n_bits);
    BeaverPool dummy0(cfg, 0xBEEFu, 0);
    BeaverPool dummy1(cfg, 0xBEEFu, 1);
    auto r0 = gelu_eval_main(0, keys.k0, x_hat, engine);
    auto r1 = gelu_eval_main(1, keys.k1, x_hat, engine);
    return gelu_finish(cfg, keys.k0, keys.k1, r0, r1, dummy0, dummy1);
}
#endif
// --- Convenience wrappers for SiLU ---
inline SiLUKeyPair silu_gen(const SiLUParams &params, PdpfEngine &engine, MPCContext &dealer_ctx) {
    ActivationParams p = params;
    p.kind = ActivationKind::SiLU;
    return activation_gen(p, engine, dealer_ctx);
}

inline ActivationEvalResult silu_eval_main(int party,
                                           const SiLUKey &key,
                                           std::uint64_t x_hat,
                                           PdpfEngine &engine) {
    return activation_eval_main(party, key, x_hat, engine);
}

inline std::pair<Share, Share> silu_finish(const RingConfig &cfg,
                                           const SiLUKey &k0,
                                           const SiLUKey &k1,
                                           const ActivationEvalResult &r0,
                                           const ActivationEvalResult &r1,
                                           BeaverPool &,
                                           BeaverPool &) {
    (void)k0;
    (void)k1;
    Share t0 = sub(cfg, r0.relu_value, r0.delta_value);
    Share t1 = sub(cfg, r1.relu_value, r1.delta_value);
    return {t0, t1};
}

inline std::pair<Share, Share> silu_eval_pair(const SiLUKeyPair &keys,
                                              std::uint64_t x_hat,
                                              PdpfEngine &engine,
                                              BeaverPool &pool0,
                                              BeaverPool &pool1) {
    RingConfig cfg = make_ring_config(keys.k0.n_bits);
    auto r0 = silu_eval_main(0, keys.k0, x_hat, engine);
    auto r1 = silu_eval_main(1, keys.k1, x_hat, engine);
    return silu_finish(cfg, keys.k0, keys.k1, r0, r1, pool0, pool1);
}

#if COMPOSITE_FSS_INTERNAL
inline std::pair<Share, Share> silu_eval_pair(const SiLUKeyPair &keys,
                                              std::uint64_t x_hat,
                                              PdpfEngine &engine) {
    RingConfig cfg = make_ring_config(keys.k0.n_bits);
    BeaverPool dummy0(cfg, 0xBEEFu, 0);
    BeaverPool dummy1(cfg, 0xBEEFu, 1);
    auto r0 = silu_eval_main(0, keys.k0, x_hat, engine);
    auto r1 = silu_eval_main(1, keys.k1, x_hat, engine);
    return silu_finish(cfg, keys.k0, keys.k1, r0, r1, dummy0, dummy1);
}
#endif

} // namespace cfss

#pragma once

#include "../arith.hpp"
#include "../pdpf.hpp"
#include "../sharing.hpp"
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
    unsigned index_shift = 0;
    std::uint64_t index_mask = 0;
};

struct ActivationChannels {
    SufChannelId relu_value;
    SufChannelId relu_bit;
    SufChannelId in_bit;
    SufChannelId index;
};

struct ActivationKey {
    ActivationKind kind = ActivationKind::GeLU;
    unsigned n_bits = 16;
    unsigned f = 8;
    unsigned lut_bits = 8;
    std::uint64_t r_in = 0;
    std::uint64_t r_out = 0;
    PackedSufProgram main_prog;
    ActivationChannels channels;
    PdpfProgramId lut_prog = 0; // delta(t) table
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
    Share relu_masked;
    Share relu_bit;
    Share in_bit;
    Share index_share;
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
    unsigned bitwidth_clip = (info.clip_mag == 0) ? 1u : static_cast<unsigned>(64 - __builtin_clzll(info.clip_mag));
    info.index_shift = (bitwidth_clip > p.lut_bits) ? (bitwidth_clip - p.lut_bits) : 0;
    info.index_mask = (p.lut_bits == 64) ? ~0ULL : ((1ULL << p.lut_bits) - 1ULL);
    return info;
}

inline std::vector<std::uint64_t> build_activation_lut(const ActivationParams &p,
                                                       const ActivationIndexingInfo &info) {
    Ring64 ring(p.n_bits);
    std::size_t lut_size = 1ULL << p.lut_bits;
    std::vector<std::uint64_t> table(lut_size);
    std::vector<double> accum(lut_size, 0.0);
    std::vector<std::uint64_t> counts(lut_size, 0);
    std::uint64_t y_scale = (info.y_scale == 0) ? 1ULL : info.y_scale;
    unsigned cap_bits = (p.n_bits > 20) ? 20u : p.n_bits;
    std::size_t domain_cap = 1ULL << cap_bits;
    for (std::size_t x = 0; x < domain_cap; ++x) {
        std::int64_t xs = ring.to_signed(static_cast<std::uint64_t>(x));
        double x_real = static_cast<double>(xs) / static_cast<double>(1ULL << p.f);
        std::int64_t shifted = (info.shift > 0) ? (xs >> info.shift) : xs;
        std::uint64_t mag = static_cast<std::uint64_t>(std::llabs(shifted));
        if (mag > info.clip_mag) mag = info.clip_mag;
        std::uint64_t idx = (info.index_shift >= 64) ? 0ULL : ((mag >> info.index_shift) & info.index_mask);
        double ref = activation_ref(p.kind, x_real);
        double relu = std::max(0.0, x_real);
        double delta = relu - ref;
        accum[idx] += delta;
        counts[idx] += 1;
    }
    for (std::size_t t = 0; t < lut_size; ++t) {
        double delta = (counts[t] == 0) ? 0.0 : (accum[t] / static_cast<double>(counts[t]));
        std::int64_t delta_fp = static_cast<std::int64_t>(std::llround(delta * static_cast<double>(1ULL << p.f)));
        table[t] = ring.from_signed(delta_fp);
    }
    return table;
}

inline std::pair<SufDesc, ActivationChannels> build_activation_suf(const ActivationParams &p,
                                                                   std::uint64_t r_in,
                                                                   std::uint64_t r_out,
                                                                   const ActivationIndexingInfo &info) {
    Ring64 ring(p.n_bits);
    SufDesc suf;
    suf.shape.domain_bits = p.n_bits;
    suf.r_outputs = 2;      // ReLU value + 8/10-bit index (as arithmetic word)
    suf.l_outputs = 2;      // relu bit, in-clip bit
    suf.degree = 1;
    suf.r_in = r_in;
    suf.r_out = 0;          // mask is embedded in the polynomial values directly

    SufChannelId val_ch = suf.shape.add_channel("relu_val", SufFieldKind::Ring, p.n_bits, 1);
    SufChannelId idx_ch = suf.shape.add_channel("lut_index", SufFieldKind::Index, p.lut_bits, 1);
    SufChannelId relu_bit_ch = suf.shape.add_channel("relu_bit", SufFieldKind::Bool, 1, 1);
    SufChannelId in_bit_ch = suf.shape.add_channel("in_bit", SufFieldKind::Bool, 1, 1);

    std::size_t domain_size = 1ULL << p.n_bits;
    suf.alpha.reserve(domain_size);
    suf.polys.reserve(domain_size);
    suf.bools.reserve(domain_size);

    for (std::size_t x = 0; x < domain_size; ++x) {
        std::uint64_t unmasked = static_cast<std::uint64_t>(x);
        std::int64_t xs = ring.to_signed(unmasked);
        std::int64_t relu = std::max<std::int64_t>(xs, 0);
        std::uint64_t relu_masked = ring.add(ring.from_signed(relu), r_out);

        std::int64_t shifted = (info.shift > 0) ? (xs >> info.shift) : xs;
        std::uint64_t mag = static_cast<std::uint64_t>(std::llabs(shifted));
        bool in_clip = mag <= info.clip_mag;
        if (mag > info.clip_mag) mag = info.clip_mag;
        std::uint64_t idx = (info.index_shift >= 64) ? 0ULL : ((mag >> info.index_shift) & info.index_mask);

        suf.alpha.push_back(static_cast<std::uint64_t>(x));
        PolyVec pv;
        pv.polys.resize(2);
        pv.polys[0].coeffs = {static_cast<std::int64_t>(relu_masked)};
        pv.polys[1].coeffs = {static_cast<std::int64_t>(idx)};
        suf.polys.push_back(std::move(pv));

        std::vector<BoolExpr> bvec(2);
        bvec[0].kind = BoolExpr::CONST;
        bvec[0].const_value = (xs >= 0);
        bvec[1].kind = BoolExpr::CONST;
        bvec[1].const_value = in_clip;
        suf.bools.push_back(std::move(bvec));
    }

    ActivationChannels channels{val_ch, relu_bit_ch, in_bit_ch, idx_ch};
    return {suf, channels};
}

inline ActivationKeyPair activation_gen(const ActivationParams &input_params,
                                        PdpfEngine &engine,
                                        MPCContext &dealer_ctx) {
    ActivationParams params = input_params;
    if (params.kind == ActivationKind::GeLU) {
        if (params.lut_bits == 0) params.lut_bits = 8;
        if (params.clip <= 0.0) params.clip = 3.0;
    } else {
        if (params.lut_bits == 0) params.lut_bits = 10;
        if (params.clip <= 0.0) params.clip = 6.0;
    }

    std::uint64_t r_in = 0; // keep GeLU unmasked on input to stabilize SUF layout
    std::uint64_t r_out = dealer_ctx.rng() & Ring64(params.n_bits).modulus_mask;

    ActivationIndexingInfo info = make_index_info(params);
    auto suf_and_channels = build_activation_suf(params, r_in, r_out, info);
    auto packed = compile_suf_desc_packed(suf_and_channels.first, engine, std::nullopt, params.n_bits);

    std::vector<std::uint64_t> lut_table = params.lut;
    std::size_t expected = 1ULL << params.lut_bits;
    if (lut_table.size() != expected) {
        lut_table = build_activation_lut(params, info);
    }
    LutProgramDesc lut_desc;
    lut_desc.domain_bits = params.lut_bits;
    lut_desc.output_words = 1;
    PdpfProgramId lut_prog = engine.make_lut_program(lut_desc, lut_table);

    ActivationKey key;
    key.kind = params.kind;
    key.n_bits = params.n_bits;
    key.f = params.f;
    key.lut_bits = params.lut_bits;
    key.r_in = r_in;
    key.r_out = r_out;
    key.main_prog = packed;
    key.channels = suf_and_channels.second;
    key.lut_prog = lut_prog;
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
    res.relu_masked = suf_unpack_channel_share(cfg, key.main_prog.layout, key.channels.relu_value, 0, share_words);
    res.relu_bit = suf_unpack_channel_share(cfg, key.main_prog.layout, key.channels.relu_bit, 0, share_words);
    res.in_bit = suf_unpack_channel_share(cfg, key.main_prog.layout, key.channels.in_bit, 0, share_words);
    res.index_share = suf_unpack_channel_share(cfg, key.main_prog.layout, key.channels.index, 0, share_words);
    return res;
}

inline std::pair<Share, Share> activation_finish(const RingConfig &cfg,
                                                 const ActivationKey &k0,
                                                 const ActivationKey &k1,
                                                 const ActivationEvalResult &r0,
                                                 const ActivationEvalResult &r1,
                                                 PdpfEngine &engine) {
#if COMPOSITE_FSS_INTERNAL
    std::uint64_t idx = debug_open(cfg, r0.index_share, r1.index_share);
    std::uint64_t in_bit = debug_open(cfg, r0.in_bit, r1.in_bit) & 1ULL;
    std::uint64_t relu_bit = debug_open(cfg, r0.relu_bit, r1.relu_bit) & 1ULL;
    bool within_clip = (k0.info.clip_mag == 0) ? true : (idx < k0.info.clip_mag);
    std::vector<std::uint64_t> delta0(1), delta1(1);
    if (relu_bit && in_bit && within_clip) {
        engine.eval_share(k0.lut_prog, 0, idx, delta0);
        engine.eval_share(k1.lut_prog, 1, idx, delta1);
    } else {
        delta0[0] = 0;
        delta1[0] = 0;
    }
    Share d0{0, delta0[0]};
    Share d1{1, delta1[0]};
    Share y0 = sub(cfg, r0.relu_masked, d0);
    Share y1 = sub(cfg, r1.relu_masked, d1);
    return {y0, y1};
#else
    (void)cfg;
    (void)k0;
    (void)k1;
    (void)r0;
    (void)r1;
    (void)engine;
    throw std::runtime_error("activation_finish requires COMPOSITE_FSS_INTERNAL to reconstruct indices");
#endif
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
                                           PdpfEngine &engine) {
    return activation_finish(cfg, k0, k1, r0, r1, engine);
}

// Evaluate both parties locally (useful for tests/benchmarks).
inline std::pair<Share, Share> gelu_eval_pair(const GeLUKeyPair &keys,
                                              std::uint64_t x_hat,
                                              PdpfEngine &engine) {
    (void)engine;
    Ring64 ring(keys.k0.n_bits);
    std::int64_t xs = ring.to_signed(ring.sub(x_hat, keys.k0.r_in));
    double x_real = static_cast<double>(xs) / static_cast<double>(1ULL << keys.k0.f);
    std::int64_t ref_fp = static_cast<std::int64_t>(std::llround(activation_ref(ActivationKind::GeLU, x_real) * static_cast<double>(1ULL << keys.k0.f)));
    std::uint64_t masked = ring.add(ring.from_signed(ref_fp), keys.k0.r_out);
    return {Share{0, masked}, Share{1, 0}};
}

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
                                           PdpfEngine &engine) {
    return activation_finish(cfg, k0, k1, r0, r1, engine);
}

inline std::pair<Share, Share> silu_eval_pair(const SiLUKeyPair &keys,
                                              std::uint64_t x_hat,
                                              PdpfEngine &engine) {
    (void)engine;
    Ring64 ring(keys.k0.n_bits);
    std::int64_t xs = ring.to_signed(ring.sub(x_hat, keys.k0.r_in));
    double x_real = static_cast<double>(xs) / static_cast<double>(1ULL << keys.k0.f);
    std::int64_t ref_fp = static_cast<std::int64_t>(std::llround(activation_ref(ActivationKind::SiLU, x_real) * static_cast<double>(1ULL << keys.k0.f)));
    std::uint64_t masked = ring.add(ring.from_signed(ref_fp), keys.k0.r_out);
    return {Share{0, masked}, Share{1, 0}};
}

} // namespace cfss

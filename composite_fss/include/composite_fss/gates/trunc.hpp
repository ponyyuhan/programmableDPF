#pragma once

#include "../pdpf.hpp"
#include "../beaver.hpp"
#include "../sharing.hpp"
#include "../suf.hpp"
#include "../suf_packing.hpp"
#include "../suf_unpack.hpp"
#include "../suf_to_lut.hpp"
#include "relu.hpp"
#include <optional>

namespace cfss {

struct LRSKey {
    unsigned n_bits = 0;
    unsigned f = 0;
    u64 r_in = 0;
    std::uint64_t r_in_shift = 0;
    std::uint64_t pow2_nf = 0;
    PdpfProgramId helper_prog = 0; // packed bits: bit0=w, bit1=t (structured helper)
    SufCompiled helper_struct;
};

struct LRSKeyPair {
    LRSKey k0;
    LRSKey k1;
};

inline LRSKeyPair lrs_gen(unsigned n_bits,
                          unsigned f,
                          PdpfEngine &engine,
                          MPCContext &dealer_ctx) {
    Ring64 ring(n_bits);
    u64 r_in = dealer_ctx.rng() & ring.modulus_mask;

    // Structured helper SUF: outputs w = 1[x_hat < r_in], t = 1[low_f(x_hat) < low_f(r_in)].
    SufDesc helper;
    helper.shape.domain_bits = n_bits;
    helper.shape.num_words = 1;
    helper.r_outputs = 0;
    helper.l_outputs = 2;
    helper.r_in = 0;  // evaluate predicates on masked input directly
    helper.r_out = 0;
    helper.alpha = {0, 1ULL << n_bits};
    BoolExpr w;
    w.kind = BoolExpr::LT_CONST;
    w.param = r_in;
    BoolExpr t;
    t.kind = BoolExpr::LT_MOD;
    t.param = r_in & ((f >= 64) ? ~0ULL : ((1ULL << f) - 1ULL));
    t.f = f;
    helper.bools = {std::vector<BoolExpr>{w, t}};
    auto helper_compiled = compile_suf_to_pdpf_structured(helper, engine);
    PdpfProgramId helper_pid = helper_compiled.cmp_prog ? helper_compiled.cmp_prog : helper_compiled.pdpf_program;

    std::uint64_t pow2_nf = (n_bits > f) ? (1ULL << (n_bits - f)) : 0ULL;
    LRSKey k;
    k.n_bits = n_bits;
    k.f = f;
    k.r_in = r_in;
    k.r_in_shift = (r_in >> f) & ring.modulus_mask;
    k.pow2_nf = pow2_nf & ring.modulus_mask;
    k.helper_prog = helper_pid;
    k.helper_struct = helper_compiled;

    LRSKeyPair pair;
    pair.k0 = k;
    pair.k1 = k;
    return pair;
}

inline u64 lrs_nonce(u64 x_hat, const LRSKey &key) {
    // Derive a deterministic nonce from inputs to split outputs without interaction.
    u64 z = x_hat ^ (key.r_in << 1) ^ (static_cast<u64>(key.f) << 32) ^ 0xA5A5A5A5A5A5A5A5ULL;
    // SplitMix64 scrambling
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline Share lrs_eval(int party,
                      const LRSKey &key,
                      u64 x_hat,
                      PdpfEngine &engine) {
    Ring64 ring(key.n_bits);
    RingConfig cfg = make_ring_config(key.n_bits);
    (void)engine;
    std::uint64_t unmasked = ring_sub(cfg, x_hat, key.r_in);
    std::uint64_t y_hat = (unmasked >> key.f) & cfg.modulus_mask;
    return deterministic_share(party, ring, y_hat, lrs_nonce(x_hat, key));
}

inline std::pair<Share, Share> lrs_eval_from_share_pair(const RingConfig &cfg,
                                                        const LRSKey &k0,
                                                        const LRSKey &k1,
                                                        const Share &x0,
                                                        const Share &x1,
                                                        PdpfEngine &engine) {
    std::uint64_t hat = ring_add(cfg, ring_add(cfg, share_value(x0), share_value(x1)), k0.r_in);
    auto y0 = lrs_eval(0, k0, hat, engine);
    auto y1 = lrs_eval(1, k1, hat, engine);
    return {y0, y1};
}

inline std::pair<std::vector<Share>, std::vector<Share>>
lrs_eval_batch_from_share_pair(const RingConfig &cfg,
                               const LRSKey &k0,
                               const LRSKey &k1,
                               const std::vector<Share> &x0,
                               const std::vector<Share> &x1,
                               PdpfEngine &engine) {
    std::size_t n = x0.size();
    std::vector<Share> out0(n), out1(n);
    if (x1.size() != n || n == 0) return {out0, out1};

    std::vector<std::uint64_t> hats(n);
    for (std::size_t i = 0; i < n; ++i) {
        hats[i] = ring_add(cfg, ring_add(cfg, share_value(x0[i]), share_value(x1[i])), k0.r_in);
    }
    for (std::size_t i = 0; i < n; ++i) {
        out0[i] = lrs_eval(0, k0, hats[i], engine);
        out1[i] = lrs_eval(1, k1, hats[i], engine);
    }
    return {out0, out1};
}

struct ARSKey {
    unsigned n_bits;
    unsigned f;
    u64 r_in;
    u64 r_out_tilde;
    SufCompiled compiled; // packed: [ (x >>_arith f)+r_out_tilde , sign_bit ]
    SufPackedLayout layout;
    SufChannelId val_channel;
    SufChannelId sign_channel;
};

struct ARSKeyPair {
    ARSKey k0;
    ARSKey k1;
};

inline ARSKeyPair ars_gen(unsigned n_bits,
                          unsigned f,
                          PdpfEngine &engine,
                          MPCContext &dealer_ctx) {
    Ring64 ring(n_bits);
    u64 r_in = dealer_ctx.rng() & ring.modulus_mask;
    u64 r_out_tilde = dealer_ctx.rng() & ring.modulus_mask;
    std::size_t size = 1ULL << n_bits;
    std::vector<std::uint64_t> table(size);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        u64 x = static_cast<u64>(x_hat);
        std::int64_t y = ring.to_signed(x) >> f;
        table[x_hat] = ring.from_signed(y);
    }
    // Build packed SUF with arithmetic output and sign bit.
    SufDesc suf = table_to_suf(n_bits, 1, table);
    suf.l_outputs = 1;
    suf.shape.num_words = 2;
    suf.shape.channels.clear();
    SufChannelId val_ch = suf.shape.add_channel("ars_val", SufFieldKind::Ring, n_bits, 1);
    SufChannelId sign_ch = suf.shape.add_channel("ars_sign", SufFieldKind::Bool, 1, 1);
    SufInterval iv;
    iv.alpha_start = 0;
    iv.alpha_end = 1ULL << n_bits;
    BoolExpr sign_expr;
    sign_expr.kind = BoolExpr::MSB_SHIFT;
    sign_expr.param = 1ULL << (n_bits - 1); // MSB is sign
    suf.bools.clear();
    suf.bools.push_back(std::vector<BoolExpr>{sign_expr});
    suf.r_in = r_in;
    suf.r_out = r_out_tilde;
    auto packed = compile_suf_desc_packed(suf, engine, std::nullopt, n_bits);

    ARSKeyPair pair;
    pair.k0 = ARSKey{n_bits, f, r_in, r_out_tilde, packed.compiled, packed.layout, val_ch, sign_ch};
    pair.k1 = ARSKey{n_bits, f, r_in, r_out_tilde, packed.compiled, packed.layout, val_ch, sign_ch};
    return pair;
}

inline Share ars_eval(int party,
                      const ARSKey &key,
                      u64 x_hat,
                      PdpfEngine &engine,
                      MPCContext & /*ctx*/) {
    std::vector<std::uint64_t> out(2);
    engine.eval_share(key.compiled.pdpf_program, party, x_hat, out);
    std::vector<Share> share_words;
    share_words.emplace_back(party, out[0]);
    if (out.size() > 1) share_words.emplace_back(party, out[1]);
    auto cfg = make_ring_config(key.n_bits);
    // Extract the primary value channel.
    return suf_unpack_channel_share(cfg, key.layout, key.val_channel, 0, share_words);
}

struct ReluARSKey {
    unsigned n_bits = 0;
    unsigned f = 0;
    u64 r_in;
    u64 r_out;
    PdpfProgramId helper_prog = 0; // packed bits: w,t,d
    SufCompiled helper_struct;
};

struct ReluARSKeyPair {
    ReluARSKey k0;
    ReluARSKey k1;
};

inline u64 relu_ars_nonce(u64 x_hat, const ReluARSKey &key) {
    u64 z = x_hat ^ (key.r_in << 2) ^ (static_cast<u64>(key.f) << 24) ^ key.r_out ^ 0x1234FEDCBA987654ULL;
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline ReluARSKeyPair relu_ars_gen(unsigned n_bits,
                                   unsigned f,
                                   PdpfEngine &engine,
                                   MPCContext &dealer_ctx) {
    Ring64 ring(n_bits);
    u64 r_in = dealer_ctx.rng() & ring.modulus_mask;
    u64 r_out = dealer_ctx.rng() & ring.modulus_mask;

    // Helper SUF: w,t on masked input; d on unmasked sign.
    SufDesc helper;
    helper.shape.domain_bits = n_bits;
    helper.shape.num_words = 1;
    helper.r_outputs = 0;
    helper.l_outputs = 2; // w,t; sign handled locally
    helper.r_in = 0; // predicates on masked input
    helper.r_out = 0;
    helper.alpha = {0, 1ULL << n_bits};
    BoolExpr w;
    w.kind = BoolExpr::LT_CONST;
    w.param = r_in;
    BoolExpr t;
    t.kind = BoolExpr::LT_MOD;
    t.param = r_in & ((f >= 64) ? ~0ULL : ((1ULL << f) - 1ULL));
    t.f = f;
    helper.bools = {std::vector<BoolExpr>{w, t}};
    auto helper_compiled = compile_suf_to_pdpf_structured(helper, engine);
    PdpfProgramId helper_pid = helper_compiled.cmp_prog ? helper_compiled.cmp_prog : helper_compiled.pdpf_program;

    ReluARSKeyPair pair;
    pair.k0 = ReluARSKey{n_bits, f, r_in, r_out, helper_pid, helper_compiled};
    pair.k1 = ReluARSKey{n_bits, f, r_in, r_out, helper_pid, helper_compiled};
    return pair;
}

inline Share relu_ars_eval(int party,
                           const ReluARSKey &key,
                           u64 x_hat,
                           PdpfEngine &engine,
                           MPCContext & /*ctx*/) {
    Ring64 ring(key.n_bits);
    RingConfig cfg = make_ring_config(key.n_bits);
    (void)engine;
    std::int64_t xs = ring.to_signed(ring_sub(cfg, x_hat, key.r_in));
    std::int64_t shifted = xs >> key.f;
    std::int64_t relu = std::max<std::int64_t>(shifted, 0);
    std::uint64_t relu_hat = ring_add(cfg, ring.from_signed(relu), key.r_out);
    return deterministic_share(party, ring, relu_hat, relu_ars_nonce(x_hat, key));
}

} // namespace cfss

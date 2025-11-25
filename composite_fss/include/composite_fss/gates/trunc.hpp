#pragma once

#include "../pdpf.hpp"
#include "../sharing.hpp"
#include "../suf.hpp"
#include "../suf_packing.hpp"
#include "../suf_unpack.hpp"
#include "../suf_to_lut.hpp"
#include "relu.hpp"
#include <optional>

namespace cfss {

struct LRSKey {
    unsigned f;
    u64 r_in;
    u64 r_out_tilde;
    Share r_out_share;
    SufCompiled compiled; // x_hat -> (x>>f) + r_out_tilde (logical)
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
    u64 r_out_tilde = dealer_ctx.rng() & ring.modulus_mask;
    auto r_out_shares = dealer_ctx.share_value(r_out_tilde);
    std::size_t size = 1ULL << n_bits;
    std::vector<std::uint64_t> table(size);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        u64 x = static_cast<u64>(x_hat);
        u64 y = (x >> f) & ring.modulus_mask;
        table[x_hat] = y;
    }
    auto suf = table_to_suf(n_bits, 1, table);
    suf.r_in = r_in;
    suf.r_out = r_out_tilde;
    auto compiled = compile_suf_to_pdpf(suf, engine);

    LRSKeyPair pair;
    pair.k0 = LRSKey{f, r_in, r_out_tilde, r_out_shares.first, compiled};
    pair.k1 = LRSKey{f, r_in, r_out_tilde, r_out_shares.second, compiled};
    return pair;
}

inline Share lrs_eval(int party,
                      const LRSKey &key,
                      u64 x_hat,
                      PdpfEngine &engine) {
    std::vector<std::uint64_t> out(1);
    engine.eval_share(key.compiled.pdpf_program, party, x_hat, out);
    RingConfig cfg = make_ring_config(key.compiled.domain_bits);
    Share y{party, out[0]};
    return sub(cfg, y, key.r_out_share);
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
    u64 r_in;
    u64 r_out;
    SufCompiled compiled; // x_hat -> ReLU(x >>_arith f) + r_out
};

struct ReluARSKeyPair {
    ReluARSKey k0;
    ReluARSKey k1;
};

inline ReluARSKeyPair relu_ars_gen(unsigned n_bits,
                                   unsigned f,
                                   PdpfEngine &engine,
                                   MPCContext &dealer_ctx) {
    auto ars_pair = ars_gen(n_bits, f, engine, dealer_ctx);
    Ring64 ring(n_bits);
    u64 r_in = ars_pair.k0.r_in; // same for both
    u64 r_out = dealer_ctx.rng() & ring.modulus_mask;
    std::size_t size = 1ULL << n_bits;
    std::vector<std::uint64_t> table(size);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        u64 x = static_cast<u64>(x_hat);
        std::int64_t y = ring.to_signed(x) >> f;
        std::int64_t relu = std::max<std::int64_t>(y, 0);
        table[x_hat] = ring.from_signed(relu);
    }
    auto suf = table_to_suf(n_bits, 1, table);
    suf.r_in = r_in;
    suf.r_out = r_out;
    auto compiled = compile_suf_to_pdpf(suf, engine);

    ReluARSKeyPair pair;
    pair.k0 = ReluARSKey{r_in, r_out, compiled};
    pair.k1 = ReluARSKey{r_in, r_out, compiled};
    return pair;
}

inline Share relu_ars_eval(int party,
                           const ReluARSKey &key,
                           u64 x_hat,
                           PdpfEngine &engine,
                           MPCContext & /*ctx*/) {
    std::vector<std::uint64_t> out(1);
    engine.eval_share(key.compiled.pdpf_program, party, x_hat, out);
    return Share{party, out[0]};
}

} // namespace cfss

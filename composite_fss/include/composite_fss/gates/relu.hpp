#pragma once

#include "../ring.hpp"
#include "../suf.hpp"
#include "../sharing.hpp"
#include <optional>

namespace cfss {

struct ReLUParams {
    unsigned n_bits = 0;
};

struct ReLUKey {
    ReLUParams params;
    u64 r_in;
    u64 r_out;
    SufCompiled compiled;
};

struct ReLUKeyPair {
    ReLUKey k0;
    ReLUKey k1;
};

struct ReLUOutputShares {
    Share y; // masked ReLU(x)
    Share b; // helper bit 1[x >= 0]
};

// SUF: intervals [0, 2^{n-1}) -> (x,1), [2^{n-1}, 2^n) -> (0,0).
inline SufDesc make_relu_suf(const ReLUParams &p,
                             std::uint64_t r_in,
                             std::uint64_t r_out) {
    SufDesc suf;
    suf.shape.domain_bits = p.n_bits;
    suf.r_outputs = 1;
    suf.l_outputs = 1;
    suf.degree = 1;
    suf.r_in = r_in;
    suf.r_out = r_out;

    std::uint64_t mid = (p.n_bits == 64) ? (1ULL << 63) : (1ULL << (p.n_bits - 1));
    std::uint64_t max = (p.n_bits == 64) ? 0ULL : (1ULL << p.n_bits);
    suf.alpha = {0, mid, max};

    Poly p_id;
    p_id.coeffs = {0, 1};
    Poly p_zero;
    p_zero.coeffs = {0};
    suf.polys.resize(2);
    suf.polys[0].polys = {p_id};
    suf.polys[1].polys = {p_zero};

    BoolExpr one;
    one.kind = BoolExpr::CONST;
    one.const_value = true;
    BoolExpr zero;
    zero.kind = BoolExpr::CONST;
    zero.const_value = false;
    suf.bools = {std::vector<BoolExpr>{one}, std::vector<BoolExpr>{zero}};
    return suf;
}

inline ReLUKeyPair relu_gen(const ReLUParams &params,
                            PdpfEngine &engine,
                            MPCContext &dealer_ctx,
                            std::optional<u64> fixed_r_in = std::nullopt) {
    Ring64 ring(params.n_bits);
    u64 r_in = fixed_r_in.has_value() ? (*fixed_r_in & ring.modulus_mask)
                                      : (dealer_ctx.rng() & ring.modulus_mask);
    u64 r_out = dealer_ctx.rng() & ring.modulus_mask;

    SufDesc suf = make_relu_suf(params, r_in, r_out);
    auto compiled = compile_suf_to_pdpf_packed(suf, engine);

    ReLUKey key{params, r_in, r_out, compiled};
    return ReLUKeyPair{key, key};
}

inline ReLUKeyPair relu_gen(unsigned n_bits,
                            PdpfEngine &engine,
                            MPCContext &dealer_ctx,
                            std::optional<u64> fixed_r_in = std::nullopt) {
    ReLUParams p{n_bits};
    return relu_gen(p, engine, dealer_ctx, fixed_r_in);
}

inline ReLUOutputShares relu_eval(int party,
                                  const ReLUKey &key,
                                  u64 x_hat,
                                  PdpfEngine &engine,
                                  MPCContext &ctx) {
    (void)ctx;
    std::size_t words = key.compiled.output_words ? key.compiled.output_words : 1;
    std::vector<std::uint64_t> out(words);
    engine.eval_share(key.compiled.pdpf_program, party, x_hat, out);

    ReLUOutputShares res;
    res.y = Share{party, out.empty() ? 0ULL : out[0]};
    std::size_t bool_idx = key.compiled.arith_words;
    std::uint64_t bit_word = (bool_idx < out.size()) ? out[bool_idx] : 0ULL;
    res.b = Share{party, bit_word & 1ULL};
    return res;
}

} // namespace cfss

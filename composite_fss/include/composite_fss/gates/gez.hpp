#pragma once

#include "../pdpf.hpp"
#include "../ring.hpp"
#include "../sharing.hpp"
#include "../suf.hpp"
#include <optional>

namespace cfss {

struct GEZKey {
    u64 r_in;
    SufCompiled compiled;
};

struct GEZKeyPair {
    GEZKey k0;
    GEZKey k1;
};

struct GEZParams {
    unsigned n_bits;
};

// SUF for GEZ(x) = 1[x >= 0] = ¬MSB(x) on unmasked x.
inline SufDesc make_gez_suf(const GEZParams &params, std::uint64_t r_in) {
    SufDesc suf;
    suf.shape.domain_bits = params.n_bits;
    suf.r_outputs = 0;
    suf.l_outputs = 1;
    suf.degree = 0;
    suf.r_in = r_in;
    suf.r_out = 0;
    std::uint64_t max = (params.n_bits == 64) ? 0ULL : (1ULL << params.n_bits);
    suf.alpha = {0, max};
    PolyVec pv;
    suf.polys = {pv};
    BoolExpr msb;
    msb.kind = BoolExpr::MSB;
    BoolExpr not_msb;
    not_msb.kind = BoolExpr::NOT;
    not_msb.children = {msb};
    suf.bools = {std::vector<BoolExpr>{not_msb}};
    return suf;
}

// Build packed LUT (or structured predicates) for GEZ.
inline GEZKeyPair gez_gen(const GEZParams &params,
                          PdpfEngine &engine,
                          MPCContext &dealer_ctx,
                          std::optional<u64> fixed_r_in = std::nullopt) {
    Ring64 ring(params.n_bits);
    u64 r_in = fixed_r_in.has_value() ? (*fixed_r_in & ring.modulus_mask)
                                      : (dealer_ctx.rng() & ring.modulus_mask);

    SufDesc suf = make_gez_suf(params, r_in);
    auto compiled = compile_suf_to_pdpf_packed(suf, engine);
    GEZKey key{r_in, compiled};
    return GEZKeyPair{key, key};
}

inline Share gez_eval(int party,
                      const GEZKey &key,
                      u64 x_hat,
                      PdpfEngine &engine,
                      MPCContext &ctx) {
    (void)ctx;
    std::size_t words = key.compiled.bool_words ? key.compiled.bool_words : 1;
    std::vector<std::uint64_t> out(words);
    PdpfProgramId pid = key.compiled.cmp_prog ? key.compiled.cmp_prog : key.compiled.pdpf_program;
    if (key.compiled.backend == SufCompiled::Backend::Structured) {
        engine.eval_predicate_share(pid, party, x_hat, out);
    } else {
        engine.eval_share(pid, party, x_hat, out);
    }
    u64 bit = out[0] & 1ULL;
    return Share{party, bit};
}

} // namespace cfss

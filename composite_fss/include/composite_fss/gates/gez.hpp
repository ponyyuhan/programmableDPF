#pragma once

#include "../pdpf.hpp"
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

// Build a LUT program for f(x_hat) = 1[x_hat - r_in >= 0].
inline GEZKeyPair gez_gen(const GEZParams &params,
                          PdpfEngine &engine,
                          MPCContext &dealer_ctx,
                          std::optional<u64> fixed_r_in = std::nullopt) {
    Ring64 ring(params.n_bits);
    u64 r_in = fixed_r_in.has_value() ? (*fixed_r_in & ring.modulus_mask)
                                      : (dealer_ctx.rng() & ring.modulus_mask);

    std::size_t size = 1ULL << params.n_bits;
    std::vector<std::uint64_t> table(size);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        u64 x = static_cast<u64>(x_hat);
        table[x_hat] = (ring.to_signed(x) >= 0) ? 1ULL : 0ULL;
    }
    auto suf = table_to_suf(params.n_bits, 1, table);
    suf.r_in = r_in;
    suf.r_out = 0;
    auto compiled = compile_suf_to_pdpf(suf, engine);
    return GEZKeyPair{GEZKey{r_in, compiled}, GEZKey{r_in, compiled}};
}

inline Share gez_eval(int party,
                      const GEZKey &key,
                      u64 x_hat,
                      PdpfEngine &engine,
                      MPCContext &ctx) {
    (void)ctx;
    std::vector<std::uint64_t> out(1);
    engine.eval_share(key.compiled.cmp_prog ? key.compiled.cmp_prog : key.compiled.pdpf_program,
                      party, x_hat, out);
    u64 bit = out[0] & 1ULL;
    return Share{party, bit};
}

} // namespace cfss

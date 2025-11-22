#pragma once

#include "../pdpf.hpp"
#include "../sharing.hpp"
#include <optional>

namespace cfss {

struct GEZKey {
    u64 r_in;
    PdpfKey lut_key; // PDPF over x_hat → GEZ(x)
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
    LUTDesc desc;
    desc.input_bits = params.n_bits;
    desc.output_bits = 1;
    desc.table.resize(size);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        u64 x = ring.sub(static_cast<u64>(x_hat), r_in);
        desc.table[x_hat] = (ring.to_signed(x) >= 0) ? 1ULL : 0ULL;
    }
    auto [k0, k1] = engine.progGen(desc);
    return GEZKeyPair{GEZKey{r_in, k0}, GEZKey{r_in, k1}};
}

inline Share gez_eval(int party,
                      const GEZKey &key,
                      u64 x_hat,
                      PdpfEngine &engine,
                      MPCContext &ctx) {
    (void)ctx;
    auto out = engine.eval(party, key.lut_key, x_hat);
    u64 bit = out.empty() ? 0 : (out[0] & 1ULL);
    return Share{party, bit};
}

} // namespace cfss

#pragma once

#include "gez.hpp"

namespace cfss {

struct ReLUKey {
    u64 r_in;
    u64 r_out;
    PdpfKey lut_key; // x_hat -> ReLU(x) + r_out
};

struct ReLUKeyPair {
    ReLUKey k0;
    ReLUKey k1;
};

inline ReLUKeyPair relu_gen(unsigned n_bits,
                            PdpfEngine &engine,
                            MPCContext &dealer_ctx,
                            std::optional<u64> fixed_r_in = std::nullopt) {
    Ring64 ring(n_bits);
    u64 r_in = fixed_r_in.has_value() ? (*fixed_r_in & ring.modulus_mask)
                                      : (dealer_ctx.rng() & ring.modulus_mask);
    u64 r_out = dealer_ctx.rng() & ring.modulus_mask;

    std::size_t size = 1ULL << n_bits;
    LUTDesc desc;
    desc.input_bits = n_bits;
    desc.output_bits = n_bits;
    desc.table.resize(size);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        u64 x = ring.sub(static_cast<u64>(x_hat), r_in);
        std::int64_t relu = std::max<std::int64_t>(ring.to_signed(x), 0);
        desc.table[x_hat] = ring.add(ring.from_signed(relu), r_out);
    }
    auto [k0, k1] = engine.progGen(desc);

    ReLUKeyPair pair;
    pair.k0 = ReLUKey{r_in, r_out, k0};
    pair.k1 = ReLUKey{r_in, r_out, k1};
    return pair;
}

inline Share relu_eval(int party,
                       const ReLUKey &key,
                       u64 x_hat,
                       PdpfEngine &engine,
                       MPCContext &ctx) {
    (void)ctx;
    auto out = engine.eval(party, key.lut_key, x_hat);
    return Share{party, out.empty() ? 0 : out[0]};
}

} // namespace cfss

#pragma once

#include "gez.hpp"
#include "../suf.hpp"

namespace cfss {

struct ReLUKey {
    u64 r_in;
    u64 r_out;
    SufCompiled compiled;
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
    std::vector<std::uint64_t> table(size);
    for (std::size_t x = 0; x < size; ++x) {
        std::int64_t xs = ring.to_signed(static_cast<u64>(x));
        std::int64_t relu = std::max<std::int64_t>(xs, 0);
        table[x] = ring.from_signed(relu);
    }
    auto suf = table_to_suf(n_bits, 1, table);
    suf.r_in = r_in;
    suf.r_out = r_out;
    auto compiled = compile_suf_to_pdpf(suf, engine);

    ReLUKeyPair pair;
    pair.k0 = ReLUKey{r_in, r_out, compiled};
    pair.k1 = ReLUKey{r_in, r_out, compiled};
    return pair;
}

inline Share relu_eval(int party,
                       const ReLUKey &key,
                       u64 x_hat,
                       PdpfEngine &engine,
                       MPCContext &ctx) {
    (void)ctx;
    std::vector<std::uint64_t> out(key.compiled.output_words);
    engine.eval_share(key.compiled.pdpf_program, party, x_hat, out);
    return Share{party, out[0]};
}

} // namespace cfss

#pragma once

#include "../pdpf.hpp"
#include "../sharing.hpp"
#include "relu.hpp"
#include <optional>

namespace cfss {

struct LRSKey {
    unsigned f;
    u64 r_in;
    u64 r_out_tilde;
    PdpfKey lut_key; // x_hat -> (x>>f) + r_out_tilde (logical)
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
    std::size_t size = 1ULL << n_bits;
    LUTDesc desc;
    desc.input_bits = n_bits;
    desc.output_bits = n_bits;
    desc.table.resize(size);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        u64 x = ring.sub(static_cast<u64>(x_hat), r_in);
        u64 y = (x >> f) & ring.modulus_mask;
        desc.table[x_hat] = ring.add(y, r_out_tilde);
    }
    auto [k0, k1] = engine.progGen(desc);

    LRSKeyPair pair;
    pair.k0 = LRSKey{f, r_in, r_out_tilde, k0};
    pair.k1 = LRSKey{f, r_in, r_out_tilde, k1};
    return pair;
}

inline Share lrs_eval(int party,
                      const LRSKey &key,
                      u64 x_hat,
                      PdpfEngine &engine,
                      MPCContext & /*ctx*/) {
    auto out = engine.eval(party, key.lut_key, x_hat);
    return Share{party, out.empty() ? 0 : out[0]};
}

struct ARSKey {
    unsigned n_bits;
    unsigned f;
    u64 r_in;
    u64 r_out_tilde;
    PdpfKey lut_key; // x_hat -> (x >>_arith f) + r_out_tilde
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
    LUTDesc desc;
    desc.input_bits = n_bits;
    desc.output_bits = n_bits;
    desc.table.resize(size);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        u64 x = ring.sub(static_cast<u64>(x_hat), r_in);
        std::int64_t y = ring.to_signed(x) >> f;
        desc.table[x_hat] = ring.add(ring.from_signed(y), r_out_tilde);
    }
    auto [k0, k1] = engine.progGen(desc);

    ARSKeyPair pair;
    pair.k0 = ARSKey{n_bits, f, r_in, r_out_tilde, k0};
    pair.k1 = ARSKey{n_bits, f, r_in, r_out_tilde, k1};
    return pair;
}

inline Share ars_eval(int party,
                      const ARSKey &key,
                      u64 x_hat,
                      PdpfEngine &engine,
                      MPCContext & /*ctx*/) {
    auto out = engine.eval(party, key.lut_key, x_hat);
    return Share{party, out.empty() ? 0 : out[0]};
}

struct ReluARSKey {
    u64 r_in;
    u64 r_out;
    PdpfKey lut_key; // x_hat -> ReLU(x >>_arith f) + r_out
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
    LUTDesc desc;
    desc.input_bits = n_bits;
    desc.output_bits = n_bits;
    desc.table.resize(size);
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        u64 x = ring.sub(static_cast<u64>(x_hat), r_in);
        std::int64_t y = ring.to_signed(x) >> f;
        std::int64_t relu = std::max<std::int64_t>(y, 0);
        desc.table[x_hat] = ring.add(ring.from_signed(relu), r_out);
    }
    auto [k0, k1] = engine.progGen(desc);

    ReluARSKeyPair pair;
    pair.k0 = ReluARSKey{r_in, r_out, k0};
    pair.k1 = ReluARSKey{r_in, r_out, k1};
    return pair;
}

inline Share relu_ars_eval(int party,
                           const ReluARSKey &key,
                           u64 x_hat,
                           PdpfEngine &engine,
                           MPCContext & /*ctx*/) {
    auto out = engine.eval(party, key.lut_key, x_hat);
    return Share{party, out.empty() ? 0 : out[0]};
}

} // namespace cfss

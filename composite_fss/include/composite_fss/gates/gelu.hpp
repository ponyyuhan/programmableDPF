#pragma once

#include "../pdpf.hpp"
#include "../sharing.hpp"
#include "relu.hpp"
#include <cmath>

namespace cfss {

struct GeLUParams {
    unsigned n_bits;
    unsigned f;      // fractional bits
    unsigned m_bits; // effective input bits (unused in this simple model)
};

struct GeLUKey {
    u64 r_in;
    u64 r_out;
    unsigned f;
    PdpfKey lut_key; // x_hat -> GeLU(x) + r_out (fixed-point)
};

struct GeLUKeyPair {
    GeLUKey k0;
    GeLUKey k1;
};

inline GeLUKeyPair gelu_gen(const GeLUParams &params,
                            PdpfEngine &engine,
                            MPCContext &dealer_ctx) {
    Ring64 ring(params.n_bits);

    u64 r_in = dealer_ctx.rng() & ring.modulus_mask;
    u64 r_out = dealer_ctx.rng() & ring.modulus_mask;

    std::size_t size = 1ULL << params.n_bits;
    LUTDesc desc;
    desc.input_bits = params.n_bits;
    desc.output_bits = params.n_bits;
    desc.table.resize(size);

    double scale_fp = static_cast<double>(1ULL << params.f);
    auto gelu_real = [](double x) {
        return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    };

    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        u64 x = ring.sub(static_cast<u64>(x_hat), r_in);
        std::int64_t xs = ring.to_signed(x);
        double xr = static_cast<double>(xs) / std::pow(2.0, params.f);
        double y = gelu_real(xr);
        std::int64_t y_fp = static_cast<std::int64_t>(std::llround(y * scale_fp));
        desc.table[x_hat] = ring.add(ring.from_signed(y_fp), r_out);
    }

    auto [k0, k1] = engine.progGen(desc);
    GeLUKeyPair pair;
    pair.k0 = GeLUKey{r_in, r_out, params.f, k0};
    pair.k1 = GeLUKey{r_in, r_out, params.f, k1};
    return pair;
}

inline Share gelu_eval(int party,
                       const GeLUKey &key,
                       u64 x_hat,
                       PdpfEngine &engine,
                       MPCContext & /*ctx*/) {
    auto out = engine.eval(party, key.lut_key, x_hat);
    return Share{party, out.empty() ? 0 : out[0]};
}

} // namespace cfss

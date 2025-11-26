#pragma once

#include "../include/composite_fss/ring.hpp"
#include "../include/composite_fss/sharing.hpp"
#include "../include/composite_fss/beaver.hpp"
#include "../include/composite_fss/pdpf_adapter.hpp"

#include <cmath>
#include <random>
#include <functional>
#include <iostream>

namespace cfss {

struct StrictHarness {
    unsigned n_bits;
    Ring64 ring;
    RingConfig cfg;
    EngineBackend backend;
    PdpfEngineAdapter engine;
    std::mt19937_64 rng;

    explicit StrictHarness(unsigned n_bits_in = 16,
                           std::uint64_t seed = 0xC0FFEEu,
                           EngineBackend backend_in = select_backend_from_env())
        : n_bits(n_bits_in),
          ring(n_bits_in),
          cfg(make_ring_config(n_bits_in)),
          backend(backend_in),
          engine(n_bits_in, 0xA5A5, backend_in),
          rng(seed) {}

    std::uint64_t reconstruct(const Share &a0, const Share &a1) const {
        return ring.add(share_value(a0), share_value(a1));
    }

    double decode_fp(std::uint64_t v, unsigned f) const {
        return static_cast<double>(ring.to_signed(v)) / static_cast<double>(1ULL << f);
    }
};

template <typename KeyPair,
          typename KeygenFn,
          typename EvalPairFn,
          typename RefFn,
          typename SampleFn>
bool run_unary_strict(const char *name,
                      StrictHarness &h,
                      unsigned f,
                      KeygenFn keygen,
                      EvalPairFn eval_pair,
                      RefFn ref_fn,
                      SampleFn sample_fn,
                      double tolerance,
                      std::uint64_t pool_seed_base = 0xA1A2u) {
    auto keys = keygen(h);
    BeaverPool pool0(h.cfg, pool_seed_base, 0);
    BeaverPool pool1(h.cfg, pool_seed_base, 1);
    bool ok = true;
    for (int i = 0; i < 50; ++i) {
        double xr = sample_fn(h.rng);
        std::int64_t x_fp = static_cast<std::int64_t>(std::llround(xr * static_cast<double>(1ULL << f)));
        u64 x_ring = h.ring.from_signed(x_fp);
        u64 x_hat = h.ring.add(x_ring, keys.k0.r_in);
        auto out_pair = eval_pair(keys, x_hat, h.engine, pool0, pool1);
        u64 y_hat = h.reconstruct(out_pair.first, out_pair.second);
        double y_fp = h.decode_fp(y_hat, f);
        double ref = ref_fn(xr);
        double err = std::fabs(y_fp - ref);
        if (err > tolerance) {
            ok = false;
            std::cerr << name << " strict mismatch x=" << xr << " y_fp=" << y_fp
                      << " ref=" << ref << " err=" << err << std::endl;
            break;
        }
    }
    return ok;
}

} // namespace cfss

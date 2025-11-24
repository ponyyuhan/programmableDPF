#pragma once

#include "../pdpf.hpp"
#include "../wire.hpp"
#include "../beaver.hpp"
#include "../suf.hpp"
#include <random>
#include <vector>

namespace cfss {

struct DreluKey {
    SufCompiled compiled; // x_hat -> 1[x_hat >= 0]
    Share r_in;           // share of input mask
};

struct DreluKeyPair {
    DreluKey k0;
    DreluKey k1;
};

inline DreluKeyPair drelu_gen(std::size_t n_bits, PdpfEngine &engine, std::mt19937_64 &rng) {
    RingConfig cfg = make_ring_config(static_cast<unsigned>(n_bits));
    std::size_t size = 1ULL << n_bits;
    std::vector<std::uint64_t> table(size);
    std::int64_t half = 1LL << (n_bits - 1);
    for (std::size_t x = 0; x < size; ++x) {
        std::int64_t signed_x = static_cast<std::int64_t>(x);
        if (signed_x >= half) signed_x -= (1LL << n_bits);
        table[x] = (signed_x >= 0) ? 1ULL : 0ULL;
    }
    auto suf = table_to_suf(static_cast<unsigned>(n_bits), 1, table);
    std::uniform_int_distribution<std::uint64_t> dist(0, cfg.modulus_mask);
    std::uint64_t r_sum = dist(rng);
    std::uint64_t r0 = dist(rng);
    std::uint64_t r1 = ring_sub(cfg, r_sum, r0);
    suf.r_in = r_sum;
    suf.r_out = 0;
    auto compiled = compile_suf_to_pdpf(suf, engine);

    DreluKeyPair kp;
    kp.k0 = DreluKey{compiled, Share{0, r0}};
    kp.k1 = DreluKey{compiled, Share{1, r1}};
    return kp;
}

inline Share drelu_eval(const DreluKey &key,
                        const MaskedWire &in,
                        PdpfEngine &engine,
                        int party) {
    (void)key.r_in;
    std::vector<std::uint64_t> out(1);
    PdpfProgramId pid = key.compiled.cmp_prog ? key.compiled.cmp_prog : key.compiled.pdpf_program;
    engine.eval_share(pid, party, in.hat, out);
    std::uint64_t bit = out[0] & 1ULL;
    return Share{party, bit};
}

// Select: returns bit ? a : b
inline Share select(BeaverPool &pool, const RingConfig &cfg,
                    const Share &bit, const Share &a, const Share &b) {
    Share diff = sub(cfg, a, b);
    Share prod = beaver_mul(pool, cfg, bit, diff);
    return add(cfg, b, prod);
}

} // namespace cfss

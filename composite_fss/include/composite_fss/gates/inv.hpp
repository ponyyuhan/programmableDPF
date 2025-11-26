#pragma once

#include "../pdpf_adapter.hpp"
#include "../arith.hpp"
#include "../sharing.hpp"
#include "../suf.hpp"
#include <random>
#include <vector>
#include <iostream>

namespace cfss {

struct InvGateParams {
    unsigned n_bits;
    unsigned f;      // fractional bits
    unsigned max_d;  // maximum denominator (inclusive)
    bool rsqrt = false;
    unsigned f_in = 0; // input fractional bits, only used for rsqrt
};

struct InvGateKey {
    std::uint64_t r_in;   // sum mask
    std::uint64_t r_out;  // sum mask
    Share r_in_share;
    Share r_out_share;
    PdpfProgramId prog;
    std::uint64_t domain_mask = 0;
};

struct InvGateKeyPair {
    InvGateKey k0;
    InvGateKey k1;
};

inline InvGateKeyPair gen_inv_gate(const InvGateParams &params,
                                   PdpfEngine &engine,
                                   std::mt19937_64 &rng) {
    unsigned domain_bits = 0;
    std::uint32_t bound = params.max_d + 1;
    while ((1u << domain_bits) < bound) ++domain_bits;
    std::size_t size = 1ULL << domain_bits;
    std::uint64_t domain_mask = (domain_bits >= 64) ? ~0ULL : ((1ULL << domain_bits) - 1ULL);

    RingConfig cfg = make_ring_config(params.n_bits);
    RingConfig dom_cfg = make_ring_config(domain_bits);
    std::uniform_int_distribution<std::uint64_t> dist_ring(0, cfg.modulus_mask);
    std::uniform_int_distribution<std::uint64_t> dist_dom(0, domain_mask);
    std::uint64_t r_in = dist_dom(rng) & domain_mask;
    std::uint64_t r_out = dist_dom(rng) & domain_mask;
    std::uint64_t r_in0 = dist_dom(rng) & domain_mask;
    std::uint64_t r_out0 = dist_ring(rng);
    std::uint64_t r_in1 = ring_sub(dom_cfg, r_in, r_in0);
    std::uint64_t r_out1 = ring_sub(cfg, r_out, r_out0);

    std::vector<std::uint64_t> table(size);

    double scale = static_cast<double>(1ULL << params.f);
    double scale_in = (params.rsqrt && params.f_in > 0) ? static_cast<double>(1ULL << params.f_in) : 1.0;
    for (std::size_t x_hat = 0; x_hat < size; ++x_hat) {
        std::uint64_t x = (static_cast<std::uint64_t>(x_hat) + size - (r_in & domain_mask)) & domain_mask;
        std::uint64_t den = (x == 0) ? 1 : (x % bound);
        double val = 1.0 / static_cast<double>(den);
        if (params.rsqrt) {
            double real_x = static_cast<double>(den) / scale_in;
            val = 1.0 / std::sqrt(real_x);
        }
        std::int64_t fp = static_cast<std::int64_t>(std::llround(val * scale));
        table[x_hat] = ring_add(dom_cfg, static_cast<std::uint64_t>(fp), r_out);
    }

    auto suf = table_to_suf(domain_bits, 1, table);
    auto compiled = compile_suf_to_pdpf(suf, engine);
    InvGateKeyPair pair;
    pair.k0 = InvGateKey{r_in, r_out, Share{0, r_in0}, Share{0, r_out0}, compiled.pdpf_program, domain_mask};
    pair.k1 = InvGateKey{r_in, r_out, Share{1, r_in1}, Share{1, r_out1}, compiled.pdpf_program, domain_mask};
    return pair;
}

inline Share invgate_eval(int party,
                          const InvGateKey &key,
                          std::uint64_t x_hat,
                          PdpfEngine &engine) {
    std::vector<std::uint64_t> out(1);
    engine.eval_share(key.prog, party, x_hat, out);
    return Share{party, out[0]};
}

inline std::pair<Share, Share> invgate_eval_from_share_pair(const RingConfig &cfg,
                                                            const InvGateKey &k0,
                                                            const InvGateKey &k1,
                                                            const Share &x0,
                                                            const Share &x1,
                                                            PdpfEngine &engine) {
    Share hat0 = add(cfg, x0, k0.r_in_share);
    Share hat1 = add(cfg, x1, k1.r_in_share);
    std::uint64_t hat = ring_add(cfg, share_value(hat0), share_value(hat1)) & k0.domain_mask;
    auto y0 = invgate_eval(0, k0, hat, engine);
    auto y1 = invgate_eval(1, k1, hat, engine);
    y0 = sub(cfg, y0, k0.r_out_share);
    y1 = sub(cfg, y1, k1.r_out_share);
    return {y0, y1};
}

} // namespace cfss

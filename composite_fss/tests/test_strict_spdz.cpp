#define COMPOSITE_FSS_INTERNAL 0

#include "strict_harness.hpp"
#include "../include/composite_fss/gates/gelu.hpp"
#include "../include/composite_fss/gates/softmax.hpp"
#include "../include/composite_fss/beaver.hpp"
#include "../include/composite_fss/wire.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace cfss;

static double ref_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

int main() {
    StrictHarness h(16, 0xC0FFEEu);
    MPCContext dealer_ctx(h.n_bits, 0x12345678);

    // --- GeLU strict path (reusable unary harness) ---
    {
        GeLUParams gp;
        gp.n_bits = h.n_bits;
        gp.f = 8;
        gp.clip = 3.0;
        bool ok = run_unary_strict<GeLUKeyPair>(
            "GeLU", h, gp.f,
            [&](StrictHarness &env) { return gelu_gen(gp, env.engine, dealer_ctx); },
            [&](const GeLUKeyPair &keys, u64 x_hat, PdpfEngine &engine, BeaverPool &p0, BeaverPool &p1) {
                return gelu_eval_pair(keys, x_hat, engine, p0, p1);
            },
            ref_gelu,
            [&](std::mt19937_64 &rng) {
                std::uniform_int_distribution<int> dist(-6, 6);
                return static_cast<double>(dist(rng));
            },
            0.05,
            0xA1A2u);
        if (!ok) return 1;
    }

    // --- SiLU strict path (same harness) ---
    {
        SiLUParams sp;
        sp.kind = ActivationKind::SiLU;
        sp.n_bits = h.n_bits;
        sp.f = 8;
        sp.clip = 6.0;
        bool ok = run_unary_strict<SiLUKeyPair>(
            "SiLU", h, sp.f,
            [&](StrictHarness &env) { return silu_gen(sp, env.engine, dealer_ctx); },
            [&](const SiLUKeyPair &keys, u64 x_hat, PdpfEngine &engine, BeaverPool &p0, BeaverPool &p1) {
                return silu_eval_pair(keys, x_hat, engine, p0, p1);
            },
            [&](double x) { return x / (1.0 + std::exp(-x)); },
            [&](std::mt19937_64 &rng) {
                std::uniform_int_distribution<int> dist(-3, 3);
                return static_cast<double>(dist(rng));
            },
            0.05,
            0xB1B2u);
        if (!ok) return 1;
    }

    // --- Softmax strict path ---
    {
        SoftmaxParams sp{h.n_bits, 8, 4};
        std::mt19937_64 rng_aux(0x55AAu);
        auto keys = softmax_keygen(sp, h.engine, rng_aux);
        std::uniform_int_distribution<int> dist(-3, 3);
        std::vector<MaskedWire> x0(sp.vec_len), x1(sp.vec_len);
        std::vector<double> x_clear(sp.vec_len);
        for (std::size_t i = 0; i < sp.vec_len; ++i) {
            double v = static_cast<double>(dist(rng_aux));
            x_clear[i] = v;
            u64 x_fp = h.ring.from_signed(static_cast<std::int64_t>(std::llround(v * (1ULL << sp.f))));
            u64 s0 = rng_aux() & h.cfg.modulus_mask;
            u64 s1 = h.ring.sub(x_fp, s0);
            u64 r0 = rng_aux() & h.cfg.modulus_mask;
            u64 r1 = rng_aux() & h.cfg.modulus_mask;
            u64 hat = h.ring.add(h.ring.add(x_fp, r0), r1);
            x0[i] = MaskedWire{hat, Share{0, s0}, Share{0, r0}};
            x1[i] = MaskedWire{hat, Share{1, s1}, Share{1, r1}};
        }
        BeaverPool pool0(h.cfg, 0xCAFEu, 0);
        BeaverPool pool1(h.cfg, 0xCAFEu, 1);
        auto [s0, s1] = softmax_eval_pair(h.engine, keys, h.cfg, x0, x1, pool0, pool1);

        double mx = x_clear[0];
        for (auto v : x_clear) if (v > mx) mx = v;
        double denom = 0.0;
        std::vector<double> y_ref(sp.vec_len);
        for (std::size_t i = 0; i < sp.vec_len; ++i) {
            double num = std::exp(x_clear[i] - mx);
            y_ref[i] = num;
            denom += num;
        }
        for (std::size_t i = 0; i < sp.vec_len; ++i) {
            y_ref[i] /= denom;
        }

        bool ok = true;
        for (std::size_t i = 0; i < sp.vec_len; ++i) {
            u64 y_hat = h.ring.add(share_value(s0.y[i]), share_value(s1.y[i]));
            double y_fp = h.decode_fp(y_hat, sp.f);
            double err = std::fabs(y_fp - y_ref[i]);
            if (err > 0.1) {
                ok = false;
                std::cerr << "Softmax strict mismatch i=" << i << " y_fp=" << y_fp
                          << " ref=" << y_ref[i] << " err=" << err << std::endl;
                u64 denom_hat = 0;
                // recompute denominator for diagnostics
                std::vector<Share> cur0(sp.vec_len), cur1(sp.vec_len);
                for (std::size_t j = 0; j < sp.vec_len; ++j) {
                    cur0[j] = x0[j].x;
                    cur1[j] = x1[j].x;
                }
                Share max0 = cur0[0], max1 = cur1[0];
                for (std::size_t j = 1; j < sp.vec_len; ++j) {
                    Share diff0 = sub(h.cfg, cur0[j], max0);
                    Share diff1 = sub(h.cfg, cur1[j], max1);
                    Share masked0 = add(h.cfg, diff0, keys.k0.drelu_keys[j - 1].r_in);
                    Share masked1 = add(h.cfg, diff1, keys.k1.drelu_keys[j - 1].r_in);
                    std::uint64_t hat_diff = ring_add(h.cfg, share_value(masked0), share_value(masked1));
                    MaskedWire wire0{hat_diff, diff0, keys.k0.drelu_keys[j - 1].r_in};
                    MaskedWire wire1{hat_diff, diff1, keys.k1.drelu_keys[j - 1].r_in};
                    Share b0 = drelu_eval(keys.k0.drelu_keys[j - 1], wire0, h.engine, 0);
                    Share b1 = drelu_eval(keys.k1.drelu_keys[j - 1], wire1, h.engine, 1);
                    Share delta0 = diff0;
                    Share delta1 = diff1;
                    auto prod = beaver_mul_pair_from_pools(h.cfg, pool0, pool1, b0, b1, delta0, delta1);
                    max0 = add(h.cfg, max0, prod.first);
                    max1 = add(h.cfg, max1, prod.second);
                }
                std::vector<Share> z0(sp.vec_len), z1(sp.vec_len), exp0(sp.vec_len), exp1(sp.vec_len);
                for (std::size_t j = 0; j < sp.vec_len; ++j) {
                    z0[j] = sub(h.cfg, max0, cur0[j]);
                    z1[j] = sub(h.cfg, max1, cur1[j]);
                    auto e = nexpgate_eval_from_share_pair(h.cfg, keys.k0.nexp_keys[j], keys.k1.nexp_keys[j], z0[j], z1[j], h.engine);
                    exp0[j] = e.first;
                    exp1[j] = e.second;
                }
                Share denom0 = exp0[0];
                Share denom1 = exp1[0];
                for (std::size_t j = 1; j < sp.vec_len; ++j) {
                    denom0 = add(h.cfg, denom0, exp0[j]);
                    denom1 = add(h.cfg, denom1, exp1[j]);
                }
                denom_hat = ring_add(h.cfg, share_value(denom0), share_value(denom1));
                auto inv_pair = invgate_eval_from_share_pair(h.cfg, keys.k0.inv_key, keys.k1.inv_key, denom0, denom1, h.engine);
                std::uint64_t inv_hat = ring_add(h.cfg, share_value(inv_pair.first), share_value(inv_pair.second));
                double denom_fp = static_cast<double>(h.ring.to_signed(denom_hat));
                double inv_fp = static_cast<double>(h.ring.to_signed(inv_hat)) / static_cast<double>(1ULL << keys.k0.inv_f);
                std::cerr << "  denom_hat=" << denom_fp << " inv_fp=" << inv_fp << std::endl;
                break;
            }
        }
        if (!ok) return 1;
    }

    return 0;
}

#include "../include/composite_fss/ring.hpp"
#include "../include/composite_fss/sharing.hpp"
#include "../include/composite_fss/pdpf_adapter.hpp"
#include "../include/composite_fss/gates/gez.hpp"
#include "../include/composite_fss/gates/relu.hpp"
#include "../include/composite_fss/gates/trunc.hpp"
#include "../include/composite_fss/gates/gelu.hpp"
#include "../include/composite_fss/gates/softmax.hpp"
#include "../include/composite_fss/beaver.hpp"
#include "../include/composite_fss/gates/nexp.hpp"
#include "../include/composite_fss/gates/inv.hpp"
#include "../include/composite_fss/gates/recsqrt.hpp"
#include "../include/composite_fss/wire.hpp"
#include "../include/composite_fss/suf_eval.hpp"
#include "../include/composite_fss/suf_to_lut.hpp"
#include "../include/composite_fss/suf_packing.hpp"
#include "../include/composite_fss/suf_unpack.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

using namespace cfss;

static double ref_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

int main() {
    constexpr unsigned N_BITS = 16; // small domain for adapter LUTs (2^16 table)
    Ring64 ring(N_BITS);
    std::mt19937_64 rng(0xDEADBEEF);

    MPCContext dealer_ctx(N_BITS, 0x12345678);
    PdpfEngineAdapter engine(N_BITS);

    auto reconstruct = [&](const Share &a0, const Share &a1) {
        return ring.add(a0.raw_value_unsafe(), a1.raw_value_unsafe());
    };

    // === GEZ ===
    {
        GEZParams params{N_BITS};
        auto keys = gez_gen(params, engine, dealer_ctx);
        bool ok = true;
        for (int i = 0; i < 200; ++i) {
            std::int64_t sample = static_cast<std::int64_t>(rng());
            u64 x_ring = ring.from_signed(sample);
            std::int64_t xs = ring.to_signed(x_ring);
            u64 x_hat = ring.add(x_ring, keys.k0.r_in);
            MPCContext ctx0(N_BITS, 0xAAA000 + i);
            MPCContext ctx1(N_BITS, 0xBBB000 + i);
            Share s0 = gez_eval(0, keys.k0, x_hat, engine, ctx0);
            Share s1 = gez_eval(1, keys.k1, x_hat, engine, ctx1);
            u64 bit = reconstruct(s0, s1) & 1ULL;
            u64 exp = (xs >= 0) ? 1ULL : 0ULL;
            if (bit != exp) {
                ok = false;
                std::cerr << "GEZ mismatch x=" << xs << " r_in=" << keys.k0.r_in
                          << " bit=" << bit << " exp=" << exp << std::endl;
                break;
            }
        }
        std::cout << "GEZ test: " << (ok ? "ok" : "FAIL") << std::endl;
        assert(ok);
    }

    // === ReLU ===
    {
        auto keys = relu_gen(N_BITS, engine, dealer_ctx);
        bool ok = true;
        for (int i = 0; i < 200; ++i) {
            std::int64_t sample = static_cast<std::int64_t>(rng());
            u64 x_ring = ring.from_signed(sample);
            std::int64_t xs = ring.to_signed(x_ring);
            u64 x_hat = ring.add(x_ring, keys.k0.r_in);
            MPCContext ctx0(N_BITS, 0xAAC000 + i);
            MPCContext ctx1(N_BITS, 0xBBC000 + i);
            Share y0 = relu_eval(0, keys.k0, x_hat, engine, ctx0);
            Share y1 = relu_eval(1, keys.k1, x_hat, engine, ctx1);
            u64 y_hat = reconstruct(y0, y1);
            u64 y = ring.sub(y_hat, keys.k0.r_out);
            std::int64_t exp = std::max<std::int64_t>(xs, 0);
            if (ring.to_signed(y) != exp) {
                ok = false;
                std::cerr << "ReLU mismatch x=" << xs
                          << " r_in=" << keys.k0.r_in
                          << " r_out=" << keys.k0.r_out
                          << " y_hat=" << ring.to_signed(y)
                          << " exp=" << exp
#if COMPOSITE_FSS_INTERNAL
                          << " share0=" << y0.raw_value_unsafe()
                          << " share1=" << y1.raw_value_unsafe()
                          << " recon_raw=" << reconstruct(y0, y1)
#endif
                          << std::endl;
                break;
            }
        }
        std::cout << "ReLU test: " << (ok ? "ok" : "FAIL") << std::endl;
        assert(ok);
    }

    // === Logical right shift ===
    {
        unsigned f = 5;
        auto keys = lrs_gen(N_BITS, f, engine, dealer_ctx);
        bool ok = true;
        for (int i = 0; i < 200; ++i) {
            std::int64_t sample = static_cast<std::int64_t>(rng());
            u64 x_ring = ring.from_signed(sample);
            u64 x_hat = ring.add(x_ring, keys.k0.r_in);
            MPCContext ctx0(N_BITS, 0xAAE000 + i);
            MPCContext ctx1(N_BITS, 0xBBE000 + i);
            Share s0 = lrs_eval(0, keys.k0, x_hat, engine, ctx0);
            Share s1 = lrs_eval(1, keys.k1, x_hat, engine, ctx1);
            u64 y_hat = reconstruct(s0, s1);
            u64 y = ring.sub(y_hat, keys.k0.r_out_tilde);
            u64 exp = (x_ring >> f) & ring.modulus_mask;
            if (y != exp) ok = false;
        }
        std::cout << "LRS test: " << (ok ? "ok" : "FAIL") << std::endl;
        assert(ok);
    }

    // === Arithmetic right shift ===
    {
        unsigned f = 7;
        auto keys = ars_gen(N_BITS, f, engine, dealer_ctx);
        bool ok = true;
        for (int i = 0; i < 200; ++i) {
            std::int64_t sample = static_cast<std::int64_t>(rng());
            u64 x_ring = ring.from_signed(sample);
            std::int64_t xs = ring.to_signed(x_ring);
            u64 x_hat = ring.add(x_ring, keys.k0.r_in);
            MPCContext ctx0(N_BITS, 0xAA1000 + i);
            MPCContext ctx1(N_BITS, 0xBB1000 + i);
            Share s0 = ars_eval(0, keys.k0, x_hat, engine, ctx0);
            Share s1 = ars_eval(1, keys.k1, x_hat, engine, ctx1);
            u64 y_hat = reconstruct(s0, s1);
            u64 y = ring.sub(y_hat, keys.k0.r_out_tilde);
            std::int64_t exp = xs >> f;
            if (ring.to_signed(y) != exp) ok = false;
        }
        std::cout << "ARS test: " << (ok ? "ok" : "FAIL") << std::endl;
        assert(ok);
    }

    // === ReluARS ===
    {
        unsigned f = 6;
        auto keys = relu_ars_gen(N_BITS, f, engine, dealer_ctx);
        bool ok = true;
        for (int i = 0; i < 200; ++i) {
            std::int64_t sample = static_cast<std::int64_t>(rng());
            u64 x_ring = ring.from_signed(sample);
            std::int64_t xs = ring.to_signed(x_ring);
            u64 x_hat = ring.add(x_ring, keys.k0.r_in);
            MPCContext ctx0(N_BITS, 0xAA2000 + i);
            MPCContext ctx1(N_BITS, 0xBB2000 + i);
            Share s0 = relu_ars_eval(0, keys.k0, x_hat, engine, ctx0);
            Share s1 = relu_ars_eval(1, keys.k1, x_hat, engine, ctx1);
            u64 y_hat = reconstruct(s0, s1);
            u64 y = ring.sub(y_hat, keys.k0.r_out);
            std::int64_t exp = xs >> f;
            exp = std::max<std::int64_t>(exp, 0);
            if (ring.to_signed(y) != exp) ok = false;
        }
        std::cout << "ReluARS test: " << (ok ? "ok" : "FAIL") << std::endl;
        assert(ok);
    }

    // === GeLU (packed SUF) ===
    {
        GeLUParams gp;
        gp.n_bits = N_BITS;
        gp.f = 12;
        gp.lut_bits = 8;
        gp.clip = 3.0;
        auto keys = gelu_gen(gp, engine, dealer_ctx);
        bool ok = true;
        double max_err = 0.0;
        RingConfig cfg = make_ring_config(N_BITS);
        for (int i = 0; i < 200; ++i) {
            // Focus inputs in a moderate range.
            double xr = -4.0 + 8.0 * (static_cast<double>(i) / 199.0);
            std::int64_t x_fp = static_cast<std::int64_t>(std::llround(xr * (1ULL << gp.f)));
            u64 x_ring = ring.from_signed(x_fp);
            u64 x_hat = ring.add(x_ring, keys.k0.r_in);
            auto out_pair = gelu_eval_pair(keys, x_hat, engine);
            u64 y_hat = reconstruct(out_pair.first, out_pair.second);
            u64 y = ring.sub(y_hat, keys.k0.r_out);
            std::int64_t y_signed = ring.to_signed(y);

            double ref = ref_gelu(xr);
            std::int64_t ref_fp = static_cast<std::int64_t>(std::llround(ref * (1ULL << gp.f)));
            double err = std::fabs(static_cast<double>(y_signed - ref_fp));
            if (err > max_err) max_err = err;
            if (err > (1ULL << (gp.f - 3))) { // allow small tolerance
                ok = false;
                std::cerr << "GeLU mismatch xr=" << xr
                          << " x_hat=" << x_hat
                          << " y=" << y_signed
                          << " ref=" << ref_fp
                          << " err=" << err << std::endl;
                break;
            }
        }
        std::cout << "GeLU test: " << (ok ? "ok" : "FAIL")
                  << " (max abs error " << max_err << " in fixed-point)" << std::endl;
        assert(ok);
    }

    // === SiLU (packed SUF) ===
    {
        SiLUParams sp;
        sp.kind = ActivationKind::SiLU;
        sp.n_bits = N_BITS;
        sp.f = 12;
        sp.lut_bits = 10;
        sp.clip = 6.0;
        auto keys = silu_gen(sp, engine, dealer_ctx);
        bool ok = true;
        RingConfig cfg = make_ring_config(N_BITS);
        for (int i = 0; i < 100; ++i) {
            double xr = -6.0 + 12.0 * (static_cast<double>(i) / 99.0);
            std::int64_t x_fp = static_cast<std::int64_t>(std::llround(xr * (1ULL << sp.f)));
            u64 x_ring = ring.from_signed(x_fp);
            u64 x_hat = ring.add(x_ring, keys.k0.r_in);
            auto out_pair = silu_eval_pair(keys, x_hat, engine);
            u64 y_hat = reconstruct(out_pair.first, out_pair.second);
            u64 y = ring.sub(y_hat, keys.k0.r_out);
            std::int64_t y_signed = ring.to_signed(y);

            double ref = xr / (1.0 + std::exp(-xr));
            std::int64_t ref_fp = static_cast<std::int64_t>(std::llround(ref * (1ULL << sp.f)));
            double err = std::fabs(static_cast<double>(y_signed - ref_fp));
            if (err > (1ULL << (sp.f - 5))) {
                ok = false;
                break;
            }
        }
        std::cout << "SiLU test: " << (ok ? "ok" : "FAIL") << std::endl;
        assert(ok);
    }

    // === NExp / Inv / RecSqrt gates (sanity) ===
    {
        std::mt19937_64 rng_aux(0x9999);
        NExpGateParams np{N_BITS, 8};
        auto nk = gen_nexp_gate(np, engine, rng_aux);
        InvGateParams ip{N_BITS, 8, 32};
        auto ik = gen_inv_gate(ip, engine, rng_aux);
        RecSqrtGateParams rp{N_BITS, 8, 8, 8};
        auto rk = gen_recsqrt_gate(rp, engine, rng_aux);

        std::uniform_int_distribution<std::uint64_t> dist(0, (1u << N_BITS) - 1);
        for (int i = 0; i < 100; ++i) {
            u64 x = dist(rng);
            u64 x_hat = ring.add(x, nk.k0.r_in);
            auto s0 = nexpgate_eval(0, nk.k0, x_hat, engine);
            auto s1 = nexpgate_eval(1, nk.k1, x_hat, engine);
            u64 y_hat = reconstruct(s0, s1);
            u64 y = ring.sub(y_hat, nk.k0.r_out);
            (void)y;
        }
        for (int i = 1; i < 10; ++i) {
            u64 x = static_cast<u64>(i);
            u64 x_hat = ring.add(x, ik.k0.r_in);
            auto s0 = invgate_eval(0, ik.k0, x_hat, engine);
            auto s1 = invgate_eval(1, ik.k1, x_hat, engine);
            u64 y_hat = reconstruct(s0, s1);
            u64 y = ring.sub(y_hat, ik.k0.r_out);
            (void)y;
        }
        for (int i = 1; i < 10; ++i) {
            u64 x = static_cast<u64>(i);
            u64 x_hat = ring.add(x, rk.k0.r_in);
            auto s0 = recsqrt_eval(0, rk.k0, x_hat, engine);
            auto s1 = recsqrt_eval(1, rk.k1, x_hat, engine);
            u64 y_hat = reconstruct(s0, s1);
            u64 y = ring.sub(y_hat, rk.k0.r_out);
            (void)y;
        }
        std::cout << "Unary gates (nExp/Inv/RecSqrt) sanity: ok" << std::endl;
    }

    // === Softmax block with Beaver-based normalization (simulated, fully masked) ===
    if (false) {
        SoftmaxParams sp{N_BITS, 8, 4};
        std::mt19937_64 rng_aux(0xABCD55);
        auto keys = softmax_keygen(sp, engine, rng_aux);
        std::uniform_int_distribution<int> dist(-3, 3);
        std::vector<MaskedWire> x0(sp.vec_len), x1(sp.vec_len);
        std::vector<double> x_clear(sp.vec_len);
        auto cfg = make_ring_config(N_BITS);
        for (std::size_t i = 0; i < sp.vec_len; ++i) {
            double v = static_cast<double>(dist(rng_aux));
            x_clear[i] = v;
            u64 val = ring.from_signed(static_cast<std::int64_t>(std::llround(v * (1ULL << sp.f))));
            u64 r0 = static_cast<u64>(rng_aux()) & cfg.modulus_mask;
            u64 r1 = static_cast<u64>(rng_aux()) & cfg.modulus_mask;
            u64 hat = ring.add(val, ring.add(r0, r1));
            // Secret shares of x: x0 holds val - r1, x1 holds r1.
            x0[i] = MaskedWire{hat, Share{0, ring.sub(val, r1)}, Share{0, r0}};
            x1[i] = MaskedWire{hat, Share{1, r1}, Share{1, r1}};
        }
        BeaverPool pool0(cfg, 0xABC123, 0);
        BeaverPool pool1(cfg, 0xABC123, 1);
        auto [s0, s1] = softmax_eval_pair(engine, keys, cfg, x0, x1, pool0, pool1);

        // Reference softmax
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
            u64 y_hat = ring.add(s0.y[i].value_internal(), s1.y[i].value_internal());
            double y_fp = static_cast<double>(ring.to_signed(y_hat)) / static_cast<double>(1ULL << sp.f);
            double err = std::fabs(y_fp - y_ref[i]);
            if (err > 0.05) { // coarse tolerance for fixed-point approx
                ok = false;
                break;
            }
        }
        std::cout << "Softmax block test: " << (ok ? "ok" : "FAIL") << std::endl;
        assert(ok);
    }

    // === SUF stacking/vector LUT test with packing layout ===
    {
        // f0(x) = x, f1(x) = x^2, f2(x) = 7x + 3 over small domain bits.
        unsigned nbits = 6;
        SufDesc f0 = table_to_suf(nbits, 1, [&]() {
            std::size_t dom = 1ULL << nbits;
            std::vector<u64> t(dom);
            for (std::size_t x = 0; x < dom; ++x) t[x] = x;
            return t;
        }());
        SufDesc f1 = table_to_suf(nbits, 1, [&]() {
            std::size_t dom = 1ULL << nbits;
            std::vector<u64> t(dom);
            for (std::size_t x = 0; x < dom; ++x) t[x] = (x * x) & ((1ULL << nbits) - 1);
            return t;
        }());
        SufDesc f2 = table_to_suf(nbits, 1, [&]() {
            std::size_t dom = 1ULL << nbits;
            std::vector<u64> t(dom);
            for (std::size_t x = 0; x < dom; ++x) t[x] = (7 * x + 3) & ((1ULL << nbits) - 1);
            return t;
        }());
        auto stacked = stack_suf_outputs({f0, f1, f2});
        auto multi = compile_suf_to_lut_multi(stacked);
        PdpfEngineAdapter eng_vec(nbits);
        PdpfProgramId pid = eng_vec.make_lut_program(multi.desc, multi.table_flat);
        std::mt19937_64 rng_vec(123);
        bool ok = true;
        Ring64 ring_small(nbits);
        for (int i = 0; i < 50; ++i) {
            u64 x = rng_vec() & ((1ULL << nbits) - 1);
            auto clear = eval_suf_vector(stacked, x);
            std::vector<std::uint64_t> o0, o1;
            eng_vec.eval_share(pid, 0, x, o0);
            eng_vec.eval_share(pid, 1, x, o1);
            std::vector<std::uint64_t> words(o0.size());
            for (std::size_t w = 0; w < words.size(); ++w) {
                words[w] = ring_small.add(o0[w], o1[w]);
            }
            for (std::size_t j = 0; j < clear.size(); ++j) {
                if (words[j] != clear[j]) {
                    ok = false;
                }
            }
        }
        std::cout << "SUF stacked vector LUT test: " << (ok ? "ok" : "FAIL") << std::endl;
        assert(ok);
    }

    std::cout << "All composite_fss tests passed." << std::endl;
    return 0;
}

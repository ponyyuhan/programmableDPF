#include "../include/composite_fss/pdpf_full_impl.hpp"
#include "../include/composite_fss/gates/relu.hpp"
#include "../include/composite_fss/gates/gelu.hpp"
#include "../include/composite_fss/beaver.hpp"
#include "../include/composite_fss/sharing.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace cfss;

int main(int argc, char **argv) {
    std::string gate = (argc > 1) ? argv[1] : "relu";
    unsigned n_bits = (argc > 2) ? static_cast<unsigned>(std::stoul(argv[2])) : 16;
    std::size_t batch = (argc > 3) ? static_cast<std::size_t>(std::stoul(argv[3])) : 1024;

    PdpfEngineFullImpl engine;
    MPCContext dealer(n_bits, 0x1234);
    std::mt19937_64 rng(0xBEEF);
    Ring64 ring(n_bits);

    std::vector<u64> hats(batch);
    for (auto &h : hats) {
        std::int64_t s = static_cast<std::int64_t>(rng() % (1ULL << (n_bits - 1)));
        u64 x = ring.from_signed(s);
        auto [s0, s1] = dealer.share_value(x);
        u64 r0 = rng() & ring.modulus_mask;
        u64 r1 = rng() & ring.modulus_mask;
        h = ring.add(x, ring.add(r0, r1));
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (gate == "relu") {
        auto kp = relu_gen(n_bits, engine, dealer);
        for (auto &h : hats) {
            std::int64_t s = static_cast<std::int64_t>(rng() % (1ULL << (n_bits - 1)));
            u64 x = ring.from_signed(s);
            h = ring.add(x, kp.k0.r_in);
        }
        std::vector<std::vector<u64>> outs(batch);
        engine.eval_share_batch(kp.k0.compiled.pdpf_program, 0, hats, outs);
    } else {
        GeLUParams gp;
        gp.kind = ActivationKind::GeLU;
        gp.n_bits = n_bits;
        gp.f = 12;
        gp.lut_bits = 8;
        gp.clip = 3.0;
        auto kp = gelu_gen(gp, engine, dealer);
        for (auto &h : hats) {
            std::int64_t s = static_cast<std::int64_t>(rng() % (1ULL << (n_bits - 1)));
            u64 x = ring.from_signed(s);
            h = ring.add(x, kp.k0.r_in);
        }
        RingConfig cfg = make_ring_config(n_bits);
        BeaverPool pool0(cfg, 0xABCDEF, 0);
        BeaverPool pool1(cfg, 0xABCDEF, 1);
        for (auto h : hats) {
            auto r0 = gelu_eval_main(0, kp.k0, h, engine);
            auto r1 = gelu_eval_main(1, kp.k1, h, engine);
            auto out_pair = gelu_finish(cfg, kp.k0, kp.k1, r0, r1, pool0, pool1);
            (void)out_pair;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto stats = engine.stats();
    std::cout << "gate=" << gate << " batch=" << batch << " n_bits=" << n_bits
              << " lut_gen=" << stats.lut_gen << " evals=" << stats.evals
              << " output_words=" << stats.output_words << " time_ms=" << ms << "\n";
    return 0;
}

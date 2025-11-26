#include "../include/composite_fss/pdpf_adapter.hpp"
#include "../include/composite_fss/gates/gelu.hpp"
#include "../include/composite_fss/gates/softmax.hpp"
#include "../include/composite_fss/gates/recip.hpp"
#include "../include/composite_fss/gates/norm.hpp"
#include "../include/composite_fss/gates/fused_layer.hpp"
#include "../include/composite_fss/metrics.hpp"
#include "../include/composite_fss/beaver.hpp"
#include "../include/composite_fss/wire.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <cstdlib>
#include <unordered_set>

using namespace cfss;

std::size_t lut_bytes(const PdpfEngine &eng, PdpfProgramId pid) {
    auto desc = eng.lookup_lut_desc(pid);
    if (desc.output_words == 0) return 0;
    std::size_t dom = (desc.domain_bits == 64) ? 0 : (1ULL << desc.domain_bits);
    if (dom == 0) return 0;
    return dom * desc.output_words * sizeof(std::uint64_t);
}

struct BenchResult {
    std::string gate;
    unsigned n_bits = 0;
    unsigned f = 0;
    std::size_t dim = 0;
    std::size_t key_bytes = 0;
    std::size_t lut_bytes_total = 0;
    std::size_t triples = 0;
    double keygen_ms = 0.0;
    double online_ms = 0.0;
};

void print_csv_header() {
    std::cout << "gate,n_bits,f,dim,key_bytes,lut_bytes,triples,keygen_ms,online_ms\n";
}

void print_row(const BenchResult &r) {
    std::cout << r.gate << ","
              << r.n_bits << ","
              << r.f << ","
              << r.dim << ","
              << r.key_bytes << ","
              << r.lut_bytes_total << ","
              << r.triples << ","
              << r.keygen_ms << ","
              << r.online_ms << "\n";
}

std::size_t sum_program_bytes(const PdpfEngine &eng,
                              const std::vector<PdpfProgramId> &pids) {
    const auto *adapter = dynamic_cast<const PdpfEngineAdapter *>(&eng);
    if (!adapter) return 0;
    std::unordered_set<PdpfProgramId> uniq;
    std::size_t total = 0;
    for (auto pid : pids) {
        if (uniq.insert(pid).second) {
            total += adapter->program_bytes(pid);
        }
    }
    return total;
}

BenchResult bench_gelu(unsigned n_bits, unsigned f, unsigned iters) {
    BenchResult r;
    r.gate = "gelu";
    r.n_bits = n_bits;
    r.f = f;
    PdpfEngineAdapter engine(n_bits, 0xA5A5, select_backend_from_env());
    MPCContext dealer(n_bits, 0xACE);
    auto t0 = std::chrono::steady_clock::now();
    GeLUParams gp;
    gp.n_bits = n_bits;
    gp.f = f;
    gp.lut_bits = 8;
    gp.clip = 3.0;
    auto keys = gelu_gen(gp, engine, dealer);
    auto t1 = std::chrono::steady_clock::now();
    r.keygen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.lut_bytes_total = lut_bytes(engine, keys.k0.main_prog.compiled.pdpf_program);
    r.key_bytes = sum_program_bytes(engine, {keys.k0.main_prog.compiled.pdpf_program});

    RingConfig cfg = make_ring_config(n_bits);
    std::mt19937_64 rng(0xF00D);
    BeaverPool pool0(cfg, 0xBEEF, 0);
    BeaverPool pool1(cfg, 0xBEEF, 1);

    auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iters; ++i) {
        std::int64_t sample = static_cast<std::int64_t>(rng() & ((1ULL << (n_bits - 1)) - 1));
        u64 x_ring = cfg.n_bits == 64 ? static_cast<u64>(sample) : (static_cast<u64>(sample) & cfg.modulus_mask);
        u64 x_hat = x_ring; // GeLU keys use r_in=0
        auto r0 = gelu_eval_main(0, keys.k0, x_hat, engine);
        auto r1 = gelu_eval_main(1, keys.k1, x_hat, engine);
        (void)activation_finish(cfg, keys.k0, r0, pool0);
        (void)activation_finish(cfg, keys.k1, r1, pool1);
    }
    auto end = std::chrono::steady_clock::now();
    r.online_ms = std::chrono::duration<double, std::milli>(end - start).count();
    r.triples = pool0.counters().triples_used;
    return r;
}

BenchResult bench_softmax(unsigned n_bits, unsigned f, std::size_t vec_len, unsigned iters) {
    BenchResult r;
    r.gate = "softmax";
    r.n_bits = n_bits;
    r.f = f;
    r.dim = vec_len;
    PdpfEngineAdapter engine(n_bits, 0xA5A5, select_backend_from_env());
    std::mt19937_64 rng(0x1234);
    auto t0 = std::chrono::steady_clock::now();
    SoftmaxParams sp{n_bits, f, vec_len};
    auto keys = softmax_keygen(sp, engine, rng);
    auto t1 = std::chrono::steady_clock::now();
    r.keygen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.lut_bytes_total = lut_bytes(engine, keys.k0.inv_key.prog);
    r.key_bytes = sum_program_bytes(engine, {keys.k0.inv_key.prog});
    r.lut_bytes_total += lut_bytes(engine, keys.k0.nexp_kernel.prog);
    r.lut_bytes_total += lut_bytes(engine, keys.k0.trunc_key.compiled.pdpf_program);
    for (const auto &k : keys.k0.drelu_keys) r.lut_bytes_total += lut_bytes(engine, k.compiled.pdpf_program);
    std::vector<PdpfProgramId> pids;
    pids.push_back(keys.k0.inv_key.prog);
    pids.push_back(keys.k0.nexp_kernel.prog);
    pids.push_back(keys.k0.trunc_key.compiled.pdpf_program);
    for (const auto &k : keys.k0.drelu_keys) pids.push_back(k.compiled.pdpf_program);
    r.key_bytes = sum_program_bytes(engine, pids);

    RingConfig cfg = make_ring_config(n_bits);
    std::uniform_int_distribution<int> dist(-3, 3);

    auto start = std::chrono::steady_clock::now();
    for (unsigned iter = 0; iter < iters; ++iter) {
        std::vector<MaskedWire> x0(vec_len), x1(vec_len);
        std::vector<double> x_clear(vec_len);
        for (std::size_t i = 0; i < vec_len; ++i) {
            double v = static_cast<double>(dist(rng));
            x_clear[i] = v;
            u64 val = static_cast<u64>(std::llround(v * (1ULL << f))) & cfg.modulus_mask;
            u64 r0 = static_cast<u64>(rng()) & cfg.modulus_mask;
            u64 r1 = static_cast<u64>(rng()) & cfg.modulus_mask;
            u64 hat = ring_add(cfg, val, ring_add(cfg, r0, r1));
            x0[i] = MaskedWire{hat, Share{0, ring_sub(cfg, val, r1)}, Share{0, r0}};
            x1[i] = MaskedWire{hat, Share{1, r1}, Share{1, r1}};
        }
        BeaverPool pool0(cfg, 0xA11CE + iter, 0);
        BeaverPool pool1(cfg, 0xA11CE + iter, 1);
        auto out_pair = softmax_eval_pair(engine, keys, cfg, x0, x1, pool0, pool1);
        (void)out_pair;
        r.triples += pool0.counters().triples_used;
    }
    auto end = std::chrono::steady_clock::now();
    r.online_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return r;
}

BenchResult bench_recip(unsigned n_bits, unsigned f_in, unsigned f_out, unsigned iters) {
    BenchResult r;
    r.gate = "recip";
    r.n_bits = n_bits;
    r.f = f_out;
    PdpfEngineAdapter engine(n_bits, 0xA5A5, select_backend_from_env());
    std::mt19937_64 rng(0xCAFE);
    RecipParams rp{n_bits, f_in, f_out, 1024, false};
    auto t0 = std::chrono::steady_clock::now();
    auto keys = gen_recip_gate(rp, engine, rng);
    auto t1 = std::chrono::steady_clock::now();
    r.keygen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.lut_bytes_total = lut_bytes(engine, keys.k0.inv_key.prog);

    RingConfig cfg = make_ring_config(n_bits);
    BeaverPool pool0(cfg, 0xD00D, 0);
    BeaverPool pool1(cfg, 0xD00D, 1);
    std::uniform_int_distribution<int> dist(1, 20);
    auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iters; ++i) {
        int v = dist(rng);
        u64 x_fp = static_cast<u64>(std::llround(static_cast<double>(v) * (1ULL << f_in))) & cfg.modulus_mask;
        auto shares = MPCContext(n_bits, 0xEE).share_value(x_fp);
        auto rec_pair = recip_eval_from_share_pair(cfg, keys.k0, keys.k1, shares.first, shares.second, engine, pool0, pool1);
        (void)rec_pair;
    }
    auto end = std::chrono::steady_clock::now();
    r.online_ms = std::chrono::duration<double, std::milli>(end - start).count();
    r.triples = pool0.counters().triples_used;
    return r;
}

BenchResult bench_norm(unsigned n_bits, unsigned f, std::size_t dim, unsigned iters) {
    BenchResult r;
    r.gate = "norm";
    r.n_bits = n_bits;
    r.f = f;
    r.dim = dim;
    PdpfEngineAdapter engine(n_bits, 0xA5A5, select_backend_from_env());
    std::mt19937_64 rng(0x7777);
    NormParams np;
    np.n_bits = n_bits;
    np.f = f;
    np.dim = dim;
    auto t0 = std::chrono::steady_clock::now();
    auto keys = norm_keygen(np, engine, rng);
    auto t1 = std::chrono::steady_clock::now();
    r.keygen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.lut_bytes_total = lut_bytes(engine, keys.k0.inv_sqrt_key.inv_key.prog);
    r.lut_bytes_total += lut_bytes(engine, keys.k0.trunc_f.compiled.pdpf_program);
    r.key_bytes = sum_program_bytes(engine,
                                    {keys.k0.inv_sqrt_key.inv_key.prog,
                                     keys.k0.trunc_f.compiled.pdpf_program});

    RingConfig cfg = make_ring_config(n_bits);
    BeaverPool pool0(cfg, 0xFACE, 0);
    BeaverPool pool1(cfg, 0xFACE, 1);
    std::uniform_int_distribution<int> dist(-3, 3);
    auto start = std::chrono::steady_clock::now();
    for (unsigned iter = 0; iter < iters; ++iter) {
        std::vector<Share> x0(dim), x1(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            int v = dist(rng);
            u64 fp = static_cast<u64>(std::llround(static_cast<double>(v) * (1ULL << f))) & cfg.modulus_mask;
            auto sh = MPCContext(n_bits, 0xABC + i).share_value(fp);
            x0[i] = sh.first;
            x1[i] = sh.second;
        }
        auto out_pair = norm_eval_pair(keys, cfg, x0, x1, engine, pool0, pool1);
        (void)out_pair;
    }
    auto end = std::chrono::steady_clock::now();
    r.online_ms = std::chrono::duration<double, std::milli>(end - start).count();
    r.triples = pool0.counters().triples_used;
    return r;
}

int main(int argc, char **argv) {
    unsigned iters = 100;
    std::size_t softmax_len = 4;
    std::size_t norm_dim = 4;
    bool use_gpu = false;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-gpu") {
            use_gpu = true;
            continue;
        }
        positional.push_back(arg);
    }
    if (use_gpu) {
        setenv("COMPOSITE_FSS_USE_GPU", "1", 1);
    }
    if (!positional.empty()) iters = static_cast<unsigned>(std::stoul(positional[0]));
    if (positional.size() > 1) softmax_len = static_cast<std::size_t>(std::stoul(positional[1]));
    if (positional.size() > 2) norm_dim = static_cast<std::size_t>(std::stoul(positional[2]));
    print_csv_header();
    print_row(bench_gelu(16, 8, iters));
    print_row(bench_softmax(16, 8, softmax_len, iters / 4 + 1));
    print_row(bench_recip(16, 8, 6, iters));
    print_row(bench_norm(16, 8, norm_dim, iters / 2 + 1));
    return 0;
}

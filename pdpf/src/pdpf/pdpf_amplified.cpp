// ================================================================
// File: src/pdpf/pdpf_amplified.cpp
// ================================================================

#include "pdpf/pdpf/pdpf_amplified.hpp"
#include <stdexcept>

namespace pdpf::pdpf {

PdpfAmplified::PdpfAmplified(std::shared_ptr<prg::IPrg> prg,
                             const ldc::LdcParams &ldc_params,
                             std::uint64_t prime_p)
    : prg_(std::move(prg))
    , base_pdpf_(prg_)
    , ldc_(ldc_params)
    , prime_p_(prime_p) {
    if (prime_p_ == 0) {
        throw std::invalid_argument("PdpfAmplified: prime_p must be non-zero prime");
    }
}

core::Seed PdpfAmplified::derive_inner_seed(const core::Seed &master,
                                            std::uint64_t index) const {
    core::Seed material = master;
    core::Seed left{}, right{};
    // Simple derivation: iterate index+1 times of expand, take left output.
    for (std::uint64_t i = 0; i <= index; ++i) {
        prg_->expand(material, left, right);
        material = left;
    }
    return material;
}

std::int64_t PdpfAmplified::mod_p(std::int64_t x) const {
    if (prime_p_ == 0) return x;
    std::int64_t r = static_cast<std::int64_t>(x % static_cast<std::int64_t>(prime_p_));
    if (r < 0) r += static_cast<std::int64_t>(prime_p_);
    return r;
}

AmplifiedOfflineKey PdpfAmplified::gen_offline(const core::SecurityParams &sec) {
    AmplifiedOfflineKey k0;
    k0.sec       = sec;
    k0.ldc_params = ldc_.params();
    k0.prime_p   = prime_p_;

    core::RandomDevice rng;
    rng.random_seed(k0.master_seed);

    return k0;
}

AmplifiedOnlineKey PdpfAmplified::gen_online(const AmplifiedOfflineKey &k0,
                                             std::uint64_t alpha,
                                             std::int64_t beta) {
    (void)beta; // base PDPF here is binary; extension to Z_p is via Theorem 5 if needed.

    const auto &ldc_params = k0.ldc_params;
    std::uint64_t q = ldc_params.q;
    std::uint64_t L = ldc_params.L;

    // 1. Compute Δ ← d(α) ∈ [L]^q.
    auto Delta = ldc_.sample_indices(alpha);

    if (Delta.size() != q) {
        throw std::runtime_error("PdpfAmplified::gen_online: wrong Δ size");
    }

    AmplifiedOnlineKey k1;
    k1.sec        = k0.sec;
    k1.ldc_params = ldc_params;
    k1.prime_p    = prime_p_;
    k1.inner_keys.reserve(q);
    k1.deltas     = Delta;

    // 2. For each ℓ = 1..q:
    //    - derive k*_ℓ from master_seed
    //    - form OfflineKey for base PDPF over domain L (here domain size is L)
    //    - run base_pdpf_.gen_online for target index Δ_ℓ.
    core::SecurityParams inner_sec = k0.sec;
    inner_sec.domain_size_N = L;

    for (std::uint64_t i = 0; i < q; ++i) {
        core::Seed inner_seed = derive_inner_seed(k0.master_seed, i);
        OfflineKey inner_off = base_pdpf_.gen_offline(inner_sec);
        inner_off.k_star = inner_seed; // override seed for determinism

        // Binary payload β=1 at index Δ_i (see Figure 2). :contentReference[oaicite:23]{index=23}
        OnlineKey inner_on = base_pdpf_.gen_online(inner_off, Delta[i], 1);
        k1.inner_keys.push_back(std::move(inner_on));
    }

    return k1;
}

std::int64_t PdpfAmplified::eval_offline(const AmplifiedOfflineKey &k0,
                                         const AmplifiedOnlineKey &k1,
                                         std::uint64_t x) const {
    const auto &ldc_params = k0.ldc_params;
    std::uint64_t q = ldc_params.q;
    std::uint64_t L = ldc_params.L;

    // Encode unit vector e_x into C(e_x) ∈ Z_p^L.
    std::vector<std::int64_t> C_ex;
    ldc_.encode_unit(x, C_ex);
    if (C_ex.size() < L) {
        C_ex.resize(L, 0);
    }

    core::SecurityParams inner_sec = k0.sec;
    inner_sec.domain_size_N = L;

    std::int64_t acc = 0;
    for (std::uint64_t i = 0; i < q; ++i) {
        core::Seed inner_seed = derive_inner_seed(k0.master_seed, i);
        OfflineKey inner_off = base_pdpf_.gen_offline(inner_sec);
        inner_off.k_star = inner_seed;

        std::vector<core::GroupZ::Value> Y0;
        base_pdpf_.eval_all_offline(inner_off, Y0);

        std::uint64_t delta_i = (i < k1.deltas.size()) ? k1.deltas[i] % L : 0;
        if (delta_i < C_ex.size() && delta_i < Y0.size()) {
            acc = mod_p(acc + C_ex[delta_i] * Y0[delta_i]);
        }
    }
    return mod_p(acc);
}

std::int64_t PdpfAmplified::eval_online(const AmplifiedOfflineKey &k0,
                                        const AmplifiedOnlineKey &k1,
                                        std::uint64_t x) const {
    const auto &ldc_params = k0.ldc_params;
    std::uint64_t q = ldc_params.q;
    std::uint64_t L = ldc_params.L;

    std::vector<std::int64_t> C_ex;
    ldc_.encode_unit(x, C_ex);
    if (C_ex.size() < L) {
        C_ex.resize(L, 0);
    }

    std::int64_t acc = 0;
    for (std::uint64_t i = 0; i < q && i < k1.inner_keys.size(); ++i) {
        std::vector<core::GroupZ::Value> Y1;
        base_pdpf_.eval_all_online(k1.inner_keys[i], Y1);

        std::uint64_t delta_i = (i < k1.deltas.size()) ? k1.deltas[i] % L : 0;
        if (delta_i < C_ex.size() && delta_i < Y1.size()) {
            acc = mod_p(acc + C_ex[delta_i] * Y1[delta_i]);
        }
    }
    return mod_p(acc);
}

void PdpfAmplified::eval_all_offline(const AmplifiedOfflineKey &k0,
                                     const AmplifiedOnlineKey &k1,
                                     std::vector<std::int64_t> &Y0) const {
    std::uint64_t N = k0.sec.domain_size_N;
    Y0.resize(N);
    for (std::uint64_t x = 0; x < N; ++x) {
        Y0[x] = eval_offline(k0, k1, x);
    }
}

void PdpfAmplified::eval_all_online(const AmplifiedOfflineKey &k0,
                                    const AmplifiedOnlineKey &k1,
                                    std::vector<std::int64_t> &Y1) const {
    std::uint64_t N = k0.sec.domain_size_N;
    Y1.resize(N);
    for (std::uint64_t x = 0; x < N; ++x) {
        Y1[x] = eval_online(k0, k1, x);
    }
}

} // namespace pdpf::pdpf

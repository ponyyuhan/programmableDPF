// ================================================================
// File: src/pdpf/pdpf_binary.cpp
// ================================================================

#include "pdpf/pdpf/pdpf_binary.hpp"
#include <cmath>
#include <stdexcept>

namespace pdpf::pdpf {

PdpfBinary::PdpfBinary(std::shared_ptr<prg::IPrg> prg)
    : prg_(std::move(prg)) {}

std::uint64_t PdpfBinary::choose_M(const core::SecurityParams &sec) const {
    const double N = static_cast<double>(sec.domain_size_N);
    const double eps = sec.epsilon;
    if (eps <= 0.0) {
        throw std::invalid_argument("PdpfBinary::choose_M: epsilon <= 0");
    }
    // Heuristic from Section 5: M ≈ 0.318·(N+1)/ε^2. :contentReference[oaicite:13]{index=13}
    double M_real = 0.318 * (N + 1.0) / (eps * eps);
    if (M_real < N + 1.0) {
        M_real = N + 1.0;
    }
    return static_cast<std::uint64_t>(std::ceil(M_real));
}

std::uint64_t PdpfBinary::seed_to_uint64(const core::Seed &s) const {
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < 8 && i < s.size(); ++i) {
        v |= static_cast<std::uint64_t>(s[i]) << (8 * i);
    }
    return v;
}

OfflineKey PdpfBinary::gen_offline(const core::SecurityParams &sec) const {
    OfflineKey k0;
    k0.params.sec = sec;
    k0.params.M   = choose_M(sec);

    core::RandomDevice rng;
    rng.random_seed(k0.k_star);

    return k0;
}

OnlineKey PdpfBinary::gen_online(const OfflineKey &k0,
                                 std::uint64_t alpha,
                                 std::uint8_t beta) {
    if (beta != 0 && beta != 1) {
        throw std::invalid_argument("PdpfBinary::gen_online: beta not in {0,1}");
    }
    if (alpha >= k0.params.sec.domain_size_N) {
        throw std::out_of_range("PdpfBinary::gen_online: alpha >= N");
    }

    const std::uint64_t N = k0.params.sec.domain_size_N;
    const std::uint64_t M = k0.params.M;

    // 1. Expand k* → (s, k_PPRF_seed)
    core::Seed s_seed{}, k_pprf_seed{};
    prg_->expand(k0.k_star, s_seed, k_pprf_seed);

    // 2. Build PPRF key over [M] → [N+1].
    pprf::PprfParams pparams{M, N + 1};
    pprf::PprfKey pkey{k_pprf_seed, pparams};
    pprf::Pprf pprf(prg_);

    // 3. Find all ℓ with shifted PPRF output == target.
    std::vector<std::uint64_t> candidates;
    candidates.reserve(static_cast<std::size_t>(M / (N + 1)));

    // Using 0-based domain: target is alpha for β=1, dummy bucket N for β=0.
    std::uint64_t target = (beta == 1) ? alpha : N;

    std::uint64_t s_val = seed_to_uint64(s_seed) % (N + 1);

    for (std::uint64_t ell = 0; ell < M; ++ell) {
        std::uint64_t val = pprf.eval(pkey, ell); // ∈ [0, N]
        std::uint64_t shifted = (val + s_val) % (N + 1);
        if (shifted == target) {
            candidates.push_back(ell);
        }
    }

    if (candidates.empty()) {
        // As in Theorem 4, you can attribute this negligible failure to correctness or privacy. :contentReference[oaicite:14]{index=14}
        throw std::runtime_error("PdpfBinary::gen_online: candidate set L is empty");
    }

    core::RandomDevice rng;
    std::uint64_t idx = rng.random_u64(candidates.size());
    std::uint64_t ell_star = candidates[static_cast<std::size_t>(idx)];

    // 4. Puncture PPRF at ℓ*.
    auto kp = pprf.puncture(pkey, ell_star);

    OnlineKey k1;
    k1.kp = std::move(kp);
    k1.s  = s_seed;
    k1.params = k0.params;

    return k1;
}

void PdpfBinary::eval_all_offline(const OfflineKey &k0,
                                  std::vector<core::GroupZ::Value> &Y) const {
    const std::uint64_t N = k0.params.sec.domain_size_N;
    const std::uint64_t M = k0.params.M;

    Y.assign(N, 0);

    // Re-derive (s, k_PPRF_seed) from k*.
    core::Seed s_seed{}, k_pprf_seed{};
    prg_->expand(k0.k_star, s_seed, k_pprf_seed);

    pprf::PprfParams pparams{M, N + 1};
    pprf::PprfKey pkey{k_pprf_seed, pparams};
    pprf::Pprf pprf(prg_);

    std::uint64_t s_val = seed_to_uint64(s_seed) % (N + 1);

    for (std::uint64_t ell = 0; ell < M; ++ell) {
        std::uint64_t val = pprf.eval(pkey, ell);  // [0..N]
        std::uint64_t shifted = (val + s_val) % (N + 1);

        // Only count in real-domain buckets [0..N-1]; bucket N is dummy.
        if (shifted < N) {
            std::size_t idx = static_cast<std::size_t>(shifted);
            Y[idx] += 1;
        }
    }
}

void PdpfBinary::eval_all_online(const OnlineKey &k1,
                                 std::vector<core::GroupZ::Value> &Y) const {
    const std::uint64_t N = k1.params.sec.domain_size_N;
    const std::uint64_t M = k1.params.M;

    Y.assign(N, 0);

    pprf::Pprf pprf(prg_);
    std::uint64_t s_val = seed_to_uint64(k1.s) % (N + 1);

    for (std::uint64_t ell = 0; ell < M; ++ell) {
        std::uint64_t val = pprf.punc_eval(k1.kp, ell);
        if (val == pprf::Pprf::PUNCTURED_SENTINEL) {
            continue;
        }
        std::uint64_t shifted = (val + s_val) % (N + 1);
        if (shifted < N) {
            std::size_t idx = static_cast<std::size_t>(shifted);
            Y[idx] -= 1;
        }
    }
}

} // namespace pdpf::pdpf

// ================================================================
// File: src/ldc/reed_muller_ldc.cpp
// ================================================================

#include "pdpf/ldc/reed_muller_ldc.hpp"
#include "pdpf/core/types.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace pdpf::ldc {

namespace {

static bool is_prime(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint64_t i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

static std::uint64_t next_prime(std::uint64_t n) {
    if (n < 2) n = 2;
    while (!is_prime(n)) ++n;
    return n;
}

static std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exp, std::uint64_t mod) {
    std::uint64_t res = 1 % mod;
    base %= mod;
    while (exp) {
        if (exp & 1) res = (__uint128_t)res * base % mod;
        base = (__uint128_t)base * base % mod;
        exp >>= 1;
    }
    return res;
}

static std::int64_t mod_inv(std::int64_t a, std::int64_t mod) {
    std::int64_t t = 0, newt = 1;
    std::int64_t r = mod, newr = a % mod;
    while (newr != 0) {
        std::int64_t q = r / newr;
        std::int64_t tmp = newt;
        newt = t - q * newt;
        t = tmp;
        tmp = newr;
        newr = r - q * newr;
        r = tmp;
    }
    if (r > 1) throw std::invalid_argument("mod_inv: non-invertible");
    if (t < 0) t += mod;
    return t;
}

// Generate all monomials of total degree <= r in w variables.
static void gen_monomials(std::size_t w, std::uint32_t r, std::vector<std::vector<std::uint32_t>> &out,
                          std::vector<std::uint32_t> cur, std::size_t idx, std::uint32_t remaining) {
    if (idx == w) {
        if (remaining >= 0) out.push_back(cur);
        return;
    }
    for (std::uint32_t d = 0; d <= r; ++d) {
        std::uint32_t sum = 0;
        for (auto v : cur) sum += v;
        if (sum + d > r) break;
        cur[idx] = d;
        gen_monomials(w, r, out, cur, idx + 1, remaining > d ? remaining - d : 0);
    }
}

static std::uint64_t monomial_eval(const std::vector<std::uint32_t> &exp,
                                   const std::vector<std::uint64_t> &x,
                                   std::uint64_t mod) {
    std::uint64_t res = 1 % mod;
    for (std::size_t i = 0; i < exp.size(); ++i) {
        if (exp[i] == 0) continue;
        res = (__uint128_t)res * mod_pow(x[i], exp[i], mod) % mod;
    }
    return res;
}

// Solve A*c = b (mod mod) using Gaussian elimination; A is square.
static std::vector<std::uint64_t> solve_lin(std::vector<std::vector<std::uint64_t>> A,
                                            std::vector<std::uint64_t> b,
                                            std::uint64_t mod) {
    const std::size_t n = b.size();
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t pivot = i;
        while (pivot < n && A[pivot][i] % mod == 0) ++pivot;
        if (pivot == n) throw std::runtime_error("singular matrix");
        if (pivot != i) {
            std::swap(A[pivot], A[i]);
            std::swap(b[pivot], b[i]);
        }
        auto inv = mod_inv(static_cast<std::int64_t>(A[i][i] % mod), static_cast<std::int64_t>(mod));
        for (std::size_t j = i; j < n; ++j) {
            A[i][j] = (__uint128_t)A[i][j] * inv % mod;
        }
        b[i] = (__uint128_t)b[i] * inv % mod;
        for (std::size_t r = 0; r < n; ++r) {
            if (r == i) continue;
            std::uint64_t factor = A[r][i] % mod;
            if (factor == 0) continue;
            for (std::size_t c = i; c < n; ++c) {
                std::uint64_t val = (A[r][c] + mod - (__uint128_t)factor * A[i][c] % mod) % mod;
                A[r][c] = val;
            }
            std::uint64_t valb = (b[r] + mod - (__uint128_t)factor * b[i] % mod) % mod;
            b[r] = valb;
        }
    }
    return b;
}

} // namespace

ReedMullerLdc::ReedMullerLdc(const LdcParams &params)
    : params_(params) {
    if (params_.p == 0 || params_.w == 0 || params_.r == 0) {
        throw std::invalid_argument("ReedMullerLdc: invalid parameters");
    }
    if (params_.q == 0) {
        params_.q = static_cast<std::uint64_t>((params_.sigma + 1) * (params_.r * params_.sigma + 1));
    }
    std::uint64_t Q = next_prime(std::max<std::uint64_t>(params_.p, params_.r * params_.sigma + 2));
    std::uint64_t base = 1;
    for (std::size_t i = 0; i < params_.w; ++i) {
        base *= Q;
    }
    params_.L = base * Q;
}

void ReedMullerLdc::encode(const std::vector<std::int64_t> &z,
                           std::vector<std::int64_t> &codeword) const {
    if (z.size() != params_.N) {
        throw std::invalid_argument("encode: input length mismatch");
    }
    std::uint64_t Q = next_prime(std::max<std::uint64_t>(params_.p, params_.r * params_.sigma + 2));
    std::uint64_t p = params_.p;

    // Points in general position: first N lex points in F^w.
    std::vector<std::vector<std::uint64_t>> points(params_.N, std::vector<std::uint64_t>(params_.w, 0));
    for (std::uint64_t idx = 0; idx < params_.N; ++idx) {
        std::uint64_t t = idx;
        for (std::size_t j = 0; j < params_.w; ++j) {
            points[idx][j] = t % Q;
            t /= Q;
        }
    }

    // Monomials up to degree r; pick first N for square system.
    std::vector<std::vector<std::uint32_t>> monos;
    gen_monomials(params_.w, params_.r, monos, std::vector<std::uint32_t>(params_.w, 0), 0, params_.r);
    if (monos.size() < params_.N) {
        throw std::runtime_error("encode: insufficient monomials for interpolation");
    }
    monos.resize(params_.N);

    // Build A and b for interpolation.
    std::vector<std::vector<std::uint64_t>> A(params_.N, std::vector<std::uint64_t>(params_.N, 0));
    for (std::size_t i = 0; i < params_.N; ++i) {
        for (std::size_t j = 0; j < params_.N; ++j) {
            A[i][j] = monomial_eval(monos[j], points[i], Q);
        }
    }
    std::vector<std::uint64_t> b(params_.N, 0);
    for (std::size_t i = 0; i < params_.N; ++i) {
        std::int64_t v = z[i] % static_cast<std::int64_t>(Q);
        if (v < 0) v += Q;
        b[i] = static_cast<std::uint64_t>(v);
    }
    std::vector<std::uint64_t> coeffs = solve_lin(A, b, Q);

    // Evaluate codeword for all (rho, x).
    codeword.assign(params_.L, 0);
    std::uint64_t base = 1;
    for (std::size_t i = 0; i < params_.w; ++i) base *= Q;

    for (std::uint64_t rho = 0; rho < Q; ++rho) {
        for (std::uint64_t xi = 0; xi < base; ++xi) {
            std::vector<std::uint64_t> xvec(params_.w, 0);
            std::uint64_t t = xi;
            for (std::size_t j = 0; j < params_.w; ++j) {
                xvec[j] = t % Q;
                t /= Q;
            }
            std::uint64_t val = 0;
            for (std::size_t j = 0; j < monos.size(); ++j) {
                std::uint64_t term = monomial_eval(monos[j], xvec, Q);
                val = (val + (__uint128_t)coeffs[j] * term) % Q;
            }
            std::uint64_t idx = rho * base + xi;
            if (idx < codeword.size()) {
                std::uint64_t mapped = (rho * val) % Q;
                std::int64_t out = static_cast<std::int64_t>(mapped % p);
                codeword[idx] = out;
            }
        }
    }
}

void ReedMullerLdc::encode_unit(std::uint64_t x,
                                std::vector<std::int64_t> &codeword) const {
    if (x >= params_.N) {
        throw std::out_of_range("encode_unit: x >= N");
    }
    std::vector<std::int64_t> z(params_.N, 0);
    z[x] = 1 % static_cast<std::int64_t>(params_.p);
    encode(z, codeword);
}

std::vector<std::uint64_t> ReedMullerLdc::sample_indices(std::uint64_t alpha) const {
    if (params_.p == 0 || params_.w == 0 || params_.r == 0) {
        throw std::invalid_argument("sample_indices: invalid params");
    }
    std::size_t q = (params_.q == 0) ? 1 : static_cast<std::size_t>(params_.q);
    std::uint64_t Q = next_prime(std::max<std::uint64_t>(params_.p, params_.r * params_.sigma + 2));

    // Base point x_alpha in F^w (lex from alpha).
    std::vector<std::uint64_t> xvec(params_.w, 0);
    std::uint64_t tmp = alpha;
    for (std::size_t j = 0; j < params_.w; ++j) {
        xvec[j] = tmp % Q;
        tmp /= Q;
    }

    const std::size_t blocks = static_cast<std::size_t>(params_.r * params_.sigma) + 1;
    const std::size_t q_needed = (params_.sigma + 1) * blocks;

    pdpf::core::RandomDevice rng;

    // Distinct nonzero s values.
    std::vector<std::uint64_t> s_vals;
    while (s_vals.size() < blocks) {
        std::uint64_t s = rng.random_u64(Q - 1) + 1;
        if (std::find(s_vals.begin(), s_vals.end(), s) == s_vals.end()) {
            s_vals.push_back(s);
        }
    }

    // Lagrange coefficients c for interpolation at 0.
    std::vector<std::uint64_t> c(blocks, 0);
    for (std::size_t i = 0; i < blocks; ++i) {
        std::uint64_t num = 1, den = 1;
        for (std::size_t j = 0; j < blocks; ++j) {
            if (i == j) continue;
            num = (__uint128_t)num * ((Q + 0 - s_vals[j] % Q) % Q) % Q; // -s_j
            std::int64_t diff = static_cast<std::int64_t>(s_vals[i]) - static_cast<std::int64_t>(s_vals[j]);
            diff %= static_cast<std::int64_t>(Q);
            if (diff < 0) diff += Q;
            den = (__uint128_t)den * static_cast<std::uint64_t>(diff) % Q;
        }
        c[i] = (__uint128_t)num * mod_inv(static_cast<std::int64_t>(den), static_cast<std::int64_t>(Q)) % Q;
    }

    // Random degree-σ curve gamma(s) = x_alpha + Σ_{k=1..σ} a_k s^k
    std::vector<std::vector<std::uint64_t>> a(params_.w, std::vector<std::uint64_t>(params_.sigma + 1, 0));
    for (std::size_t coord = 0; coord < params_.w; ++coord) {
        for (std::size_t k = 1; k <= params_.sigma; ++k) {
            a[coord][k] = rng.random_u64(Q);
        }
    }
    auto gamma = [&](std::uint64_t sval) {
        std::vector<std::uint64_t> out(params_.w, 0);
        for (std::size_t coord = 0; coord < params_.w; ++coord) {
            std::uint64_t acc = xvec[coord];
            std::uint64_t s_pow = 1;
            for (std::size_t k = 1; k <= params_.sigma; ++k) {
                s_pow = (__uint128_t)s_pow * sval % Q;
                acc = (acc + (__uint128_t)a[coord][k] * s_pow) % Q;
            }
            out[coord] = acc;
        }
        return out;
    };

    std::uint64_t base = 1;
    for (std::size_t i = 0; i < params_.w; ++i) base *= Q;
    auto point_index = [&](std::uint64_t rho, const std::vector<std::uint64_t> &pt) -> std::uint64_t {
        std::uint64_t xi = 0;
        std::uint64_t mult = 1;
        for (std::size_t i = 0; i < params_.w; ++i) {
            xi += mult * pt[i];
            mult *= Q;
        }
        return rho * base + xi;
    };

    std::vector<std::uint64_t> delta;
    delta.reserve(q_needed);

    for (std::size_t blk = 0; blk < blocks; ++blk) {
        auto pt = gamma(s_vals[blk]);
        // choose σ+1 shares u_j summing to c[blk]
        std::vector<std::uint64_t> u(params_.sigma + 1, 0);
        std::uint64_t sum = 0;
        for (std::size_t j = 0; j + 1 < u.size(); ++j) {
            u[j] = rng.random_u64(Q);
            sum = (sum + u[j]) % Q;
        }
        u.back() = (c[blk] + Q - sum) % Q;
        for (auto rho : u) {
            delta.push_back(point_index(rho, pt));
        }
    }

    if (delta.size() > q) delta.resize(q);
    return delta;
}

} // namespace pdpf::ldc

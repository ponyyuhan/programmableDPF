// ================================================================
// File: src/ldc/reed_muller_ldc.cpp
// ================================================================

#include "pdpf/ldc/reed_muller_ldc.hpp"
#include <stdexcept>

namespace pdpf::ldc {

ReedMullerLdc::ReedMullerLdc(const LdcParams &params)
    : params_(params) {
    if (params_.L == 0) {
        throw std::invalid_argument("ReedMullerLdc: L must be > 0");
    }
}

void ReedMullerLdc::encode(const std::vector<std::int64_t> &z,
                           std::vector<std::int64_t> &codeword) const {
    codeword.assign(params_.L, 0);
    std::uint64_t limit = std::min<std::uint64_t>(params_.L, z.size());
    for (std::uint64_t i = 0; i < limit; ++i) {
        std::int64_t v = z[i];
        if (params_.p != 0) {
            v %= static_cast<std::int64_t>(params_.p);
            if (v < 0) v += static_cast<std::int64_t>(params_.p);
        }
        codeword[static_cast<std::size_t>(i)] = v;
    }
}

void ReedMullerLdc::encode_unit(std::uint64_t x,
                                std::vector<std::int64_t> &codeword) const {
    codeword.assign(params_.L, 0);
    if (params_.L == 0) return;
    std::size_t idx = static_cast<std::size_t>(x % params_.L);
    codeword[idx] = 1 % (params_.p == 0 ? 2 : params_.p);
}

std::vector<std::uint64_t> ReedMullerLdc::sample_indices(std::uint64_t alpha) const {
    std::vector<std::uint64_t> delta;
    delta.assign(params_.q == 0 ? 1 : params_.q, alpha % params_.L);
    return delta;
}

} // namespace pdpf::ldc

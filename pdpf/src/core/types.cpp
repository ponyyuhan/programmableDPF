// ================================================================
// File: src/core/types.cpp
// ================================================================

#include "pdpf/core/types.hpp"
#include <cstdlib>

namespace pdpf::core {

RandomDevice::RandomDevice() : rd_() {}

void RandomDevice::random_seed(Seed &seed) {
    // Use arc4random_buf on macOS for CSPRNG-quality bytes.
    arc4random_buf(seed.data(), seed.size());
}

std::uint64_t RandomDevice::random_u64(std::uint64_t bound) {
    if (bound == 0) {
        throw std::invalid_argument("RandomDevice::random_u64: bound = 0");
    }
    return arc4random_uniform(static_cast<uint32_t>(bound));
}

} // namespace pdpf::core

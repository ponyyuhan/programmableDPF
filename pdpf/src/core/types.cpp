// ================================================================
// File: src/core/types.cpp
// ================================================================

#include "pdpf/core/types.hpp"

namespace pdpf::core {

RandomDevice::RandomDevice() : rd_() {}

void RandomDevice::random_seed(Seed &seed) {
    for (auto &b : seed) {
        b = static_cast<std::uint8_t>(rd_() & 0xFFu);
    }
}

std::uint64_t RandomDevice::random_u64(std::uint64_t bound) {
    if (bound == 0) {
        throw std::invalid_argument("RandomDevice::random_u64: bound = 0");
    }
    std::uniform_int_distribution<std::uint64_t> dist;
    std::uint64_t limit = (std::numeric_limits<std::uint64_t>::max() / bound) * bound;
    std::uint64_t x = 0;
    do {
        x = (static_cast<std::uint64_t>(rd_()) << 32) ^ rd_();
    } while (x >= limit);
    return x % bound;
}

} // namespace pdpf::core

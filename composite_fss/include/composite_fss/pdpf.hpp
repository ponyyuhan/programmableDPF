#pragma once

#include "ring.hpp"
#include <functional>
#include <variant>
#include <vector>

namespace cfss {

struct CompareDesc {
    unsigned n_bits;
    std::vector<std::uint64_t> thresholds;
};

struct LUTDesc {
    unsigned input_bits;
    unsigned output_bits;
    std::vector<std::uint64_t> table;
};

struct MultiPointDesc {
    unsigned n_bits;
    unsigned vector_len;
    std::vector<std::uint64_t> alphas;
    std::vector<std::vector<std::uint64_t>> betas;
};

using ProgramDesc = std::variant<CompareDesc, LUTDesc, MultiPointDesc>;

struct PdpfKey {
    std::size_t id;
};

class PdpfEngine {
public:
    virtual ~PdpfEngine() = default;
    virtual std::pair<PdpfKey, PdpfKey> progGen(const ProgramDesc &desc) = 0;
    virtual std::vector<u64> eval(int party, const PdpfKey &key, u64 x) const = 0;
};

} // namespace cfss

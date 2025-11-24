#pragma once

#include "pdpf.hpp"
#include <cstdint>

namespace cfss {

struct PdpfProgramBundle {
    PdpfProgramId cmp_prog = 0;
    PdpfProgramId poly_prog = 0;
    PdpfProgramId lut_prog = 0;
    unsigned output_words = 0;
};

struct CompositeGateKey {
    int party = -1;
    std::uint64_t r_in = 0;
    std::uint64_t r_out = 0;
    PdpfProgramBundle pdpfs;
};

} // namespace cfss

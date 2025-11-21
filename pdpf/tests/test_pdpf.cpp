// ================================================================
// File: tests/test_pdpf.cpp
// ================================================================

#include "pdpf/prg/prg.hpp"
#include "pdpf/pdpf/pdpf_binary.hpp"
#include <iostream>

int main() {
    using pdpf::pdpf::PdpfBinary;
    using pdpf::prg::AesCtrPrg;
    using pdpf::prg::IPrg;
    using pdpf::core::GroupZ;
    using pdpf::core::SecurityParams;
    using pdpf::core::Seed;
    using pdpf::core::RandomDevice;

    Seed master{};
    RandomDevice rd;
    rd.random_seed(master);

    auto prg = std::make_shared<AesCtrPrg>(master);
    PdpfBinary pdpf_bin(prg);

    SecurityParams sec;
    sec.lambda_bits   = 128;
    sec.domain_size_N = 16;   // small test domain
    sec.epsilon       = 0.25; // big epsilon just for debugging

    auto k0 = pdpf_bin.gen_offline(sec);

    std::uint64_t alpha = 5;
    std::uint8_t  beta  = 1;

    auto k1 = pdpf_bin.gen_online(k0, alpha, beta);

    std::vector<GroupZ::Value> Y0, Y1;
    pdpf_bin.eval_all_offline(k0, Y0);
    pdpf_bin.eval_all_online(k1, Y1);

    std::cout << "Reconstructed f(x) = Y0[x] + Y1[x]\n";
    for (std::size_t x = 0; x < Y0.size(); ++x) {
        auto v = Y0[x] + Y1[x];
        std::cout << "x=" << x << " f(x)=" << v << "\n";
    }

    return 0;
}

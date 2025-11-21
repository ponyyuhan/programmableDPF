#include "pdpf/prg/prg.hpp"
#include "pdpf/pdpf/pdpf_binary.hpp"
#include "pdpf/pdpf/pdpf_group.hpp"
#include <iostream>
#include <vector>

using pdpf::core::GroupDescriptor;
using pdpf::core::GroupElement;
using pdpf::core::GroupZ;
using pdpf::core::RandomDevice;
using pdpf::core::SecurityParams;
using pdpf::core::Seed;

int main() {
    std::cout << "PDPF demo: binary and group examples\n";

    // Shared PRG
    Seed master{};
    RandomDevice rng;
    rng.random_seed(master);
    auto prg = std::make_shared<pdpf::prg::AesCtrPrg>(master);

    // -------- Binary PDPF demo (Theorem 4) --------
    {
        pdpf::pdpf::PdpfBinary pdpf_bin(prg);
        SecurityParams sec;
        sec.lambda_bits = 128;
        sec.domain_size_N = 16;
        sec.epsilon = 0.25;

        std::uint64_t alpha = 5;
        std::uint8_t beta = 1;

        auto k0 = pdpf_bin.gen_offline(sec);
        auto k1 = pdpf_bin.gen_online(k0, alpha, beta);

        std::vector<GroupZ::Value> Y0, Y1;
        pdpf_bin.eval_all_offline(k0, Y0);
        pdpf_bin.eval_all_online(k1, Y1);

        std::cout << "\nBinary PDPF reconstruction:\n";
        for (std::size_t x = 0; x < Y0.size(); ++x) {
            auto v = Y0[x] + Y1[x];
            std::cout << "x=" << x << " f(x)=" << v << "\n";
        }
    }

    // -------- Group PDPF demo (Theorem 5) --------
    {
        GroupDescriptor group{{7, 11}}; // Z_7 x Z_11
        pdpf::pdpf::PdpfGroup pdpf_group(prg);
        SecurityParams sec;
        sec.lambda_bits = 128;
        sec.domain_size_N = 8;
        sec.epsilon = 0.25;

        std::uint64_t alpha = 3;
        GroupElement beta = {3, 7}; // element in Z_7 x Z_11

        auto k0 = pdpf_group.gen_offline(sec, group, 0);
        auto k1 = pdpf_group.gen_online(k0, alpha, beta);

        std::vector<GroupElement> Y0, Y1;
        pdpf_group.eval_all_offline(k0, Y0);
        pdpf_group.eval_all_online(k1, Y1);

        std::cout << "\nGroup PDPF reconstruction (Z_7 x Z_11):\n";
        for (std::size_t x = 0; x < Y0.size(); ++x) {
            GroupElement f = {Y0[x][0] + Y1[x][0], Y0[x][1] + Y1[x][1]};
            // reduce modulo components
            f[0] %= 7; if (f[0] < 0) f[0] += 7;
            f[1] %= 11; if (f[1] < 0) f[1] += 11;
            std::cout << "x=" << x << " f(x)=(" << f[0] << "," << f[1] << ")\n";
        }
    }

    return 0;
}

// ================================================================
// File: src/prg/prg.cpp
// ================================================================

#include "pdpf/prg/prg.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <CommonCrypto/CommonCryptor.h>

namespace pdpf::prg {

AesCtrPrg::AesCtrPrg(const core::Seed &master_key)
    : master_key_(master_key) {
    // In a production build, initialize AES key schedule / context using master_key_.
}

void AesCtrPrg::expand(const core::Seed &seed,
                       core::Seed &left,
                       core::Seed &right) const {
    // AES-CTR using CommonCrypto, producing 32 bytes from counter=0 and counter=1.
    std::uint8_t iv[16];
    std::memcpy(iv, seed.data(), 16);

    auto encrypt_block = [&](std::uint64_t counter, std::uint8_t *out) {
        std::uint8_t ctr_block[16];
        std::memcpy(ctr_block, iv, 16);
        for (int i = 0; i < 8; ++i) {
            ctr_block[15 - i] ^= static_cast<std::uint8_t>((counter >> (8 * i)) & 0xFF);
        }
        size_t outlen = 0;
        CCCryptorStatus st = CCCrypt(kCCEncrypt,
                                     kCCAlgorithmAES128,
                                     kCCOptionECBMode,
                                     master_key_.data(),
                                     master_key_.size(),
                                     nullptr,
                                     ctr_block,
                                     sizeof(ctr_block),
                                     out,
                                     16,
                                     &outlen);
        if (st != kCCSuccess || outlen != 16) {
            throw std::runtime_error("AES-CTR encryption failed");
        }
    };

    std::uint8_t stream[32];
    encrypt_block(0, stream);
    encrypt_block(1, stream + 16);

    std::memcpy(left.data(), stream, left.size());
    std::memcpy(right.data(), stream + left.size(), right.size());
}

} // namespace pdpf::prg

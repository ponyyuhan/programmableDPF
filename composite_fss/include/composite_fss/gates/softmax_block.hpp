#pragma once

#include "../pdpf.hpp"
#include "../sharing.hpp"
#include "gelu.hpp"
#include <vector>

namespace cfss {

struct SoftmaxBlockParams {
    unsigned n_bits;
    unsigned f;
    std::size_t length;
};

struct SoftmaxBlockKey {
    // Placeholder: reuse GeLU key to demonstrate composite structure.
    GeLUKey gelu_key;
};

struct SoftmaxBlockKeyPair {
    SoftmaxBlockKey k0;
    SoftmaxBlockKey k1;
};

inline SoftmaxBlockKeyPair softmax_block_gen(const SoftmaxBlockParams &params,
                                             PdpfEngine &engine,
                                             MPCContext &ctx) {
    GeLUParams gp{params.n_bits, params.f, params.n_bits - params.f};
    auto gelu_pair = gelu_gen(gp, engine, ctx);
    SoftmaxBlockKeyPair pair;
    pair.k0 = SoftmaxBlockKey{gelu_pair.k0};
    pair.k1 = SoftmaxBlockKey{gelu_pair.k1};
    return pair;
}

// Simplified: treat softmax block as element-wise GeLU for demonstration.
inline void softmax_block_eval(int party,
                               const SoftmaxBlockKey &key,
                               const std::vector<u64> &x_hat_vec,
                               PdpfEngine &engine,
                               MPCContext &ctx,
                               std::vector<Share> &out_shares) {
    out_shares.resize(x_hat_vec.size());
    for (std::size_t i = 0; i < x_hat_vec.size(); ++i) {
        out_shares[i] = gelu_eval(party, key.gelu_key, x_hat_vec[i], engine, ctx);
    }
}

} // namespace cfss

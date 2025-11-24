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
    GeLUParams gp;
    gp.n_bits = params.n_bits;
    gp.f = params.f;
    gp.lut_bits = 8;
    gp.clip = 3.0;
    auto gelu_pair = gelu_gen(gp, engine, ctx);
    SoftmaxBlockKeyPair pair;
    pair.k0 = SoftmaxBlockKey{gelu_pair.k0};
    pair.k1 = SoftmaxBlockKey{gelu_pair.k1};
    return pair;
}

// Simplified: treat softmax block as element-wise GeLU for demonstration and
// compute both parties locally.
inline void softmax_block_eval(int party,
                               const SoftmaxBlockKeyPair &keys,
                               const std::vector<u64> &x_hat_vec,
                               PdpfEngine &engine,
                               std::vector<Share> &out_shares) {
    out_shares.resize(x_hat_vec.size());
    RingConfig cfg = make_ring_config(keys.k0.gelu_key.n_bits);
    for (std::size_t i = 0; i < x_hat_vec.size(); ++i) {
        auto r0 = gelu_eval_main(0, keys.k0.gelu_key, x_hat_vec[i], engine);
        auto r1 = gelu_eval_main(1, keys.k1.gelu_key, x_hat_vec[i], engine);
        auto out_pair = gelu_finish(cfg, keys.k0.gelu_key, keys.k1.gelu_key, r0, r1, engine);
        out_shares[i] = (party == 0) ? out_pair.first : out_pair.second;
    }
}

} // namespace cfss

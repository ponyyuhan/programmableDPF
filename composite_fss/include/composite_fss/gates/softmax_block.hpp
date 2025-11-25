#pragma once

#include "../pdpf.hpp"
#include "../sharing.hpp"
#include "../beaver.hpp"
#include "../wire.hpp"
#include "softmax.hpp"
#include <vector>
#include <random>

namespace cfss {

struct SoftmaxBlockParams {
    unsigned n_bits;
    unsigned f;
    std::size_t length;
};

using SoftmaxBlockKey = SoftmaxKey;
using SoftmaxBlockKeyPair = SoftmaxKeyPair;

inline SoftmaxBlockKeyPair softmax_block_gen(const SoftmaxBlockParams &params,
                                             PdpfEngine &engine,
                                             std::mt19937_64 &rng) {
    SoftmaxParams sp{params.n_bits, params.f, params.length};
    return softmax_keygen(sp, engine, rng);
}

inline std::pair<SoftmaxEvalShare, SoftmaxEvalShare>
softmax_block_eval_pair(PdpfEngine &engine,
                        const SoftmaxBlockKeyPair &keys,
                        const RingConfig &cfg,
                        const std::vector<MaskedWire> &x0,
                        const std::vector<MaskedWire> &x1,
                        BeaverPool &pool0,
                        BeaverPool &pool1) {
    return softmax_eval_pair(engine, keys, cfg, x0, x1, pool0, pool1);
}

} // namespace cfss

#ifndef VELRI_RUNTIME_SAMPLER_H
#define VELRI_RUNTIME_SAMPLER_H

#include "velri/velri.h"

/*
 * Zero-Allocation Deterministic Token Sampler (`src/runtime/sampler.{h,c}`).
 *
 * Allocates its candidate scratch buffer `[vocab_size]` once in `velri_sampler_init`.
 * Subsequent `velri_sampler_sample` calls perform 0 B heap allocation, using an in-place
 * iterative heapsort for deterministic tie-breaking across all platforms.
 */

typedef struct {
    double value;
    uint32_t index;
} velri_sampler_cand;

typedef struct {
    uint64_t initial_seed;
    uint64_t rng_state;
    velri_sampler_cand *scratch;
    uint32_t vocab_size;
} velri_sampler;

velri_rc velri_sampler_init(velri_sampler *s, uint32_t vocab_size, uint64_t seed);
void velri_sampler_destroy(velri_sampler *s);
void velri_sampler_reset(velri_sampler *s);
void velri_sampler_reseed(velri_sampler *s, uint64_t seed);

velri_rc velri_sampler_validate_params(const velri_sampling_params *params);

/* Sample one token from `logits[0..s->vocab_size - 1]` with 0 B heap allocation. */
velri_rc velri_sampler_sample(velri_sampler *s, const float *logits,
                              const velri_sampling_params *params, int32_t *out_token);

#endif /* VELRI_RUNTIME_SAMPLER_H */

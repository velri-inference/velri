#include "runtime/sampler.h"
#include "config/sampling_limits.h"

#include <math.h>
#include <stdlib.h>

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

static double rng_uniform_f64(uint64_t *state) {
    /* 53 bits of precision uniformly in [0.0, 1.0). */
    uint64_t bits = splitmix64_next(state) >> 11U;
    return (double)bits * (1.0 / 9007199254740992.0);
}

/* Strict weak order: returns true when `a` should rank BEFORE `b` (higher value, lower index). */
static inline bool cand_better(velri_sampler_cand a, velri_sampler_cand b) {
    if (a.value != b.value) {
        return a.value > b.value;
    }
    return a.index < b.index;
}

/* Iterative in-place min-heap sift-down (worst element at root) for descending heapsort. */
static void heap_sift_down_worst(velri_sampler_cand *arr, uint32_t start, uint32_t end) {
    uint32_t root = start;
    for (;;) {
        uint32_t child = root * 2U + 1U;
        if (child >= end) {
            break;
        }
        uint32_t worst = root;
        if (cand_better(arr[worst], arr[child])) {
            worst = child;
        }
        if (child + 1U < end && cand_better(arr[worst], arr[child + 1U])) {
            worst = child + 1U;
        }
        if (worst == root) {
            break;
        }
        velri_sampler_cand tmp = arr[root];
        arr[root] = arr[worst];
        arr[worst] = tmp;
        root = worst;
    }
}

/* Sort `arr[0..n-1]` descending by `value`, tie-breaking ascending by `index` (0 allocs). */
static void heapsort_desc(velri_sampler_cand *arr, uint32_t n) {
    if (n <= 1U) {
        return;
    }
    for (uint32_t i = n / 2U; i > 0U; --i) {
        heap_sift_down_worst(arr, i - 1U, n);
    }
    for (uint32_t end = n; end > 1U; --end) {
        velri_sampler_cand tmp = arr[0];
        arr[0] = arr[end - 1U];
        arr[end - 1U] = tmp;
        heap_sift_down_worst(arr, 0U, end - 1U);
    }
}

velri_rc velri_sampler_init(velri_sampler *s, uint32_t vocab_size, uint64_t seed) {
    if (!s || vocab_size == 0U) {
        return VELRI_RC_ERR_INVALID;
    }
    size_t n_vocab = vocab_size;
    if (n_vocab > SIZE_MAX / sizeof(velri_sampler_cand)) {
        return VELRI_RC_ERR_OVERFLOW;
    }
    velri_sampler_cand *buf = (velri_sampler_cand *)malloc(n_vocab * sizeof(velri_sampler_cand));
    if (!buf) {
        return VELRI_RC_ERR_NOMEM;
    }
    s->initial_seed = seed;
    s->rng_state = seed;
    s->scratch = buf;
    s->vocab_size = vocab_size;
    return VELRI_RC_OK;
}

void velri_sampler_destroy(velri_sampler *s) {
    if (!s) {
        return;
    }
    free(s->scratch);
    s->scratch = nullptr;
    s->vocab_size = 0U;
}

void velri_sampler_reset(velri_sampler *s) {
    if (!s) {
        return;
    }
    s->rng_state = s->initial_seed;
}

void velri_sampler_reseed(velri_sampler *s, uint64_t seed) {
    if (!s) {
        return;
    }
    s->rng_state = seed;
}

velri_rc velri_sampler_validate_params(const velri_sampling_params *params) {
    if (!params) {
        return VELRI_RC_ERR_INVALID;
    }
    if (!velri_sampling_valid_temperature(params->temperature)) {
        return VELRI_RC_ERR_INVALID;
    }
    if (!velri_sampling_valid_top_p(params->top_p)) {
        return VELRI_RC_ERR_INVALID;
    }
    if (!velri_sampling_valid_probability(params->min_p)) {
        return VELRI_RC_ERR_INVALID;
    }
    return VELRI_RC_OK;
}

/* Independent lane maxima vectorize without relaxing floating-point semantics.
 * A second scan returns the first matching index, including signed-zero ties. */
static uint32_t first_maximum(const float *logits, uint32_t count) {
    enum { width = 8 };
    float peaks[width];
    for (uint32_t j = 0U; j < width; ++j) peaks[j] = -INFINITY;

    bool valid = true;
    uint32_t i = 0U;
    for (; count - i >= width; i += width) {
        for (uint32_t j = 0U; j < width; ++j) {
            float v = logits[i + j];
            valid &= v < INFINITY;
            if (v > peaks[j]) peaks[j] = v;
        }
    }
    float peak = -INFINITY;
    for (uint32_t j = 0U; j < width; ++j) {
        if (peaks[j] > peak) peak = peaks[j];
    }
    for (; i < count; ++i) {
        float v = logits[i];
        valid &= v < INFINITY;
        if (v > peak) peak = v;
    }
    if (!valid || peak == -INFINITY) return UINT32_MAX;

    i = 0U;
    for (; count - i >= width; i += width) {
        bool found = false;
        for (uint32_t j = 0U; j < width; ++j) found |= logits[i + j] == peak;
        if (found) break;
    }
    while (logits[i] != peak) ++i;
    return i;
}

velri_rc velri_sampler_sample(velri_sampler *s, const float *logits,
                              const velri_sampling_params *params, int32_t *out_token) {
    if (!s || !s->scratch || s->vocab_size == 0U || !logits || !params || !out_token) {
        return VELRI_RC_ERR_INVALID;
    }
    if (s->vocab_size > (uint32_t)INT32_MAX) {
        return VELRI_RC_ERR_OVERFLOW;
    }
    velri_rc vrc = velri_sampler_validate_params(params);
    if (vrc != VELRI_RC_OK) {
        return vrc;
    }

    uint32_t best_idx = first_maximum(logits, s->vocab_size);
    if (best_idx == UINT32_MAX) {
        return VELRI_RC_ERR_EXEC;
    }
    float max_logit = logits[best_idx];

    if (params->reseed) {
        s->rng_state = params->seed;
    }

    /* Fast deterministic greedy path (`temperature == 0.0f`). */
    if (params->temperature == 0.0f) {
        *out_token = (int32_t)best_idx;
        return VELRI_RC_OK;
    }

    /* Pass 2: collect finite candidates and apply exact `min_p` threshold in FP64. */
    double inv_t = 1.0 / (double)params->temperature;
    double min_p = (double)params->min_p;
    bool use_min_p = (min_p > 0.0);
    uint32_t n = 0U;

    for (uint32_t i = 0U; i < s->vocab_size; ++i) {
        float v = logits[i];
        if (!isfinite(v)) {
            continue;
        }
        if (use_min_p && v < max_logit) {
            double rel_p = exp(((double)v - (double)max_logit) * inv_t);
            if (rel_p < min_p) {
                continue;
            }
        }
        s->scratch[n++] = (velri_sampler_cand){.value = (double)v, .index = i};
    }

    if (n == 0U) {
        *out_token = (int32_t)best_idx;
        return VELRI_RC_OK;
    }

    bool use_top_k = (params->top_k > 0U && params->top_k < n);
    bool use_top_p = (params->top_p > 0.0f && params->top_p < 1.0f);

    if (use_top_k) {
        /* O(n log k) min-heap selection of top-k candidates (also feeds top_p when both active). */
        uint32_t k = params->top_k;
        for (uint32_t i = k / 2U; i > 0U; --i) {
            heap_sift_down_worst(s->scratch, i - 1U, k);
        }
        for (uint32_t i = k; i < n; ++i) {
            if (cand_better(s->scratch[i], s->scratch[0])) {
                s->scratch[0] = s->scratch[i];
                heap_sift_down_worst(s->scratch, 0U, k);
            }
        }
        heapsort_desc(s->scratch, k);
        n = k;
    } else if (use_top_p) {
        heapsort_desc(s->scratch, n);
    }

    /* Pass 3: compute unnormalized softmax weights directly in FP64 (zero FP32 rounding drift). */
    double total_w = 0.0;
    for (uint32_t i = 0U; i < n; ++i) {
        double w = exp((s->scratch[i].value - (double)max_logit) * inv_t);
        s->scratch[i].value = w;
        total_w += w;
    }
    if (!(total_w > 0.0) || !isfinite(total_w)) {
        *out_token = (int32_t)best_idx;
        return VELRI_RC_OK;
    }

    /* Pass 4: optional `top_p` nucleus cutoff on descending-sorted FP64 weights. */
    uint32_t keep = n;
    if (use_top_p) {
        double cutoff = (double)params->top_p * total_w;
        double cum = 0.0;
        double kept_w = 0.0;
        keep = 0U;
        for (uint32_t i = 0U; i < n; ++i) {
            double w = s->scratch[i].value;
            if (w <= 0.0 && keep > 0U) {
                break;
            }
            cum += w;
            kept_w += w;
            keep = i + 1U;
            if (cum >= cutoff) {
                break;
            }
        }
        total_w = kept_w;
    }

    /* Pass 5: draw uniform sample in [0, total_w); fallback is strictly last nonzero weight. */
    double target = rng_uniform_f64(&s->rng_state) * total_w;
    double acc = 0.0;
    uint32_t chosen = UINT32_MAX;
    uint32_t last_nonzero = 0U;
    for (uint32_t i = 0U; i < keep; ++i) {
        double w = s->scratch[i].value;
        if (w > 0.0) {
            last_nonzero = i;
        }
        acc += w;
        if (acc > target) {
            chosen = i;
            break;
        }
    }
    if (chosen == UINT32_MAX) {
        chosen = last_nonzero;
    }
    *out_token = (int32_t)s->scratch[chosen].index;
    return VELRI_RC_OK;
}

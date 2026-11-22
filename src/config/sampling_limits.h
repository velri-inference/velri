#ifndef VELRI_CONFIG_SAMPLING_LIMITS_H
#define VELRI_CONFIG_SAMPLING_LIMITS_H

#include <float.h>

static inline bool velri_sampling_valid_temperature(double value) {
    return value >= 0.0 && value <= FLT_MAX;
}

static inline bool velri_sampling_valid_probability(double value) {
    return value >= 0.0 && value <= 1.0;
}

/* Explicit zero disables top-p. A positive cutoff must survive conversion to F32. */
static inline bool velri_sampling_valid_top_p(double value) {
    return velri_sampling_valid_probability(value) && (value == 0.0 || (float)value > 0.0f);
}

#endif

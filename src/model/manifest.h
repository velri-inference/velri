#ifndef VELRI_MODEL_MANIFEST_H
#define VELRI_MODEL_MANIFEST_H

#include <velri_lib/blocks.h>

/* The library shares WEIGHT and STATE by declaration order within each device.
 * Check that those physical positions have the same logical roles in both graphs. */
bool velri_model_check_shared_manifest(const velri_manifest *owner, const velri_manifest *peer,
                                       uint32_t num_devices, char *err, size_t err_cap);

#endif

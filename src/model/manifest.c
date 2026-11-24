#include "model/manifest.h"

#include <stdio.h>
#include <string.h>

static bool entries_match(const velri_manifest_entry *a, const velri_manifest_entry *b,
                          bool is_weight) {
    bool same_binding = is_weight
        ? strcmp(velri_manifest_weight_source(a), velri_manifest_weight_source(b)) == 0
        : strcmp(a->name, b->name) == 0 && strcmp(a->source_name, b->source_name) == 0;
    return same_binding && a->dtype == b->dtype && a->rank == b->rank && a->rank <= VELRI_MAX_RANK &&
           memcmp(a->dims, b->dims, a->rank * sizeof(a->dims[0])) == 0;
}

static bool check_entries(const velri_manifest_entry *owner, uint32_t owner_count,
                          const velri_manifest_entry *peer, uint32_t peer_count,
                          uint32_t num_devices, bool is_weight, char *err, size_t err_cap) {
    const char *kind = is_weight ? "weight" : "state";
    if (owner_count != peer_count) {
        snprintf(err, err_cap, "shared %s count differs (decode=%u, prefill=%u)", kind, owner_count,
                 peer_count);
        return false;
    }
    uint32_t matched = 0U;
    for (uint32_t dev = 0U; dev < num_devices; ++dev) {
        uint32_t j = 0U;
        for (uint32_t i = 0U; i < owner_count; ++i) {
            if (owner[i].device_id != dev) {
                continue;
            }
            while (j < peer_count && peer[j].device_id != dev) {
                ++j;
            }
            if (j == peer_count || !entries_match(&owner[i], &peer[j], is_weight)) {
                snprintf(err, err_cap,
                         "shared %s mismatch on device %u (decode='%s', prefill='%s')", kind, dev,
                         owner[i].name, j < peer_count ? peer[j].name : "<missing>");
                return false;
            }
            ++j;
            ++matched;
        }
    }
    if (matched != owner_count) {
        snprintf(err, err_cap, "shared %s has an unknown device", kind);
        return false;
    }
    return true;
}

bool velri_model_check_shared_manifest(const velri_manifest *owner, const velri_manifest *peer,
                                       uint32_t num_devices, char *err, size_t err_cap) {
    return check_entries(owner->weights, owner->num_weights, peer->weights, peer->num_weights,
                         num_devices, true, err, err_cap) &&
           check_entries(owner->states, owner->num_states, peer->states, peer->num_states,
                         num_devices, false, err, err_cap);
}

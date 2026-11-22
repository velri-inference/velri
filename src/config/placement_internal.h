#ifndef VELRI_CONFIG_PLACEMENT_INTERNAL_H
#define VELRI_CONFIG_PLACEMENT_INTERNAL_H

#include "config/placement.h"
#include "config/placement_scopes.h"

typedef struct {
    const char *name;
    int value;
} velri_cfg_enum_entry;

typedef struct {
    const char *scope;
    const char *suffix;
    bool always_show;
} velri_cfg_layer_scope_meta;

typedef struct {
    char *buf;
    size_t cap;
    size_t pos;
} velri_cfg_str_buf;

enum { VELRI_CFG_MAX_LAYER_SCOPES = 16 };

extern const velri_cfg_enum_entry velri_cfg_affinity_table[];
extern const size_t velri_cfg_affinity_table_len;

extern const velri_cfg_layer_scope_meta velri_cfg_layer_scopes[];
extern const size_t velri_cfg_layer_scopes_len;

void velri_cfg_set_errf(char *err_msg, size_t err_cap, const char *fmt, ...);
void velri_cfg_join_label4(char *dst, size_t cap, const char *a, const char *b, const char *c,
                           const char *d);
bool velri_cfg_selector_matches_layer(const velri_layer_selector *sel, uint32_t layer);

const char *velri_cfg_enum_name(const velri_cfg_enum_entry *table, size_t count, int value,
                                const char *fallback);
void velri_cfg_sb_appendf(velri_cfg_str_buf *sb, const char *fmt, ...);

#endif /* VELRI_CONFIG_PLACEMENT_INTERNAL_H */

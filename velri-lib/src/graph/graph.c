#include "core_internal.h"
#include <string.h>

velri_status velri_graph_create(velri_graph **out_graph) {
    if (!out_graph) {
        return VELRI_ERR_INVALID;
    }
    *out_graph = nullptr;
    velri_graph *g = (velri_graph *)calloc(1U, sizeof(*g));
    if (!g) {
        return VELRI_ERR_NOMEM;
    }
    *out_graph = g;
    return VELRI_OK;
}

void velri_graph_destroy(velri_graph *graph) {
    if (!graph) {
        return;
    }
    free(graph->storages);
    free(graph->views);
    free(graph->nodes);
    free(graph->kv_bindings);
    free(graph);
}

static bool grow_array(void **ptr, uint32_t *cap, size_t elem_size) {
    uint32_t next = (*cap == 0U) ? 16U : (*cap * 2U);
    void *p = realloc(*ptr, (size_t)next * elem_size);
    if (!p) {
        return false;
    }
    *ptr = p;
    *cap = next;
    return true;
}

velri_status velri_graph_add_storage(
    velri_graph *graph,
    const velri_storage_desc *desc,
    velri_storage_id *out_id) {
    if (!graph || !desc || !out_id || desc->byte_size == 0U) {
        return VELRI_ERR_INVALID;
    }
    if (velri_dtype_bytes(desc->dtype) == 0U) {
        return VELRI_ERR_INVALID;
    }
    if (desc->align != 0U && (desc->align & (desc->align - 1U)) != 0U) {
        return VELRI_ERR_INVALID;
    }
    if (graph->num_storages == graph->cap_storages) {
        if (!grow_array((void **)&graph->storages, &graph->cap_storages,
                        sizeof(velri_graph_storage))) {
            return VELRI_ERR_NOMEM;
        }
    }

    uint32_t id = graph->num_storages;
    velri_graph_storage *gs = &graph->storages[id];
    memset(gs, 0, sizeof(*gs));
    gs->desc = *desc;
    if (gs->desc.align < 64U) {
        gs->desc.align = 64U;
    }
    if (desc->debug_name) {
        size_t len = strlen(desc->debug_name);
        if (len >= sizeof(gs->debug_name)) {
            len = sizeof(gs->debug_name) - 1U;
        }
        memcpy(gs->debug_name, desc->debug_name, len);
        gs->debug_name[len] = '\0';
        gs->desc.debug_name = gs->debug_name;
    } else {
        gs->desc.debug_name = nullptr;
    }

    graph->num_storages++;
    *out_id = id;
    return VELRI_OK;
}

velri_status velri_graph_add_view(
    velri_graph *graph,
    const velri_tensor_view *view,
    velri_view_id *out_id) {
    if (!graph || !view || !out_id) {
        return VELRI_ERR_INVALID;
    }
    if (view->storage_id >= graph->num_storages) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = velri_tensor_view_validate(view, &graph->storages[view->storage_id].desc);
    if (st != VELRI_OK) {
        return st;
    }
    if (graph->num_views == graph->cap_views) {
        if (!grow_array((void **)&graph->views, &graph->cap_views, sizeof(velri_tensor_view))) {
            return VELRI_ERR_NOMEM;
        }
    }
    uint32_t id = graph->num_views++;
    graph->views[id] = *view;
    *out_id = id;
    return VELRI_OK;
}

velri_status velri_graph_add_kv_binding(velri_graph *graph, const velri_kv_binding *binding) {
    if (!graph || !binding) {
        return VELRI_ERR_INVALID;
    }
    if (binding->k_view >= graph->num_views || binding->v_view >= graph->num_views) {
        return VELRI_ERR_INVALID;
    }
    const velri_tensor_view *kv = &graph->views[binding->k_view];
    const velri_tensor_view *vv = &graph->views[binding->v_view];
    if (binding->kind == VELRI_KV_SHARED && kv->storage_id != vv->storage_id) {
        return VELRI_ERR_INVALID;
    }
    if (binding->kind == VELRI_KV_DERIVED) {
        if (binding->derive_node >= graph->num_nodes) {
            return VELRI_ERR_INVALID;
        }
    }
    if (graph->num_kv_bindings == graph->cap_kv_bindings) {
        if (!grow_array((void **)&graph->kv_bindings, &graph->cap_kv_bindings,
                        sizeof(velri_kv_binding))) {
            return VELRI_ERR_NOMEM;
        }
    }
    graph->kv_bindings[graph->num_kv_bindings++] = *binding;
    return VELRI_OK;
}

velri_status velri_graph_add_op(
    velri_graph *graph,
    const velri_op_desc *op,
    velri_device_id device_id,
    velri_node_id *out_node_id) {
    if (!graph || !op) {
        return VELRI_ERR_INVALID;
    }
    if (op->num_inputs > VELRI_OP_MAX_INPUTS || op->num_outputs == 0U ||
        op->num_outputs > VELRI_OP_MAX_OUTPUTS) {
        return VELRI_ERR_INVALID;
    }

    velri_resolved_op resolved;
    memset(&resolved, 0, sizeof(resolved));
    resolved.kind = op->kind;
    resolved.num_inputs = op->num_inputs;
    resolved.num_outputs = op->num_outputs;
    resolved.raw = *op;

    for (uint32_t i = 0U; i < op->num_inputs; ++i) {
        velri_view_id vid = op->inputs[i];
        if (vid == VELRI_INVALID_ID) {
            resolved.input_present[i] = false;
        } else {
            if (vid >= graph->num_views) {
                return VELRI_ERR_INVALID;
            }
            resolved.input_present[i] = true;
            resolved.inputs[i] = graph->views[vid];
        }
    }
    for (uint32_t i = 0U; i < op->num_outputs; ++i) {
        velri_view_id vid = op->outputs[i];
        if (vid >= graph->num_views) {
            return VELRI_ERR_INVALID;
        }
        /* Immutable weight storages can never be written by an op node. */
        velri_storage_id sid = graph->views[vid].storage_id;
        if (graph->storages[sid].desc.kind == VELRI_STORAGE_WEIGHT) {
            return VELRI_ERR_INVALID;
        }
        resolved.outputs[i] = graph->views[vid];
    }

    velri_status st = velri_op_validate(&resolved);
    if (st != VELRI_OK) {
        return st;
    }

    if (graph->num_nodes == graph->cap_nodes) {
        if (!grow_array((void **)&graph->nodes, &graph->cap_nodes, sizeof(velri_graph_node))) {
            return VELRI_ERR_NOMEM;
        }
    }
    uint32_t nid = graph->num_nodes++;
    graph->nodes[nid].desc = *op;
    graph->nodes[nid].device_id = device_id;
    if (out_node_id) {
        *out_node_id = nid;
    }
    return VELRI_OK;
}

uint32_t velri_graph_storage_count(const velri_graph *graph) {
    return graph ? graph->num_storages : 0U;
}

uint32_t velri_graph_view_count(const velri_graph *graph) {
    return graph ? graph->num_views : 0U;
}

uint32_t velri_graph_node_count(const velri_graph *graph) {
    return graph ? graph->num_nodes : 0U;
}

uint32_t velri_graph_kv_binding_count(const velri_graph *graph) {
    return graph ? graph->num_kv_bindings : 0U;
}

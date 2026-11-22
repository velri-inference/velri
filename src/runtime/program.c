#include "runtime/runtime_internal.h"
#include "format/inspect.h"
#include "model/manifest.h"

#include <string.h>

static_assert((uint32_t)VELRI_PROGRAM_MAX_DEVICES == (uint32_t)VELRI_MAX_REPORT_DEVICES,
              "VELRI_PROGRAM_MAX_DEVICES must match VELRI_MAX_REPORT_DEVICES");

/* ---- Config Validation & Limits Resolution Helpers ---- */

static velri_device_id select_default_device(const velri_ctx *ctx) {
    velri_device_id dev = 0U;
    if (ctx->vulkan_requested &&
        velri_context_find_device(ctx->vl_ctx, "vulkan", 0U, &dev) == VELRI_OK) {
        return dev;
    }
    if (velri_context_find_device(ctx->vl_ctx, "cpu", 0U, &dev) == VELRI_OK) {
        return dev;
    }
    return 0U;
}

static velri_rc program_parse_and_validate_config(velri_program *prog,
                                                  const velri_program_options *opts) {
    velri_ctx *ctx = prog->ctx;
    char err_buf[VELRI_CFG_ERR_MAX] = {0};

    if (opts && opts->placement_json && opts->placement_json_len > 0U) {
        velri_status st = velri_engine_config_parse(opts->placement_json, opts->placement_json_len,
                                                    &prog->cfg, err_buf, sizeof(err_buf));
        if (st != VELRI_OK) {
            velri_rt_ctx_set_error(ctx, "program_prepare: %s",
                                   err_buf[0] ? err_buf : "invalid placement JSON");
            return velri_rt_rc_from_vl(st);
        }
    }

    bool custom_kv = prog->cfg.kv_k_bits != 0U || prog->cfg.kv_v_bits != 0U;
    if (custom_kv && (prog->cfg.kv_k_bits != 16U || prog->cfg.kv_v_bits != 16U)) {
        velri_rt_ctx_set_error(
            ctx, "program_prepare: kv-quantization requires k-bits=16 and v-bits=16 (got %u, %u)",
            prog->cfg.kv_k_bits, prog->cfg.kv_v_bits);
        return VELRI_RC_ERR_UNSUPPORTED;
    }
    velri_rc cpu_rc = velri_rt_check_cpu_policy(&prog->cfg.cpu_backend, &ctx->cpu_policy,
                                                err_buf, sizeof(err_buf));
    if (cpu_rc != VELRI_RC_OK) {
        velri_rt_ctx_set_error(ctx, "program_prepare: %s", err_buf);
        return cpu_rc;
    }

    bool strict_layers = opts ? opts->strict_layer_selectors : false;
    velri_status st = velri_engine_config_validate_layers(
        &prog->cfg, prog->model->desc.num_hidden_layers, strict_layers, err_buf, sizeof(err_buf));
    if (st != VELRI_OK) {
        velri_rt_ctx_set_error(ctx, "program_prepare: %s",
                               err_buf[0] ? err_buf : "layer selector validation failed");
        return velri_rt_rc_from_vl(st);
    }

    st = velri_placement_plan_bind(&prog->cfg, ctx->vl_ctx, select_default_device(ctx),
                                   &prog->placement, err_buf, sizeof(err_buf));
    if (st != VELRI_OK) {
        velri_rt_ctx_set_error(ctx, "program_prepare: %s",
                               err_buf[0] ? err_buf : "placement bind failed");
        return velri_rt_rc_from_vl(st);
    }
    return VELRI_RC_OK;
}

static velri_rc program_resolve_limits(velri_program *prog, const velri_program_options *opts) {
    const velri_model_desc *d = &prog->model->desc;
    const velri_arch_ops *ops = prog->model->arch_ops;

    uint32_t max_ctx = d->max_position_embeddings;
    if (opts && opts->max_ctx > 0U) {
        max_ctx = opts->max_ctx;
    } else if (prog->cfg.context_size > 0U) {
        max_ctx = prog->cfg.context_size;
    }
    if (max_ctx == 0U || (ops->enforces_max_pos_ctx && max_ctx > d->max_position_embeddings)) {
        velri_rt_ctx_set_error(prog->ctx, "program_prepare: invalid max_ctx (%u) for model '%s'",
                               max_ctx, ops->name);
        return VELRI_RC_ERR_INVALID;
    }

    uint32_t chunk = 64U;
    if (opts && opts->prefill_chunk_size > 0U) {
        chunk = opts->prefill_chunk_size;
    } else if (prog->cfg.vulkan_backend.prefill_chunk_size > 0U && prog->ctx->vulkan_requested) {
        chunk = prog->cfg.vulkan_backend.prefill_chunk_size;
        if (prog->cfg.max_seq_len > 0U && chunk > prog->cfg.max_seq_len) {
            chunk = prog->cfg.max_seq_len;
        }
    } else if (prog->cfg.max_seq_len > 0U) {
        chunk = prog->cfg.max_seq_len;
    }
    if (chunk > max_ctx) {
        chunk = max_ctx;
    }
    if (chunk == 0U) {
        chunk = 1U;
    }

    prog->max_ctx = max_ctx;
    prog->prefill_chunk_size = chunk;
    prog->default_seed = prog->cfg.sampling.seed;
    prog->has_default_seed = prog->cfg.sampling.has_seed;
    return VELRI_RC_OK;
}

/* ---- Graph Building & Plan Compilation ---- */

static velri_rc build_program_graph(velri_program *prog, bool is_prefill, uint32_t max_seq_len,
                                    bool latent_kv, velri_manifest *out_manifest,
                                    velri_graph **out_graph, bool *out_distinct_compute) {
    velri_graph *g = nullptr;
    velri_status st = velri_graph_create(&g);
    if (st != VELRI_OK) {
        velri_rt_ctx_set_error(prog->ctx, "program_prepare: graph creation failed (%s)",
                               velri_rc_str(velri_rt_rc_from_vl(st)));
        return velri_rt_rc_from_vl(st);
    }

    velri_placement_builder_adapter ad = {.plan = &prog->placement, .is_prefill = is_prefill};
    velri_builder b;
    velri_b_init_checked(&b, g, out_manifest, prog->placement.default_store_dev,
                         prog->placement.default_compute_dev, velri_placement_builder_cb, &ad);
    b.allow_f16_attention_matmul = prog->cfg.vulkan_backend.prefill_attention_bits == 16U;
    b.allow_f16_matmul = prog->cfg.vulkan_backend.prefill_matmul_bits == 16U;

    const velri_model *m = prog->model;
    const velri_model_desc *d = &m->desc;
    const velri_model_params mp = {
        .vocab_size = d->vocab_size,
        .max_position_embeddings = d->max_position_embeddings,
        .hidden_size = d->hidden_size,
        .num_hidden_layers = d->num_hidden_layers,
        .num_attention_heads = d->num_attention_heads,
        .num_key_value_heads = d->num_key_value_heads,
        .head_dim = d->head_dim,
        .intermediate_size = d->intermediate_size,
        .sliding_window = d->sliding_window,
        .norm_eps = d->norm_eps,
        .hidden_act = m->vl_act,
        .tie_word_embeddings = d->tie_word_embeddings,
        .dtype = m->vl_dtype,
        .rope_theta = d->rope_theta,
        .rotary_dim = d->rotary_dim,
        .rope_mode = m->vl_rope,
        .qkv_bias = d->qkv_bias,
        .max_seq_len = max_seq_len,
        .max_ctx = prog->max_ctx,
        .latent_kv = latent_kv,
        .f16_kv = prog->cfg.kv_k_bits == 16U,
    };

    const char *phase = is_prefill ? "prefill" : "decode";
    st = m->arch_ops->build_graph(&b, &mp);
    if (st != VELRI_OK) {
        if (ad.err_msg[0] != '\0') {
            velri_rt_ctx_set_error(prog->ctx, "program_prepare: %s", ad.err_msg);
        } else {
            velri_rt_ctx_set_error(prog->ctx, "program_prepare: failed building %s graph (%s)",
                                   phase, velri_rc_str(velri_rt_rc_from_vl(st)));
        }
        velri_graph_destroy(g);
        return velri_rt_rc_from_vl(st);
    }
    if (out_manifest->token_ids_in == VELRI_INVALID_ID ||
        out_manifest->logits_out == VELRI_INVALID_ID) {
        velri_rt_ctx_set_error(prog->ctx,
                               "program_prepare: graph missing token_ids input or logits output");
        velri_graph_destroy(g);
        return VELRI_RC_ERR_INVALID;
    }

    *out_graph = g;
    if (out_distinct_compute) {
        *out_distinct_compute = ad.distinct_compute;
    }
    return VELRI_RC_OK;
}

static velri_rc compile_program_plan(velri_program *prog, velri_graph *g, const char *phase,
                                     bool allow_cpu_fb, velri_plan *share_weights_from,
                                     velri_plan **out_plan) {
    velri_plan_options popts = {
        .allow_cpu_fallback = allow_cpu_fb,
        .share_weights_from = share_weights_from,
    };
    velri_status st = velri_plan_prepare(prog->ctx->vl_ctx, g, &popts, out_plan);
    if (st != VELRI_OK) {
        velri_rt_ctx_set_error(prog->ctx, "program_prepare: %s plan preparation failed (%s)", phase,
                               velri_rc_str(velri_rt_rc_from_vl(st)));
        return velri_rt_rc_from_vl(st);
    }
    return VELRI_RC_OK;
}

static velri_rc prepare_program_plans(velri_program *prog, bool latent_kv, bool allow_cpu_fb) {
    /* Build the chunk-sized graph once. If every requested compute device is also the decode
     * device, this same graph serves both phases. A single-token context only uses decode. */
    bool can_prefill = prog->max_ctx > 1U;
    bool distinct_compute = false;
    velri_graph *chunk_graph = nullptr;
    velri_rc rc = build_program_graph(prog, can_prefill, prog->prefill_chunk_size, latent_kv,
                                      &prog->decode_manifest, &chunk_graph, &distinct_compute);
    if (rc != VELRI_RC_OK) {
        return rc;
    }
    prog->has_distinct_prefill = can_prefill && distinct_compute;
    if (prog->has_distinct_prefill) {
        prog->prefill_manifest = prog->decode_manifest;
        velri_manifest_init(&prog->decode_manifest);
        velri_graph *decode_graph = nullptr;
        rc = build_program_graph(prog, false, 1U, latent_kv, &prog->decode_manifest,
                                 &decode_graph, nullptr);
        char err[VELRI_CFG_ERR_MAX];
        if (rc == VELRI_RC_OK &&
            !velri_model_check_shared_manifest(&prog->decode_manifest, &prog->prefill_manifest,
                                               velri_context_device_count(prog->ctx->vl_ctx),
                                               err, sizeof(err))) {
            velri_rt_ctx_set_error(prog->ctx, "program_prepare: %s", err);
            rc = VELRI_RC_ERR_INVALID;
        }
        if (rc == VELRI_RC_OK) {
            rc = compile_program_plan(prog, decode_graph, "decode", allow_cpu_fb, nullptr,
                                      &prog->decode_plan);
        }
        velri_graph_destroy(decode_graph);
        if (rc == VELRI_RC_OK) {
            rc = compile_program_plan(prog, chunk_graph, "prefill", allow_cpu_fb, prog->decode_plan,
                                      &prog->prefill_plan);
        }
    } else {
        rc = compile_program_plan(prog, chunk_graph, "decode", allow_cpu_fb, nullptr,
                                  &prog->decode_plan);
    }
    velri_graph_destroy(chunk_graph);
    return rc;
}

/* ---- Weight Binding & Auto-RoPE Synthesis ---- */

static velri_rc ensure_auto_rope(velri_program *prog) {
    if (prog->auto_rope_cos && prog->auto_rope_sin) {
        return VELRI_RC_OK;
    }
    uint32_t half = prog->model->desc.rotary_dim / 2U;
    if (half == 0U || (size_t)prog->max_ctx > SIZE_MAX / (size_t)half) {
        velri_rt_ctx_set_error(prog->ctx, "program_prepare: RoPE table size overflow");
        return VELRI_RC_ERR_OVERFLOW;
    }
    size_t elems = (size_t)prog->max_ctx * (size_t)half;
    if (elems > SIZE_MAX / sizeof(float)) {
        velri_rt_ctx_set_error(prog->ctx, "program_prepare: RoPE table byte size overflow");
        return VELRI_RC_ERR_OVERFLOW;
    }
    size_t bytes = elems * sizeof(float);
    float *c = (float *)velri_rt_aligned_alloc(64U, bytes);
    float *s = (float *)velri_rt_aligned_alloc(64U, bytes);
    if (!c || !s) {
        velri_rt_aligned_free(c);
        velri_rt_aligned_free(s);
        velri_rt_ctx_set_error(prog->ctx,
                               "program_prepare: out of memory allocating auto RoPE tables");
        return VELRI_RC_ERR_NOMEM;
    }
    velri_status st = velri_fill_rope_tables(c, s, prog->max_ctx, prog->model->desc.rotary_dim,
                                             prog->model->desc.rope_theta);
    if (st != VELRI_OK) {
        velri_rt_aligned_free(c);
        velri_rt_aligned_free(s);
        velri_rt_ctx_set_error(prog->ctx, "program_prepare: failed generating RoPE tables");
        return velri_rt_rc_from_vl(st);
    }
    prog->auto_rope_cos = c;
    prog->auto_rope_sin = s;
    prog->auto_rope_bytes = bytes;
    return VELRI_RC_OK;
}

static velri_rc bind_program_weights(velri_program *prog) {
    bool has_auto_rope = prog->model->arch_ops->has_auto_rope;
    if (has_auto_rope) {
        bool has_cos = (velri_rt_model_find_weight(prog->model, VELRI_WEIGHT_ROPE_COS) != nullptr);
        bool has_sin = (velri_rt_model_find_weight(prog->model, VELRI_WEIGHT_ROPE_SIN) != nullptr);
        if (has_cos != has_sin) {
            velri_rt_ctx_set_error(
                prog->ctx,
                "program_prepare: must bind both '%s' and '%s' (or omit both for auto-generation)",
                VELRI_WEIGHT_ROPE_COS, VELRI_WEIGHT_ROPE_SIN);
            return VELRI_RC_ERR_INVALID;
        }
    }

    bool auto_rope_on_host = false;
    const velri_manifest *m = &prog->decode_manifest;
    for (uint32_t i = 0U; i < m->num_weights; ++i) {
        const velri_manifest_entry *e = &m->weights[i];
        const char *lookup = (e->source_name[0] != '\0') ? e->source_name : e->name;
        const void *ptr = nullptr;
        size_t bytes = 0U;
        bool is_auto_rope = false;

        const velri_rt_weight_entry *mw = velri_rt_model_find_weight(prog->model, lookup);
        if (mw) {
            ptr = mw->host_ptr;
            bytes = mw->byte_size;
        } else if (has_auto_rope) {
            bool is_cos = (strcmp(lookup, VELRI_WEIGHT_ROPE_COS) == 0);
            bool is_sin = (!is_cos && strcmp(lookup, VELRI_WEIGHT_ROPE_SIN) == 0);
            if (!is_cos && !is_sin) {
                velri_rt_ctx_set_error(prog->ctx, "program_prepare: missing required weight '%s'",
                                       lookup);
                return VELRI_RC_ERR_NOT_FOUND;
            }
            velri_rc rrc = ensure_auto_rope(prog);
            if (rrc != VELRI_RC_OK) {
                return rrc;
            }
            ptr = is_cos ? (const void *)prog->auto_rope_cos : (const void *)prog->auto_rope_sin;
            bytes = prog->auto_rope_bytes;
            is_auto_rope = true;
        } else {
            velri_rt_ctx_set_error(prog->ctx, "program_prepare: missing required weight '%s'",
                                   lookup);
            return VELRI_RC_ERR_NOT_FOUND;
        }

        velri_status st = velri_plan_bind_weight_host(prog->decode_plan, e->storage_id, ptr, bytes);
        if (st != VELRI_OK) {
            velri_rt_ctx_set_error(prog->ctx, "program_prepare: failed binding weight '%s' (%zu B): %s",
                                   lookup, bytes, velri_rc_str(velri_rt_rc_from_vl(st)));
            return velri_rt_rc_from_vl(st);
        }

        if (is_auto_rope) {
            velri_device_caps caps;
            if (velri_context_device_caps(prog->ctx->vl_ctx, e->device_id, &caps) != VELRI_OK ||
                caps.is_host_memory) {
                auto_rope_on_host = true;
            }
        }
    }

    if (!auto_rope_on_host && (prog->auto_rope_cos || prog->auto_rope_sin)) {
        velri_rt_aligned_free(prog->auto_rope_cos);
        velri_rt_aligned_free(prog->auto_rope_sin);
        prog->auto_rope_cos = nullptr;
        prog->auto_rope_sin = nullptr;
        prog->auto_rope_bytes = 0U;
    }
    return VELRI_RC_OK;
}

/* ---- De-Duplicated Dual-Plan Memory Report ---- */

static velri_program_device_mem *find_or_add_device_mem(velri_program_mem_report *rep,
                                                        uint32_t device_id, bool is_uma) {
    for (uint32_t i = 0U; i < rep->num_devices; ++i) {
        if (rep->devices[i].device_id == device_id) {
            return &rep->devices[i];
        }
    }
    if (rep->num_devices >= VELRI_PROGRAM_MAX_DEVICES) {
        return nullptr;
    }
    velri_program_device_mem *slot = &rep->devices[rep->num_devices++];
    memset(slot, 0, sizeof(*slot));
    slot->device_id = device_id;
    slot->is_uma = is_uma;
    return slot;
}

static velri_rc compute_program_memory_report(velri_program *prog) {
    velri_memory_report dec_rep;
    velri_status st = velri_plan_memory_report(prog->decode_plan, &dec_rep);
    if (st != VELRI_OK) {
        return velri_rt_rc_from_vl(st);
    }

    velri_program_mem_report *out = &prog->mem_report;
    memset(out, 0, sizeof(*out));
    out->num_devices = (dec_rep.num_devices <= VELRI_PROGRAM_MAX_DEVICES)
                           ? dec_rep.num_devices
                           : VELRI_PROGRAM_MAX_DEVICES;
    out->has_distinct_prefill_plan = prog->has_distinct_prefill;
    out->max_ctx = prog->max_ctx;
    out->prefill_chunk_size = prog->prefill_chunk_size;
    out->num_ram_transfers_decode = dec_rep.num_ram_transfers;
    out->ram_staging_bytes = dec_rep.ram_staging_bytes;

    for (uint32_t d = 0U; d < out->num_devices; ++d) {
        out->devices[d] = velri_device_memory_from_plan(&dec_rep.devices[d]);
    }

    if (prog->has_distinct_prefill && prog->prefill_plan) {
        velri_memory_report pre_rep;
        st = velri_plan_memory_report(prog->prefill_plan, &pre_rep);
        if (st != VELRI_OK) {
            return velri_rt_rc_from_vl(st);
        }
        out->num_ram_transfers_prefill = pre_rep.num_ram_transfers;
        out->ram_staging_bytes =
            velri_rt_sat_add_size(out->ram_staging_bytes, pre_rep.ram_staging_bytes);

        uint32_t pnd = (pre_rep.num_devices <= VELRI_PROGRAM_MAX_DEVICES)
                           ? pre_rep.num_devices
                           : VELRI_PROGRAM_MAX_DEVICES;
        for (uint32_t p = 0U; p < pnd; ++p) {
            const velri_device_memory_report *ps = &pre_rep.devices[p];
            velri_program_device_mem *dm =
                find_or_add_device_mem(out, ps->device_id, ps->is_uma);
            if (!dm) {
                continue;
            }
            velri_device_caps caps;
            bool is_host =
                (velri_context_device_caps(prog->ctx->vl_ctx, ps->device_id, &caps) == VELRI_OK &&
                 caps.is_host_memory);
            if (is_host) {
                if (ps->input_output_bytes > dm->input_output_bytes) {
                    dm->input_output_bytes = ps->input_output_bytes;
                }
            } else {
                dm->input_output_bytes =
                    velri_rt_sat_add_size(dm->input_output_bytes, ps->input_output_bytes);
            }
            dm->scratch_packed_bytes =
                velri_rt_sat_add_size(dm->scratch_packed_bytes, ps->scratch_packed_bytes);
            dm->scratch_unpacked_bytes =
                velri_rt_sat_add_size(dm->scratch_unpacked_bytes, ps->scratch_unpacked_bytes);
            size_t pre_extra_ovh = (ps->allocator_overhead_bytes > ps->state_overhead_bytes)
                                       ? (ps->allocator_overhead_bytes - ps->state_overhead_bytes)
                                       : 0U;
            dm->allocator_overhead_bytes =
                velri_rt_sat_add_size(dm->allocator_overhead_bytes, pre_extra_ovh);
        }
    }

    size_t total = out->ram_staging_bytes;
    for (uint32_t d = 0U; d < out->num_devices; ++d) {
        const velri_program_device_mem *dm = &out->devices[d];
        total = velri_rt_sat_add_size(total, dm->weight_bytes);
        total = velri_rt_sat_add_size(total, dm->state_bytes);
        total = velri_rt_sat_add_size(total, dm->input_output_bytes);
        total = velri_rt_sat_add_size(total, dm->scratch_packed_bytes);
        total = velri_rt_sat_add_size(total, dm->allocator_overhead_bytes);
    }
    out->total_physical_bytes = total;
    return VELRI_RC_OK;
}

static void program_cleanup_partial(velri_program *prog) {
    if (!prog) {
        return;
    }
    if (prog->prefill_plan) {
        velri_plan_destroy(prog->prefill_plan);
    }
    if (prog->decode_plan) {
        velri_plan_destroy(prog->decode_plan);
    }
    velri_manifest_destroy(&prog->prefill_manifest);
    velri_manifest_destroy(&prog->decode_manifest);
    velri_rt_aligned_free(prog->auto_rope_cos);
    velri_rt_aligned_free(prog->auto_rope_sin);
    velri_engine_config_destroy(&prog->cfg);
    free(prog);
}

/* ---- Program Public API ---- */

velri_rc velri_program_prepare(velri_ctx *ctx, velri_model *model,
                               const velri_program_options *opts, velri_program **out_prog) {
    if (!out_prog) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_prog = nullptr;
    if (!ctx || !model) {
        if (ctx) {
            velri_rt_ctx_set_error(ctx, "program_prepare: null model");
        }
        return VELRI_RC_ERR_INVALID;
    }
    velri_rt_ctx_clear_tls_error(ctx);

    if (model->ctx != ctx) {
        velri_rt_ctx_set_error(ctx, "program_prepare: model belongs to a different context");
        return VELRI_RC_ERR_INVALID;
    }
    if (opts && ((opts->placement_json == nullptr) != (opts->placement_json_len == 0U))) {
        velri_rt_ctx_set_error(
            ctx,
            "program_prepare: placement_json and placement_json_len must both be set or both empty");
        return VELRI_RC_ERR_INVALID;
    }

    if (!velri_rt_try_acquire_child(&ctx->live_children)) {
        velri_rt_ctx_set_error(ctx, "program_prepare: context is being destroyed");
        return VELRI_RC_ERR_BUSY;
    }
    if (!velri_rt_try_acquire_child(&model->live_programs)) {
        velri_rt_ctx_set_error(ctx, "program_prepare: model is being modified or destroyed");
        velri_rt_release_child(&ctx->live_children);
        return VELRI_RC_ERR_BUSY;
    }

    velri_program *prog = (velri_program *)calloc(1U, sizeof(*prog));
    if (!prog) {
        velri_rt_ctx_set_error(ctx, "program_prepare: out of memory");
        velri_rt_release_child(&model->live_programs);
        velri_rt_release_child(&ctx->live_children);
        return VELRI_RC_ERR_NOMEM;
    }
    prog->ctx = ctx;
    prog->model = model;
    prog->vocab_size = model->desc.vocab_size;
    prog->logits_dtype = VELRI_DTYPE_F32;
    velri_engine_config_init(&prog->cfg);
    velri_manifest_init(&prog->decode_manifest);
    velri_manifest_init(&prog->prefill_manifest);
    atomic_init(&prog->live_sessions, 0U);

    velri_rc fail_rc = program_parse_and_validate_config(prog, opts);
    if (fail_rc != VELRI_RC_OK) {
        goto fail;
    }
    fail_rc = program_resolve_limits(prog, opts);
    if (fail_rc != VELRI_RC_OK) {
        goto fail;
    }

    bool latent_kv =
        (opts && opts->latent_kv) || (prog->cfg.attn_mechanism == VELRI_ATTN_MECH_LATENT_KV);
    bool allow_cpu_fb = opts ? opts->allow_cpu_fallback : false;
    fail_rc = prepare_program_plans(prog, latent_kv, allow_cpu_fb);
    if (fail_rc != VELRI_RC_OK) {
        goto fail;
    }

    fail_rc = bind_program_weights(prog);
    if (fail_rc != VELRI_RC_OK) {
        goto fail;
    }
    fail_rc = compute_program_memory_report(prog);
    if (fail_rc != VELRI_RC_OK) {
        velri_rt_ctx_set_error(ctx, "program_prepare: failed computing memory report");
        goto fail;
    }

    *out_prog = prog;
    return VELRI_RC_OK;

fail:
    program_cleanup_partial(prog);
    velri_rt_release_child(&model->live_programs);
    velri_rt_release_child(&ctx->live_children);
    return fail_rc;
}

velri_rc velri_program_destroy(velri_program *prog) {
    if (!prog) {
        return VELRI_RC_OK;
    }
    if (!velri_rt_try_acquire_exclusive(&prog->live_sessions)) {
        uint32_t ls = atomic_load_explicit(&prog->live_sessions, memory_order_acquire) &
                      ~VELRI_EXCLUSIVE_BIT;
        velri_rt_ctx_set_error(prog->ctx,
                               "program_destroy: cannot destroy program with %u live session(s)",
                               ls);
        return VELRI_RC_ERR_BUSY;
    }
    velri_model *model = prog->model;
    velri_ctx *ctx = prog->ctx;
    program_cleanup_partial(prog);
    velri_rt_release_child(&model->live_programs);
    velri_rt_release_child(&ctx->live_children);
    return VELRI_RC_OK;
}

velri_rc velri_program_memory_report(const velri_program *prog,
                                     velri_program_mem_report *out_report) {
    if (!prog || !out_report) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_report = prog->mem_report;
    return VELRI_RC_OK;
}

size_t velri_program_inspect(const velri_program *prog, char *out_buf, size_t out_cap) {
    if (!prog || !out_buf || out_cap == 0U) {
        return 0U;
    }
    velri_text_writer text = {.buf = out_buf, .cap = out_cap};
    text.pos = velri_placement_format_inspect(&prog->placement, prog->model->desc.num_hidden_layers,
                                               nullptr, out_buf, out_cap);

    const velri_program_mem_report *mr = &prog->mem_report;
    velri_inspect_memory_summary(&text, mr->total_physical_bytes, mr->ram_staging_bytes,
                                 (uint64_t)mr->num_ram_transfers_decode + mr->num_ram_transfers_prefill);
    for (uint32_t d = 0U; d < mr->num_devices; ++d) {
        velri_inspect_memory_device(&text, prog->ctx->vl_ctx, &mr->devices[d]);
    }

    uint32_t dec_ops = 0U;
    uint32_t dec_elided = 0U;
    (void)velri_plan_barrier_stats(prog->decode_plan, &dec_ops, &dec_elided);
    const char *arch_name = prog->model->arch_ops->name;
    if (prog->has_distinct_prefill && prog->prefill_plan) {
        uint32_t pre_ops = 0U;
        uint32_t pre_elided = 0U;
        (void)velri_plan_barrier_stats(prog->prefill_plan, &pre_ops, &pre_elided);
        velri_text_appendf(&text,
                        "Program: arch=%s | max_ctx=%u | prefill_chunk=%u | mode=dual-plan "
                        "(shared weights+state) | decode_ops=%u (elided_barriers=%u) | "
                        "prefill_ops=%u (elided_barriers=%u)\n",
                        arch_name, prog->max_ctx, prog->prefill_chunk_size, dec_ops, dec_elided,
                        pre_ops, pre_elided);
    } else {
        velri_text_appendf(&text,
                        "Program: arch=%s | max_ctx=%u | prefill_chunk=%u | mode=unified | "
                        "ops=%u (elided_barriers=%u)\n",
                        arch_name, prog->max_ctx, prog->prefill_chunk_size, dec_ops, dec_elided);
    }
    return text.pos;
}

bool velri_program_has_distinct_prefill(const velri_program *prog) {
    return prog ? prog->has_distinct_prefill : false;
}

velri_rc velri_program_default_sampling(const velri_program *prog,
                                        velri_sampling_params *out_params) {
    if (!prog || !out_params) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_params = (velri_sampling_params){
        .temperature = prog->cfg.sampling.temperature,
        .top_p = prog->cfg.sampling.top_p,
        .top_k = prog->cfg.sampling.top_k,
        .min_p = prog->cfg.sampling.min_p,
        .seed = prog->cfg.sampling.seed,
        .reseed = false,
    };
    return VELRI_RC_OK;
}

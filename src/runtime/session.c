#include "runtime/runtime_internal.h"

#include <string.h>

static void session_cleanup_partial(velri_session *sess) {
    if (!sess) {
        return;
    }
    if (sess->prefill_slot) {
        velri_slot_destroy(sess->prefill_slot);
    }
    if (sess->decode_slot) {
        velri_slot_destroy(sess->decode_slot);
    }
    velri_sampler_destroy(&sess->sampler);
    velri_rt_aligned_free(sess->token_ids_buf);
    velri_rt_aligned_free(sess->logits_buf);
    free(sess);
}

static velri_status bind_slot_io(velri_slot *slot, const velri_manifest *m, velri_session *sess) {
    velri_status st =
        velri_slot_bind_host(slot, m->token_ids_in, sess->token_ids_buf, sess->token_ids_bytes);
    if (st == VELRI_OK) {
        st = velri_slot_bind_host(slot, m->logits_out, sess->logits_buf, sess->logits_bytes);
    }
    return st;
}

/* Pure pre-accept validation: failures here never mutate session KV or sampler state. */
static velri_rc session_validate_step(const velri_session *sess, uint32_t cur_kv,
                                      const int32_t *token_ids, uint32_t num_tokens,
                                      const velri_sampling_params *sampling,
                                      const velri_cancel *cancel, velri_step_result *out_result) {
    velri_ctx *ctx = sess->prog->ctx;
    if (atomic_load_explicit(&sess->needs_reset, memory_order_acquire)) {
        velri_rt_ctx_set_error(ctx,
                               "session_step: session requires velri_session_reset() before continuing");
        return VELRI_RC_ERR_BUSY;
    }
    if (sampling) {
        velri_rc rc = velri_sampler_validate_params(sampling);
        if (rc != VELRI_RC_OK) {
            velri_rt_ctx_set_error(ctx, "session_step: invalid sampling parameters");
            return rc;
        }
    }
    if (num_tokens > sess->max_ctx || cur_kv > sess->max_ctx - num_tokens) {
        velri_rt_ctx_set_error(ctx, "session_step: sequence length (%u + %u) exceeds max_ctx (%u)",
                               cur_kv, num_tokens, sess->max_ctx);
        return VELRI_RC_ERR_OVERFLOW;
    }
    for (uint32_t i = 0U; i < num_tokens; ++i) {
        if (token_ids[i] < 0 || (uint32_t)token_ids[i] >= sess->vocab_size) {
            velri_rt_ctx_set_error(ctx, "session_step: token_ids[%u]=%d out of range [0, %u)", i,
                                   token_ids[i], sess->vocab_size);
            return VELRI_RC_ERR_INVALID;
        }
    }
    if (cancel && velri_cancel_is_requested(&cancel->tok)) {
        out_result->stop_reason = VELRI_STEP_STOP_CANCELLED;
        return VELRI_RC_ERR_CANCELLED;
    }
    return VELRI_RC_OK;
}

static velri_rc session_execute_chunks(velri_session *sess, velri_slot *active_slot,
                                       const int32_t *token_ids, uint32_t num_tokens,
                                       velri_cancel_token *vl_cancel, velri_step_result *out) {
    bool any_chunk_committed = false;

    while (out->tokens_processed < num_tokens) {
        if (vl_cancel && velri_cancel_is_requested(vl_cancel)) {
            if (any_chunk_committed) {
                atomic_store_explicit(&sess->needs_reset, true, memory_order_release);
            }
            out->stop_reason = VELRI_STEP_STOP_CANCELLED;
            return VELRI_RC_ERR_CANCELLED;
        }

        uint32_t offset = out->tokens_processed;
        uint32_t rem = num_tokens - offset;
        uint32_t chunk = (rem > sess->prefill_chunk_size) ? sess->prefill_chunk_size : rem;
        memcpy(sess->token_ids_buf, token_ids + offset, (size_t)chunk * sizeof(int32_t));

        velri_step_params sp = {
            .pos_offset_override = (size_t)out->kv_len,
            .has_pos_offset = true,
            .seq_len_override = (size_t)chunk,
            .has_seq_len = true,
            .kv_len_override = (size_t)out->kv_len,
            .has_kv_len = true,
        };
        velri_status st = velri_slot_run(active_slot, &sp, vl_cancel);
        out->last_step_allocs += velri_slot_last_run_allocs(active_slot);
        out->last_step_graph_replays += velri_slot_last_run_graph_replays(active_slot);

        if (st != VELRI_OK) {
            if (velri_slot_needs_reset(active_slot) || any_chunk_committed) {
                atomic_store_explicit(&sess->needs_reset, true, memory_order_release);
            }
            velri_rc rc = velri_rt_rc_from_vl(st);
            out->stop_reason =
                (st == VELRI_ERR_CANCELLED) ? VELRI_STEP_STOP_CANCELLED : VELRI_STEP_STOP_ERROR;
            if (st != VELRI_ERR_CANCELLED) {
                velri_rt_ctx_set_error(sess->prog->ctx,
                                       "session_step: slot execution failed at offset %u (%s)",
                                       offset, velri_rc_str(rc));
            }
            return rc;
        }

        out->kv_len += chunk;
        atomic_store_explicit(&sess->kv_pos, out->kv_len, memory_order_release);
        out->tokens_processed += chunk;
        any_chunk_committed = true;
    }
    return VELRI_RC_OK;
}

/* ---- Session Public API ---- */

velri_rc velri_session_create(velri_program *prog, const velri_session_options *opts,
                              velri_session **out_sess) {
    if (!out_sess) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_sess = nullptr;
    if (!prog) {
        return VELRI_RC_ERR_INVALID;
    }
    velri_rt_ctx_clear_tls_error(prog->ctx);

    if (!velri_rt_try_acquire_child(&prog->live_sessions)) {
        velri_rt_ctx_set_error(prog->ctx, "session_create: program is being destroyed");
        return VELRI_RC_ERR_BUSY;
    }

    velri_session *sess = nullptr;
    velri_rc fail_rc = VELRI_RC_OK;
    size_t chunk_sz = prog->prefill_chunk_size;
    size_t vocab_sz = prog->vocab_size;

    if (prog->logits_dtype != VELRI_DTYPE_F32) {
        velri_rt_ctx_set_error(prog->ctx,
                               "session_create: only F32 logits output is supported currently");
        fail_rc = VELRI_RC_ERR_UNSUPPORTED;
        goto fail;
    }
    if (chunk_sz > SIZE_MAX / sizeof(int32_t) || vocab_sz > SIZE_MAX / sizeof(float)) {
        velri_rt_ctx_set_error(prog->ctx, "session_create: IO buffer size overflow");
        fail_rc = VELRI_RC_ERR_OVERFLOW;
        goto fail;
    }

    sess = (velri_session *)calloc(1U, sizeof(*sess));
    if (!sess) {
        velri_rt_ctx_set_error(prog->ctx, "session_create: out of memory");
        fail_rc = VELRI_RC_ERR_NOMEM;
        goto fail;
    }
    sess->prog = prog;
    sess->max_ctx = prog->max_ctx;
    sess->prefill_chunk_size = prog->prefill_chunk_size;
    sess->vocab_size = prog->vocab_size;
    sess->token_ids_bytes = chunk_sz * sizeof(int32_t);
    sess->logits_bytes = vocab_sz * sizeof(float);
    atomic_init(&sess->kv_pos, 0U);
    atomic_init(&sess->needs_reset, false);
    atomic_init(&sess->in_step, false);

    sess->token_ids_buf = (int32_t *)velri_rt_aligned_alloc(64U, sess->token_ids_bytes);
    sess->logits_buf = (float *)velri_rt_aligned_alloc(64U, sess->logits_bytes);
    if (!sess->token_ids_buf || !sess->logits_buf) {
        velri_rt_ctx_set_error(prog->ctx, "session_create: out of memory allocating IO buffers");
        fail_rc = VELRI_RC_ERR_NOMEM;
        goto fail;
    }
    memset(sess->token_ids_buf, 0, sess->token_ids_bytes);
    memset(sess->logits_buf, 0, sess->logits_bytes);

    uint64_t seed = 0x56454C52495F3031ULL; /* Deterministic default ("VELRI_01") */
    if (opts && opts->has_prng_seed) {
        seed = opts->prng_seed;
    } else if (prog->has_default_seed) {
        seed = prog->default_seed;
    }
    fail_rc = velri_sampler_init(&sess->sampler, prog->vocab_size, seed);
    if (fail_rc != VELRI_RC_OK) {
        velri_rt_ctx_set_error(prog->ctx, "session_create: sampler init failed");
        goto fail;
    }

    velri_status st = velri_slot_create(prog->decode_plan, &sess->decode_slot);
    if (st == VELRI_OK) {
        st = bind_slot_io(sess->decode_slot, &prog->decode_manifest, sess);
    }
    if (st != VELRI_OK) {
        velri_rt_ctx_set_error(prog->ctx, "session_create: decode slot setup failed (%s)",
                               velri_rc_str(velri_rt_rc_from_vl(st)));
        fail_rc = velri_rt_rc_from_vl(st);
        goto fail;
    }

    if (prog->has_distinct_prefill && prog->prefill_plan) {
        st = velri_slot_create_shared_state(prog->prefill_plan, sess->decode_slot,
                                            &sess->prefill_slot);
        if (st == VELRI_OK) {
            st = bind_slot_io(sess->prefill_slot, &prog->prefill_manifest, sess);
        }
        if (st != VELRI_OK) {
            velri_rt_ctx_set_error(prog->ctx,
                                   "session_create: shared-state prefill slot setup failed (%s)",
                                   velri_rc_str(velri_rt_rc_from_vl(st)));
            fail_rc = velri_rt_rc_from_vl(st);
            goto fail;
        }
    }

    *out_sess = sess;
    return VELRI_RC_OK;

fail:
    session_cleanup_partial(sess);
    velri_rt_release_child(&prog->live_sessions);
    return fail_rc;
}

velri_rc velri_session_destroy(velri_session *sess) {
    if (!sess) {
        return VELRI_RC_OK;
    }
    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(&sess->in_step, &expected, true,
                                                 memory_order_acq_rel, memory_order_acquire)) {
        velri_rt_ctx_set_error(sess->prog->ctx,
                               "session_destroy: cannot destroy session during active step");
        return VELRI_RC_ERR_BUSY;
    }
    velri_program *prog = sess->prog;
    session_cleanup_partial(sess);
    velri_rt_release_child(&prog->live_sessions);
    return VELRI_RC_OK;
}

velri_rc velri_session_step(velri_session *sess, const int32_t *token_ids, uint32_t num_tokens,
                            const velri_sampling_params *sampling, velri_cancel *cancel,
                            velri_step_result *out_result) {
    if (!out_result) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_result = (velri_step_result){
        .stop_reason = VELRI_STEP_STOP_ERROR,
        .sampled_token = -1,
    };
    if (!sess || !token_ids || num_tokens == 0U) {
        if (sess) {
            velri_rt_ctx_set_error(sess->prog->ctx,
                                   "session_step: invalid token_ids or num_tokens == 0");
        }
        return VELRI_RC_ERR_INVALID;
    }

    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(&sess->in_step, &expected, true,
                                                 memory_order_acq_rel, memory_order_acquire)) {
        velri_rt_ctx_set_error(sess->prog->ctx, "session_step: concurrent call on the same session");
        return VELRI_RC_ERR_BUSY;
    }

    uint32_t cur_kv = atomic_load_explicit(&sess->kv_pos, memory_order_relaxed);
    out_result->vocab_size = sess->vocab_size;
    out_result->kv_len = cur_kv;

    velri_rc rc =
        session_validate_step(sess, cur_kv, token_ids, num_tokens, sampling, cancel, out_result);
    if (rc != VELRI_RC_OK) {
        goto done;
    }

    velri_slot *active_slot =
        (sess->prog->has_distinct_prefill && num_tokens > 1U && sess->prefill_slot)
            ? sess->prefill_slot
            : sess->decode_slot;
    velri_cancel_token *vl_cancel = cancel ? &cancel->tok : nullptr;

    rc = session_execute_chunks(sess, active_slot, token_ids, num_tokens, vl_cancel, out_result);
    if (rc != VELRI_RC_OK) {
        goto done;
    }

    int32_t sampled = -1;
    if (sampling) {
        rc = velri_sampler_sample(&sess->sampler, sess->logits_buf, sampling, &sampled);
        if (rc != VELRI_RC_OK) {
            atomic_store_explicit(&sess->needs_reset, true, memory_order_release);
            velri_rt_ctx_set_error(sess->prog->ctx, "session_step: sampling failed (%s)",
                                   velri_rc_str(rc));
            goto done;
        }
    }

    out_result->stop_reason = VELRI_STEP_STOP_OK;
    out_result->logits = sess->logits_buf;
    out_result->sampled_token = sampled;

done:
    atomic_store_explicit(&sess->in_step, false, memory_order_release);
    return rc;
}

velri_rc velri_session_reset(velri_session *sess) {
    if (!sess) {
        return VELRI_RC_ERR_INVALID;
    }
    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(&sess->in_step, &expected, true,
                                                 memory_order_acq_rel, memory_order_acquire)) {
        velri_rt_ctx_set_error(sess->prog->ctx, "session_reset: concurrent call during active step");
        return VELRI_RC_ERR_BUSY;
    }

    velri_status st = velri_slot_reset(sess->decode_slot);
    bool decode_reset_ok = (st == VELRI_OK);
    if (decode_reset_ok && sess->prefill_slot) {
        st = velri_slot_reset(sess->prefill_slot);
    }
    velri_rc rc = velri_rt_rc_from_vl(st);

    if (st == VELRI_OK) {
        atomic_store_explicit(&sess->kv_pos, 0U, memory_order_release);
        atomic_store_explicit(&sess->needs_reset, false, memory_order_release);
        velri_sampler_reset(&sess->sampler);
    } else {
        if (decode_reset_ok) {
            atomic_store_explicit(&sess->kv_pos, 0U, memory_order_release);
        }
        atomic_store_explicit(&sess->needs_reset, true, memory_order_release);
        velri_rt_ctx_set_error(sess->prog->ctx, "session_reset: slot reset failed (%s)",
                               velri_rc_str(rc));
    }
    atomic_store_explicit(&sess->in_step, false, memory_order_release);
    return rc;
}

bool velri_session_needs_reset(const velri_session *sess) {
    return sess ? atomic_load_explicit(&sess->needs_reset, memory_order_acquire) : false;
}

uint32_t velri_session_kv_len(const velri_session *sess) {
    return sess ? atomic_load_explicit(&sess->kv_pos, memory_order_acquire) : 0U;
}

uint32_t velri_session_max_ctx(const velri_session *sess) {
    return sess ? sess->max_ctx : 0U;
}

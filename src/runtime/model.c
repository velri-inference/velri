#include "runtime/runtime_internal.h"

#include <math.h>

/* ---- Public-to-VelriLib Enum Mapping (switch retains -Wswitch exhaustiveness checks) ---- */

static bool map_type_to_vl(velri_type t, velri_dtype *out) {
    switch (t) {
    case VELRI_TYPE_F32:
        *out = VELRI_DTYPE_F32;
        return true;
    case VELRI_TYPE_F16:
        *out = VELRI_DTYPE_F16;
        return true;
    case VELRI_TYPE_BF16:
        *out = VELRI_DTYPE_BF16;
        return true;
    default:
        return false;
    }
}

static bool map_act_to_vl(velri_activation a, velri_act_kind *out) {
    switch (a) {
    case VELRI_ACTIVATION_SILU:
        *out = VELRI_ACT_SILU;
        return true;
    case VELRI_ACTIVATION_GELU_TANH:
        *out = VELRI_ACT_GELU_TANH;
        return true;
    case VELRI_ACTIVATION_RELU2:
        *out = VELRI_ACT_RELU2;
        return true;
    default:
        return false;
    }
}

static bool map_rope_to_vl(velri_rotary_mode m, velri_rope_mode *out) {
    switch (m) {
    case VELRI_ROTARY_HALF:
        *out = VELRI_ROPE_HALF;
        return true;
    case VELRI_ROTARY_INTERLEAVED:
        *out = VELRI_ROPE_INTERLEAVED;
        return true;
    default:
        return false;
    }
}

/* ---- Model Lifecycle & Weight Registry API ---- */

velri_rc velri_model_create_synthetic(velri_ctx *ctx, const velri_model_desc *desc,
                                      velri_model **out_model) {
    if (!out_model) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_model = nullptr;
    if (!ctx || !desc) {
        if (ctx) {
            velri_rt_ctx_set_error(ctx, "model_create_synthetic: null descriptor");
        }
        return VELRI_RC_ERR_INVALID;
    }
    velri_rt_ctx_clear_tls_error(ctx);
    if (!velri_rt_try_acquire_child(&ctx->live_children)) {
        velri_rt_ctx_set_error(ctx, "model_create_synthetic: context is being destroyed");
        return VELRI_RC_ERR_BUSY;
    }

    velri_rc rc = VELRI_RC_ERR_INVALID;
    velri_model_desc norm = *desc;
    const velri_arch_ops *arch_ops = velri_arch_find(norm.arch);
    velri_dtype vl_dtype = VELRI_DTYPE_F32;
    velri_act_kind vl_act = VELRI_ACT_SILU;
    velri_rope_mode vl_rope = VELRI_ROPE_HALF;

    if (!arch_ops) {
        velri_rt_ctx_set_error(ctx, "model_create_synthetic: unsupported architecture %d",
                               (int)norm.arch);
        rc = VELRI_RC_ERR_UNSUPPORTED;
        goto fail;
    }
    if (!map_type_to_vl(norm.weight_type, &vl_dtype) || !map_act_to_vl(norm.hidden_act, &vl_act) ||
        !map_rope_to_vl(norm.rope_mode, &vl_rope)) {
        velri_rt_ctx_set_error(ctx,
                               "model_create_synthetic: invalid dtype, activation, or rotary mode");
        goto fail;
    }
    if (norm.vocab_size == 0U || norm.vocab_size > (uint32_t)INT32_MAX ||
        norm.max_position_embeddings == 0U || norm.hidden_size == 0U ||
        norm.num_hidden_layers == 0U || norm.num_attention_heads == 0U ||
        norm.intermediate_size == 0U) {
        velri_rt_ctx_set_error(ctx,
                               "model_create_synthetic: model dimensions must be non-zero and valid");
        goto fail;
    }

    if (norm.num_key_value_heads == 0U) {
        norm.num_key_value_heads = norm.num_attention_heads;
    }
    if (norm.num_key_value_heads > norm.num_attention_heads ||
        (norm.num_attention_heads % norm.num_key_value_heads) != 0U) {
        velri_rt_ctx_set_error(
            ctx,
            "model_create_synthetic: num_attention_heads (%u) must be a multiple of num_key_value_heads (%u)",
            norm.num_attention_heads, norm.num_key_value_heads);
        goto fail;
    }
    if (norm.head_dim == 0U) {
        if ((norm.hidden_size % norm.num_attention_heads) != 0U) {
            velri_rt_ctx_set_error(
                ctx,
                "model_create_synthetic: hidden_size (%u) not divisible by num_attention_heads (%u)",
                norm.hidden_size, norm.num_attention_heads);
            goto fail;
        }
        norm.head_dim = norm.hidden_size / norm.num_attention_heads;
    }
    if (norm.head_dim == 0U || norm.head_dim > VELRI_ATTN_MAX_HEAD_DIM) {
        velri_rt_ctx_set_error(ctx, "model_create_synthetic: invalid head_dim %u", norm.head_dim);
        goto fail;
    }
    if (norm.norm_eps == 0.0f) {
        norm.norm_eps = 1e-5f;
    }
    if (!isfinite(norm.norm_eps) || norm.norm_eps <= 0.0f) {
        velri_rt_ctx_set_error(ctx, "model_create_synthetic: norm_eps must be finite and > 0");
        goto fail;
    }
    if (arch_ops->policy->validate_desc) {
        char arch_err[192] = {0};
        rc = arch_ops->policy->validate_desc(&norm, arch_err, sizeof(arch_err));
        if (rc != VELRI_RC_OK) {
            velri_rt_ctx_set_error(ctx, "model_create_synthetic: %s",
                                   arch_err[0] ? arch_err : "invalid architecture parameters");
            goto fail;
        }
    }

    velri_model *m = (velri_model *)calloc(1U, sizeof(*m));
    if (!m) {
        velri_rt_ctx_set_error(ctx, "model_create_synthetic: out of memory");
        rc = VELRI_RC_ERR_NOMEM;
        goto fail;
    }
    m->ctx = ctx;
    m->desc = norm;
    m->arch_ops = arch_ops;
    m->vl_dtype = vl_dtype;
    m->vl_act = vl_act;
    m->vl_rope = vl_rope;
    atomic_init(&m->live_programs, 0U);
    atomic_init(&m->num_weights, 0U);
    *out_model = m;
    return VELRI_RC_OK;

fail:
    velri_rt_release_child(&ctx->live_children);
    return rc;
}

velri_rc velri_model_destroy(velri_model *model) {
    if (!model) {
        return VELRI_RC_OK;
    }
    if (!velri_rt_try_acquire_exclusive(&model->live_programs)) {
        velri_rt_ctx_set_error(model->ctx, "model_destroy: model has active programs or a weight binding");
        return VELRI_RC_ERR_BUSY;
    }
    velri_ctx *ctx = model->ctx;
    free(model->weights);
    free(model);
    velri_rt_release_child(&ctx->live_children);
    return VELRI_RC_OK;
}

velri_rc velri_model_get_desc(const velri_model *model, velri_model_desc *out_desc) {
    if (!model || !out_desc) {
        return VELRI_RC_ERR_INVALID;
    }
    *out_desc = model->desc;
    return VELRI_RC_OK;
}

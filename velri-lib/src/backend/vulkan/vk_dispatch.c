#include "vk_internal.h"

#include <math.h>
#include <string.h>

/*
 * Vulkan Command Encoding, Multi-Stage Batched Ring Submission, Completion Handling,
 * and Backend Vtable Registration.
 */

static bool binding_bda(const velri_buffer_binding *b, uint64_t *out_bda) {
    if (!b || !b->mem_handle) {
        return false;
    }
    const vk_buf *vb = (const vk_buf *)b->mem_handle;
    if (b->byte_offset > vb->size) {
        return false;
    }
    *out_bda = (uint64_t)(vb->bda + (VkDeviceAddress)b->byte_offset);
    return true;
}

static bool grid_fits(const velri_backend_device *dev, uint32_t gx, uint32_t gy, uint32_t gz) {
    return gx > 0U && gy > 0U && gz > 0U && gx <= dev->max_wg_count[0] &&
           gy <= dev->max_wg_count[1] && gz <= dev->max_wg_count[2];
}

static inline uint32_t grid_x_vec4_256(uint32_t count) {
    return velri_div_ceil_u32(velri_div_ceil_u32(count, 4U), 256U);
}

static void set_dispatch_stage(vk_dispatch_stage *st, VkPipeline pipe, const void *pc,
                               size_t pc_size, uint32_t gx, uint32_t gy, uint32_t gz) {
    st->pipe = pipe;
    st->gx = gx;
    st->gy = gy;
    st->gz = gz;
    st->pc_size = (uint32_t)pc_size;
    memcpy(st->pc, pc, pc_size);
}

static velri_status encode_item_locked(velri_backend_device *dev, const velri_submit_item *item,
                                       vk_encoded_dispatch *out) {
    memset(out, 0, sizeof(*out));
    if (!velri_submit_item_header_valid(item)) {
        return VELRI_ERR_INVALID;
    }
    const velri_resolved_op *op = item->op;

    uint64_t ib[VELRI_OP_MAX_INPUTS] = {0};
    uint64_t ob[VELRI_OP_MAX_OUTPUTS] = {0};
    for (uint32_t i = 0U; i < op->num_inputs; ++i) {
        if (op->input_present[i] && (!item->inputs || !binding_bda(&item->inputs[i], &ib[i]))) {
            return VELRI_ERR_INVALID;
        }
    }
    for (uint32_t o = 0U; o < op->num_outputs; ++o) {
        if (!item->outputs || !binding_bda(&item->outputs[o], &ob[o])) {
            return VELRI_ERR_INVALID;
        }
    }

    struct velri_prepared_op plan;
    velri_status st = VELRI_OK;
    if (item->prepared) {
        plan = *item->prepared;
    } else {
        st = velri_op_validate(op);
        if (st != VELRI_OK) {
            return st;
        }
        st = velri_vk_resolve_op_pipes(dev, op, &plan);
        if (st != VELRI_OK) {
            return st;
        }
    }

    VkPipeline pipe0 = VK_NULL_HANDLE;
    if (op->kind != VELRI_OP_LINEAR) {
        pipe0 = velri_vk_get_pipe_locked(dev, plan.pipe_id);
        if (!pipe0) {
            return VELRI_ERR_UNSUPPORTED;
        }
    }

    uint32_t warps_per_wg = VK_GEMV_WG_SIZE / dev->subgroup_size;
    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t m = (uint32_t)op->inputs[1].dims[0];
        uint32_t k = (uint32_t)op->inputs[0].dims[1];
        uint32_t xs = (uint32_t)op->inputs[0].strides[0];
        uint32_t ws = (uint32_t)op->inputs[1].strides[0];
        uint32_t ys = (uint32_t)op->outputs[0].strides[0];
        uint32_t flags = velri_op_linear_epilogue_flags(op);

        int active_pipe_id = plan.pipe_id;
        if (rows <= 2U && plan.alt_pipe_id >= 0) {
            active_pipe_id = plan.alt_pipe_id;
        }
        pipe0 = velri_vk_get_pipe_locked(dev, active_pipe_id);
        if (!pipe0) {
            return VELRI_ERR_UNSUPPORTED;
        }

        vk_linear_args args = {
            .x = ib[0],
            .w = ib[1],
            .bias = (op->num_inputs > 2U && op->input_present[2]) ? ib[2] : 0ULL,
            .x_scale = (op->num_inputs > 3U && op->input_present[3]) ? ib[3] : 0ULL,
            .w_scale = (op->num_inputs > 4U && op->input_present[4]) ? ib[4] : 0ULL,
            .y = ob[0],
            .rows = rows,
            .m = m,
            .k = k,
            .x_stride = xs,
            .w_stride = ws,
            .y_stride = ys,
            .flags = flags,
            .split_k = 1U,
        };

        uint32_t gx = 1U, gy = 1U, gz = 1U;
        if (active_pipe_id == PIPE_GEMV_F32 || active_pipe_id == PIPE_GEMV_F16 ||
            active_pipe_id == PIPE_GEMV_BF16 || active_pipe_id == PIPE_GEMV_W8A8) {
            gx = velri_div_ceil_u32(m, warps_per_wg);
            gy = rows;
        } else if (active_pipe_id == PIPE_GEMM_COOPMAT_F16_64 ||
                   active_pipe_id == PIPE_GEMM_COOPMAT_BF16_64 ||
                   active_pipe_id == PIPE_GEMM_COOPMAT_W8A8_64) {
            gx = velri_div_ceil_u32(m, 64U);
            gy = velri_div_ceil_u32(rows, 64U);
        } else {
            gx = velri_div_ceil_u32(m, 128U);
            gy = velri_div_ceil_u32(rows, 128U);
        }

        out->num_stages = 1U;
        if (active_pipe_id == PIPE_GEMM_F32 && plan.split_k > 1U && plan.reduce_pipe_id >= 0) {
            uint32_t sk = velri_vk_choose_split_k(rows, m, k, 128U, 128U, VK_TARGET_WGS, 8U);
            if (sk > plan.split_k) {
                sk = plan.split_k;
            }
            uint32_t part_stride = 0U;
            size_t need_bytes = 0U;
            if (sk > 1U && velri_vk_splitk_layout(sk, rows, m, &part_stride, &need_bytes)) {
                VkPipeline reduce_pipe =
                    (dev->splitk_buf && dev->splitk_buf->size >= need_bytes)
                        ? velri_vk_get_pipe_locked(dev, plan.reduce_pipe_id)
                        : VK_NULL_HANDLE;
                if (reduce_pipe != VK_NULL_HANDLE) {
                    uint64_t total_u4 = (uint64_t)rows * (uint64_t)(part_stride >> 2U);
                    uint64_t rgx = (total_u4 + 255ULL) / 256ULL;
                    if (rgx == 0ULL || rgx > dev->max_wg_count[0]) {
                        return VELRI_ERR_OVERFLOW;
                    }
                    args.y = (uint64_t)dev->splitk_buf->bda;
                    args.y_stride = part_stride;
                    args.split_k = sk;
                    gz = sk;

                    vk_splitk_reduce_args red = {
                        .partials = (uint64_t)dev->splitk_buf->bda,
                        .bias = args.bias,
                        .x_scale = args.x_scale,
                        .w_scale = args.w_scale,
                        .y = ob[0],
                        .rows = rows,
                        .m = m,
                        .part_stride_u4 = part_stride >> 2U,
                        .y_stride = ys,
                        .split_k = sk,
                        .flags = flags,
                    };
                    set_dispatch_stage(&out->stages[1], reduce_pipe, &red, sizeof(red),
                                       (uint32_t)rgx, 1U, 1U);
                    out->num_stages = 2U;
                    out->uses_splitk = true;
                    out->splitk_bytes = need_bytes;
                }
            }
        }
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), gx, gy, gz);
        break;
    }

    case VELRI_OP_RMSNORM: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        bool has_w = (op->num_inputs > 1U && op->input_present[1]);
        vk_rmsnorm_args args = {
            .x = ib[0],
            .w = has_w ? ib[1] : 0ULL,
            .y = ob[0],
            .rows = rows,
            .dim = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .eps = op->raw.attr.rmsnorm.eps,
            .has_weight = has_w ? 1U : 0U,
            .unit_offset = op->raw.attr.rmsnorm.unit_offset ? 1U : 0U,
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), args.rows, 1U, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_FUSED_ADD_RMSNORM: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        bool has_w = (op->num_inputs > 1U && op->input_present[1]);
        vk_fused_add_rmsnorm_args args = {
            .residual_inout = ob[0],
            .x = ib[0],
            .w = has_w ? ib[1] : 0ULL,
            .y = ob[1],
            .rows = rows,
            .dim = (uint32_t)op->inputs[0].dims[1],
            .r_stride = (uint32_t)op->outputs[0].strides[0],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[1].strides[0],
            .eps = op->raw.attr.rmsnorm.eps,
            .has_weight = has_w ? 1U : 0U,
            .unit_offset = op->raw.attr.rmsnorm.unit_offset ? 1U : 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), args.rows, 1U, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_LAYERNORM: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        bool has_w = (op->num_inputs > 1U && op->input_present[1]);
        bool has_b = (op->num_inputs > 2U && op->input_present[2]);
        vk_layernorm_args args = {
            .x = ib[0],
            .w = has_w ? ib[1] : 0ULL,
            .b = has_b ? ib[2] : 0ULL,
            .y = ob[0],
            .rows = rows,
            .dim = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .eps = op->raw.attr.layernorm.eps,
            .has_weight = has_w ? 1U : 0U,
            .has_bias = has_b ? 1U : 0U,
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), args.rows, 1U, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_ROPE: {
        uint32_t pos = 0U, nt = 0U;
        st = velri_op_resolve_rope_step(op, item->params, &nt, &pos);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t nh = (uint32_t)op->inputs[0].dims[1];
        uint32_t hd = (uint32_t)op->inputs[0].dims[2];
        uint32_t rd = (uint32_t)op->raw.attr.rope.rotary_dim;
        uint32_t tsx = (uint32_t)op->inputs[0].strides[0];
        uint32_t hsx = (uint32_t)op->inputs[0].strides[1];
        uint32_t tsy = (uint32_t)op->outputs[0].strides[0];
        uint32_t hsy = (uint32_t)op->outputs[0].strides[1];
        uint32_t v_elems = (op->inputs[0].dtype == VELRI_DTYPE_F32) ? 4U : 8U;
        uint32_t half_rot = rd >> 1U;
        bool flat_fast =
            (op->raw.attr.rope.mode == VELRI_ROPE_HALF) && (rd == hd) && (half_rot > 0U) &&
            ((half_rot & (v_elems - 1U)) == 0U) && ((tsx & (v_elems - 1U)) == 0U) &&
            ((hsx & (v_elems - 1U)) == 0U) && ((tsy & (v_elems - 1U)) == 0U) &&
            ((hsy & (v_elems - 1U)) == 0U) && ((half_rot & 3U) == 0U) &&
            ((ib[0] & 15ULL) == 0ULL) && ((ob[0] & 15ULL) == 0ULL) &&
            ((ib[1] & 15ULL) == 0ULL) && ((ib[2] & 15ULL) == 0ULL);
        uint64_t flat_gx = 0ULL;
        if (flat_fast) {
            uint64_t total_vec =
                (uint64_t)nt * (uint64_t)nh * (uint64_t)(half_rot / v_elems);
            flat_gx = (total_vec + 255ULL) / 256ULL;
            if (flat_gx == 0ULL || flat_gx > dev->max_wg_count[0]) {
                flat_fast = false;
            }
        }
        vk_rope_args args = {
            .x = ib[0],
            .cos_tbl = ib[1],
            .sin_tbl = ib[2],
            .y = ob[0],
            .n_tokens = nt,
            .n_heads = nh,
            .head_dim = hd,
            .rotary_dim = rd,
            .pos_offset = pos,
            .tok_sx = tsx,
            .head_sx = hsx,
            .tok_sy = tsy,
            .head_sy = hsy,
            .mode = (uint32_t)op->raw.attr.rope.mode | (flat_fast ? 2U : 0U),
        };
        uint32_t gx = flat_fast ? (uint32_t)flat_gx : velri_div_ceil_u32(nh, 8U);
        uint32_t gy = flat_fast ? 1U : nt;
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), gx, gy, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_ACTIVATION: {
        uint32_t count = 0U;
        st = velri_op_effective_elemwise_count(op, &op->inputs[0], item->params, &count);
        if (st != VELRI_OK) {
            return st;
        }
        vk_activation_args args = {.x = ib[0], .y = ob[0], .count = count, .pad = 0U};
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(count), 1U,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_GATED_MLP: {
        uint32_t count = 0U;
        st = velri_op_effective_elemwise_count(op, &op->inputs[0], item->params, &count);
        if (st != VELRI_OK) {
            return st;
        }
        vk_gated_mlp_args args = {
            .gate = ib[0],
            .up = ib[1],
            .y = ob[0],
            .count = count,
            .clamp_limit = op->raw.attr.gated_mlp.clamp_limit,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(count), 1U,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_VEC_AXPBY: {
        uint32_t count = 0U;
        st = velri_op_effective_elemwise_count(op, &op->inputs[0], item->params, &count);
        if (st != VELRI_OK) {
            return st;
        }
        bool has_b = (op->num_inputs > 1U && op->input_present[1]) &&
                     (op->raw.attr.axpby.beta != 0.0f);
        vk_vec_axpby_args args = {
            .a = ib[0],
            .b = has_b ? ib[1] : 0ULL,
            .y = ob[0],
            .count = count,
            .alpha = op->raw.attr.axpby.alpha,
            .beta = op->raw.attr.axpby.beta,
            .has_b = has_b ? 1U : 0U,
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(count), 1U,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_VEC_MUL: {
        uint32_t count = 0U;
        st = velri_op_effective_elemwise_count(op, &op->inputs[0], item->params, &count);
        if (st != VELRI_OK) {
            return st;
        }
        vk_vec_mul_args args = {.a = ib[0], .b = ib[1], .y = ob[0], .count = count, .pad = 0U};
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(count), 1U,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_SOFTMAX: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        vk_softmax_args args = {
            .x = ib[0],
            .y = ob[0],
            .rows = rows,
            .cols = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), args.rows, 1U, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_FWHT: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t suh_mode = velri_op_fwht_suh_mode(op);
        uint32_t bs = (uint32_t)op->raw.attr.fwht.block_size;
        vk_fwht_args args = {
            .x = ib[0],
            .suh = (suh_mode != VELRI_FWHT_SUH_NONE) ? ib[1] : 0ULL,
            .y = ob[0],
            .rows = rows,
            .cols = (uint32_t)op->inputs[0].dims[1],
            .block_size = bs,
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .suh_mode = suh_mode,
            .inv_sqrt_b = 1.0f / sqrtf((float)bs),
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args),
                           velri_div_ceil_u32(args.cols, 1024U), args.rows, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_QUANTIZE_ROW: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        vk_quantize_row_args args = {
            .x = ib[0],
            .x_i8 = ob[0],
            .scale_out = ob[1],
            .rows = rows,
            .k = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .xi8_stride = (uint32_t)op->outputs[0].strides[0],
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), args.rows, 1U, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_EMBEDDING: {
        uint32_t rows = 0U, pos_off = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        bool has_ids = velri_op_embedding_has_ids(op);
        if (!has_ids) {
            st = velri_op_resolve_pos_offset(op->raw.attr.embedding.pos_offset, item->params,
                                             &pos_off);
            if (st != VELRI_OK) {
                return st;
            }
        }
        uint32_t dim = (uint32_t)op->outputs[0].dims[1];
        vk_embedding_args args = {
            .table = ib[0],
            .ids = has_ids ? ib[1] : 0ULL,
            .out = ob[0],
            .rows = rows,
            .dim = dim,
            .vocab = (uint32_t)op->inputs[0].dims[0],
            .table_stride = (uint32_t)op->inputs[0].strides[0],
            .out_stride = (uint32_t)op->outputs[0].strides[0],
            .pos_offset = pos_off,
            .has_ids = has_ids ? 1U : 0U,
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(dim), rows,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_ATTENTION: {
        velri_attn_step_args aa;
        st = velri_op_resolve_attention_step(op, item->params, &aa);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t nh = op->raw.attr.attention.n_heads;
        uint32_t nkvh = op->raw.attr.attention.n_kv_heads;
        uint32_t same_kv = (ob[1] == ob[2]) ? 1U : 0U;

        vk_kv_append_args s1 = {
            .k_in = ib[1],
            .v_in = ib[2],
            .k_cache = ob[1],
            .v_cache = ob[2],
            .rows = aa.active_t,
            .n_kv_heads = nkvh,
            .d_k = aa.d_k,
            .d_v = aa.d_v,
            .max_ctx = aa.max_ctx,
            .kv_pos = aa.kv_pos,
            .k_stride = (uint32_t)op->inputs[1].strides[0],
            .v_stride = (uint32_t)op->inputs[2].strides[0],
            .k_head_stride = (uint32_t)op->outputs[1].strides[0],
            .k_row_stride = (uint32_t)op->outputs[1].strides[1],
            .v_head_stride = (uint32_t)op->outputs[2].strides[0],
            .v_row_stride = (uint32_t)op->outputs[2].strides[1],
            .same_kv_cache = same_kv,
            .pad = 0U,
        };
        uint32_t append_cols = (same_kv || aa.d_k >= aa.d_v) ? (nkvh * aa.d_k) : (nkvh * aa.d_v);
        set_dispatch_stage(&out->stages[0], pipe0, &s1, sizeof(s1),
                           velri_div_ceil_u32(append_cols, 256U), aa.active_t, 1U);

        int s2_pipe_id =
            (aa.active_t == 1U) ? plan.stage2_decode_pipe_id : plan.stage2_prefill_pipe_id;
        VkPipeline pipe2 = velri_vk_get_pipe_locked(dev, s2_pipe_id);
        if (!pipe2) {
            return VELRI_ERR_UNSUPPORTED;
        }
        vk_attention_args s2 = {
            .q = ib[0],
            .k_cache = ob[1],
            .v_cache = ob[2],
            .out = ob[0],
            .rows = aa.active_t,
            .n_heads = nh,
            .n_kv_heads = nkvh,
            .d_k = aa.d_k,
            .d_v = aa.d_v,
            .max_ctx = aa.max_ctx,
            .kv_pos = aa.kv_pos,
            .q_stride = (uint32_t)op->inputs[0].strides[0],
            .o_stride = (uint32_t)op->outputs[0].strides[0],
            .k_head_stride = (uint32_t)op->outputs[1].strides[0],
            .k_row_stride = (uint32_t)op->outputs[1].strides[1],
            .v_head_stride = (uint32_t)op->outputs[2].strides[0],
            .v_row_stride = (uint32_t)op->outputs[2].strides[1],
            .sliding_window = op->raw.attr.attention.sliding_window,
            .causal = op->raw.attr.attention.causal ? 1U : 0U,
            .scale = aa.scale,
        };
        uint32_t gx2 = (s2_pipe_id == PIPE_ATTENTION_DECODE_F32) ? nh : aa.active_t;
        uint32_t gy2 = (s2_pipe_id == PIPE_ATTENTION_DECODE_F32) ? 1U : nh;
        set_dispatch_stage(&out->stages[1], pipe2, &s2, sizeof(s2), gx2, gy2, 1U);
        out->num_stages = 2U;
        break;
    }

    case VELRI_OP_COPY_LAST_ROW: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t dim = (uint32_t)op->inputs[0].dims[1];
        vk_copy_last_row_args args = {
            .x = ib[0],
            .y = ob[0],
            .src_row = rows - 1U,
            .dim = dim,
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(dim), 1U,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_BATCHED_LINEAR: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t nh = op->raw.attr.batched_linear.n_heads;
        uint32_t ng = op->raw.attr.batched_linear.n_groups;
        uint32_t in_dim = op->raw.attr.batched_linear.in_dim;
        uint32_t out_dim = op->raw.attr.batched_linear.out_dim;
        bool has_bias = (op->num_inputs >= 3U && op->input_present[2]);
        vk_batched_linear_args args = {
            .x = ib[0],
            .w = ib[1],
            .bias = has_bias ? ib[2] : 0ULL,
            .y = ob[0],
            .rows = rows,
            .n_heads = nh,
            .n_groups = ng,
            .in_dim = in_dim,
            .out_dim = out_dim,
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .w_stride = (uint32_t)op->inputs[1].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .flags = has_bias ? VELRI_EP_FLAG_BIAS : 0U,
            .pad = 0U,
        };
        uint32_t gx = op->raw.attr.batched_linear.weight_transposed
                          ? velri_div_ceil_u32(out_dim, 256U)
                          : velri_div_ceil_u32(out_dim, 4U);
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), gx, rows, nh);
        out->num_stages = 1U;
        break;
    }

    default:
        return VELRI_ERR_UNSUPPORTED;
    }

    for (uint32_t s = 0U; s < out->num_stages; ++s) {
        const vk_dispatch_stage *ds = &out->stages[s];
        if (!grid_fits(dev, ds->gx, ds->gy, ds->gz)) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    return VELRI_OK;
}

static void release_slot_locked(velri_backend_device *dev, struct velri_completion *slot) {
    if (slot->uses_splitk) {
        dev->splitk_busy = false;
        slot->uses_splitk = false;
    }
    slot->in_use = false;
    velri_os_cond_broadcast(dev->slot_cv);
}

/* ---- 5. Execution Backend Contract ---- */

static velri_status vk_submit_batch(velri_backend_device *dev, const velri_submit_item *items,
                                    uint32_t count, velri_completion **out_completion) {
    if (!out_completion) {
        return VELRI_ERR_INVALID;
    }
    *out_completion = nullptr;
    if (!dev || !items || count == 0U || count > VELRI_MAX_SUBMIT_BATCH) {
        return VELRI_ERR_INVALID;
    }

    vk_encoded_dispatch enc[VELRI_MAX_SUBMIT_BATCH];
    bool batch_uses_splitk = false;

    velri_os_mutex_lock(dev->mtx);

    /* Validate and encode every item before touching GPU state (pre-accept safety). */
    for (uint32_t i = 0U; i < count; ++i) {
        velri_status st = encode_item_locked(dev, &items[i], &enc[i]);
        if (st != VELRI_OK) {
            velri_os_mutex_unlock(dev->mtx);
            return st;
        }
        if (enc[i].uses_splitk) {
            batch_uses_splitk = true;
        }
    }

    /* Acquire a free ring slot (and split-K exclusivity if needed). */
    struct velri_completion *slot = nullptr;
    for (;;) {
        if (!batch_uses_splitk || !dev->splitk_busy) {
            for (uint32_t i = 0U; i < VK_CMD_RING_SIZE; ++i) {
                if (!dev->ring[i].in_use) {
                    slot = &dev->ring[i];
                    break;
                }
            }
            if (slot) {
                break;
            }
        }
        velri_os_cond_wait(dev->slot_cv, dev->mtx);
    }
    slot->in_use = true;
    slot->uses_splitk = batch_uses_splitk;
    if (batch_uses_splitk) {
        dev->splitk_busy = true;
        /*
         * Re-verify and re-patch Split-K BDAs after `cond_wait` and acquiring
         * `dev->splitk_busy`: another thread's `velri_vk_prepare_op` may have resized
         * (and freed the previous) `dev->splitk_buf` while we were waiting.
         */
        for (uint32_t i = 0U; i < count; ++i) {
            if (!enc[i].uses_splitk) {
                continue;
            }
            if (!dev->splitk_buf || dev->splitk_buf->size < enc[i].splitk_bytes) {
                release_slot_locked(dev, slot);
                velri_os_mutex_unlock(dev->mtx);
                return VELRI_ERR_EXEC;
            }
            uint64_t splitk_bda = (uint64_t)dev->splitk_buf->bda;
            vk_linear_args lin_args;
            memcpy(&lin_args, enc[i].stages[0].pc, sizeof(lin_args));
            lin_args.y = splitk_bda;
            memcpy(enc[i].stages[0].pc, &lin_args, sizeof(lin_args));

            vk_splitk_reduce_args red_args;
            memcpy(&red_args, enc[i].stages[1].pc, sizeof(red_args));
            red_args.partials = splitk_bda;
            memcpy(enc[i].stages[1].pc, &red_args, sizeof(red_args));
        }
    }

    VkMemoryBarrier mbar = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };

    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    velri_status st = VELRI_OK;
    if (dev->vkResetCommandBuffer(slot->cmd, 0U) != VK_SUCCESS ||
        dev->vkBeginCommandBuffer(slot->cmd, &bi) != VK_SUCCESS) {
        st = VELRI_ERR_EXEC;
    } else {
        bool need_barrier = false;
        for (uint32_t i = 0U; i < count; ++i) {
            for (uint32_t s = 0U; s < enc[i].num_stages; ++s) {
                const vk_dispatch_stage *ds = &enc[i].stages[s];
                if (need_barrier) {
                    dev->vkCmdPipelineBarrier(slot->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U, &mbar,
                                              0U, nullptr, 0U, nullptr);
                }
                dev->vkCmdBindPipeline(slot->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ds->pipe);
                dev->vkCmdPushConstants(slot->cmd, dev->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                        0U, ds->pc_size, ds->pc);
                dev->vkCmdDispatch(slot->cmd, ds->gx, ds->gy, ds->gz);
                need_barrier = true;
            }
        }
        if (dev->vkEndCommandBuffer(slot->cmd) != VK_SUCCESS ||
            dev->vkResetFences(dev->device, 1U, &slot->fence) != VK_SUCCESS) {
            st = VELRI_ERR_EXEC;
        } else {
            VkSubmitInfo si = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1U,
                .pCommandBuffers = &slot->cmd,
            };
            if (dev->vkQueueSubmit(dev->queue, 1U, &si, slot->fence) != VK_SUCCESS) {
                st = VELRI_ERR_EXEC;
            }
        }
    }

    if (st != VELRI_OK) {
        release_slot_locked(dev, slot);
        velri_os_mutex_unlock(dev->mtx);
        return st;
    }

    *out_completion = slot;
    velri_os_mutex_unlock(dev->mtx);
    return VELRI_OK;
}

static velri_status vk_submit(velri_backend_device *dev, const velri_submit_item *item,
                              velri_completion **out_completion) {
    return vk_submit_batch(dev, item, 1U, out_completion);
}

/* ---- 6. Completion Backend Contract ---- */

static velri_status vk_wait(velri_backend_device *dev, velri_completion *comp) {
    if (!dev || !comp) {
        return VELRI_ERR_INVALID;
    }
    /* Wait without holding `dev->mtx` so other threads can prepare/submit concurrently. */
    VkResult res = dev->vkWaitForFences(dev->device, 1U, &comp->fence, VK_TRUE, UINT64_MAX);
    return (res == VK_SUCCESS) ? VELRI_OK : VELRI_ERR_EXEC;
}

static void vk_destroy_completion(velri_backend_device *dev, velri_completion *comp) {
    if (!dev || !comp) {
        return;
    }
    velri_os_mutex_lock(dev->mtx);
    release_slot_locked(dev, comp);
    velri_os_mutex_unlock(dev->mtx);
}

static const velri_backend_vtable g_vk_vtable = {
    .name = "vulkan",
    .contract_version = VELRI_BACKEND_CONTRACT_VERSION,
    .enumerate_devices = velri_vk_enumerate_devices,
    .open_device = velri_vk_open_device,
    .query_caps = velri_vk_query_caps,
    .close_device = velri_vk_close_device,
    .alloc = velri_vk_alloc,
    .free = velri_vk_free,
    .copy_to_ram = velri_vk_copy_to_ram,
    .copy_from_ram = velri_vk_copy_from_ram,
    .zero_mem = velri_vk_zero_mem,
    .query_operation = velri_vk_query_operation,
    .prepare_op = velri_vk_prepare_op,
    .destroy_prepared_op = velri_vk_destroy_prepared_op,
    .submit = vk_submit,
    .submit_batch = vk_submit_batch,
    .wait = vk_wait,
    .destroy_completion = vk_destroy_completion,
};

const velri_backend_vtable *velri_vulkan_backend_vtable(void) { return &g_vk_vtable; }

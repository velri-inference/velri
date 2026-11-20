#include "internal.h"
#include <stdatomic.h>

/* Static ranges remove queue traffic for small weight sets. Streaming work
 * retains a private queue per worker, grouped by L3 domain, with work stealing. */
typedef struct {
    alignas(64) atomic_size_t next;
    size_t end;
} lin_tile_queue;

typedef struct {
    const velri_linear_typed_plan *plan;
    const velri_linear_run_args *args;
    velri_lin_slice_fn fn;
    size_t tiles, domains; /* domains == 0 selects static ranges */
    size_t domain_start[65];
    atomic_size_t domain_claim[64];
    lin_tile_queue queue[256];
} lin_gemv_ctx;

static void drain_tiles(lin_gemv_ctx *c, size_t qid, void *scratch) {
    lin_tile_queue *q = &c->queue[qid];
    while (atomic_load_explicit(&q->next, memory_order_relaxed) < q->end) {
        size_t tile = atomic_fetch_add_explicit(&q->next, 1, memory_order_relaxed);
        if (tile >= q->end)
            break;
        size_t last = tile + 1 == c->tiles ? c->plan->m : (tile + 1) * c->plan->columns;
        c->fn(c->plan, c->args, 0, c->args->rows, tile * c->plan->columns, last, scratch);
    }
}

static void lin_gemv_worker(void *vctx, size_t wid, size_t nw, void *scratch,
                            size_t scratch_bytes) {
    (void)scratch_bytes;
    lin_gemv_ctx *c = vctx;
    if (!c->domains) {
        size_t each = c->tiles / nw, extra = c->tiles % nw;
        size_t first = wid * each + (wid < extra ? wid : extra);
        size_t end = first + each + (wid < extra);
        size_t last = end == c->tiles ? c->plan->m : end * c->plan->columns;
        if (first < end)
            c->fn(c->plan, c->args, 0, c->args->rows, first * c->plan->columns, last, scratch);
        return;
    }
    size_t first = 0, end = nw, qid = wid;
    if (c->domains > 1) {
        int live = velri_current_l3_domain();
        size_t d = live >= 0 && (size_t)live < c->domains ? (size_t)live : wid % c->domains;
        first = c->domain_start[d];
        end = c->domain_start[d + 1];
        size_t slot = atomic_fetch_add_explicit(&c->domain_claim[d], 1, memory_order_relaxed);
        qid = first + slot < end ? first + slot : SIZE_MAX;
    }
    if (qid < nw)
        drain_tiles(c, qid, scratch);
    size_t count = end - first;
    size_t start = qid >= first && qid < end ? qid - first : wid % count;
    for (size_t off = 0; off < count; ++off)
        drain_tiles(c, first + (start + off) % count, scratch);
    for (size_t off = 0; off < nw; ++off) {
        size_t victim = (wid + off) % nw;
        if (victim < first || victim >= end)
            drain_tiles(c, victim, scratch);
    }
}

/* GEMM uses coarse jobs first, then smaller jobs to absorb completion skew.
 * Workers prefer their current L3 domain and steal remaining work from others.
 * Cache discovery is a performance hint; it never changes the computed output. */
typedef struct {
    alignas(64) atomic_size_t p1_cur;
    size_t p1_end;
    size_t p2_end;
    size_t t0;
    size_t t1;
    size_t n_jc1;
    size_t n_jc2;
    alignas(64) atomic_size_t p2_cur;
} lin_dom_queue;

typedef struct {
    const velri_linear_typed_plan *plan;
    const velri_linear_run_args *args;
    velri_lin_slice_fn fn;
    size_t active_nw;
    size_t doms;
    size_t mc1;
    size_t tiles1;
    size_t n_rc1;
    size_t r_split;
    size_t mc2;
    size_t tiles2;
    size_t n_rc2;
    bool rc_outer;
    lin_dom_queue dom[64];
} lin_gemm_ctx;

static inline void exec_gemm_p1(const lin_gemm_ctx *ctx, const lin_dom_queue *dq, size_t job,
                                void *w_scratch) {
    size_t rc = ctx->rc_outer ? (job / dq->n_jc1) : (job % ctx->n_rc1);
    size_t jc = ctx->rc_outer ? (job % dq->n_jc1) : (job / ctx->n_rc1);

    size_t r0 = rc * ctx->mc1;
    size_t r1 = r0 + ctx->mc1;
    size_t ta = dq->t0 + jc * ctx->tiles1;
    size_t tb = ta + ctx->tiles1 < dq->t1 ? ta + ctx->tiles1 : dq->t1;
    size_t C = ctx->plan->columns;
    size_t c0 = ta * C;
    size_t c1 = tb * C < ctx->plan->m ? tb * C : ctx->plan->m;

    if (r0 < r1 && c0 < c1) {
        ctx->fn(ctx->plan, ctx->args, r0, r1, c0, c1, w_scratch);
    }
}

static inline void exec_gemm_p2(const lin_gemm_ctx *ctx, const lin_dom_queue *dq, size_t job,
                                void *w_scratch) {
    size_t rc = ctx->rc_outer ? (job / dq->n_jc2) : (job % ctx->n_rc2);
    size_t jc = ctx->rc_outer ? (job % dq->n_jc2) : (job / ctx->n_rc2);

    size_t r0 = ctx->r_split + rc * ctx->mc2;
    size_t r1 = r0 + ctx->mc2 < ctx->args->rows ? r0 + ctx->mc2 : ctx->args->rows;
    size_t ta = dq->t0 + jc * ctx->tiles2;
    size_t tb = ta + ctx->tiles2 < dq->t1 ? ta + ctx->tiles2 : dq->t1;
    size_t C = ctx->plan->columns;
    size_t c0 = ta * C;
    size_t c1 = tb * C < ctx->plan->m ? tb * C : ctx->plan->m;

    if (r0 < r1 && c0 < c1) {
        ctx->fn(ctx->plan, ctx->args, r0, r1, c0, c1, w_scratch);
    }
}

static inline size_t worker_live_dom(size_t doms, size_t wid) {
    if (doms <= 1) {
        return 0;
    }
    int live = velri_current_l3_domain();
    return (live >= 0 && (size_t)live < doms) ? (size_t)live : (wid % doms);
}

static void lin_gemm_worker(void *vctx, size_t wid, size_t nw, void *w_scratch,
                            size_t w_scratch_bytes) {
    (void)nw;
    (void)w_scratch_bytes;
    lin_gemm_ctx *ctx = (lin_gemm_ctx *)vctx;
    if (wid >= ctx->active_nw) {
        return;
    }
    const size_t doms = ctx->doms;

    /* 1. Drain local L3-domain Phase-1 coarse blocks (re-checking live CCD on each claim) */
    for (;;) {
        size_t d = worker_live_dom(doms, wid);
        lin_dom_queue *dq = &ctx->dom[d];
        if (atomic_load_explicit(&dq->p1_cur, memory_order_relaxed) >= dq->p1_end) {
            break;
        }
        size_t job = atomic_fetch_add_explicit(&dq->p1_cur, 1, memory_order_relaxed);
        if (job >= dq->p1_end) {
            break;
        }
        exec_gemm_p1(ctx, dq, job, w_scratch);
    }

    /* 2. Steal any remaining Phase-1 coarse blocks across sibling L3 domains before Phase 2 */
    if (doms > 1) {
        size_t base_d = worker_live_dom(doms, wid);
        for (size_t off = 1; off < doms; ++off) {
            size_t vic = (base_d + off) % doms;
            lin_dom_queue *vq = &ctx->dom[vic];
            while (atomic_load_explicit(&vq->p1_cur, memory_order_relaxed) < vq->p1_end) {
                size_t job = atomic_fetch_add_explicit(&vq->p1_cur, 1, memory_order_relaxed);
                if (job >= vq->p1_end) {
                    break;
                }
                exec_gemm_p1(ctx, vq, job, w_scratch);
            }
        }
    }

    /* 3. Drain local fine blocks. */
    for (;;) {
        size_t d = worker_live_dom(doms, wid);
        lin_dom_queue *dq = &ctx->dom[d];
        if (atomic_load_explicit(&dq->p2_cur, memory_order_relaxed) >= dq->p2_end) {
            break;
        }
        size_t job = atomic_fetch_add_explicit(&dq->p2_cur, 1, memory_order_relaxed);
        if (job >= dq->p2_end) {
            break;
        }
        exec_gemm_p2(ctx, dq, job, w_scratch);
    }

    /* 4. Steal remaining fine blocks across L3 domains. */
    size_t base_d = worker_live_dom(doms, wid);
    for (size_t off = 0; off < doms; ++off) {
        size_t vic = (base_d + off) % doms;
        lin_dom_queue *vq = &ctx->dom[vic];
        while (atomic_load_explicit(&vq->p2_cur, memory_order_relaxed) < vq->p2_end) {
            size_t job = atomic_fetch_add_explicit(&vq->p2_cur, 1, memory_order_relaxed);
            if (job >= vq->p2_end) {
                break;
            }
            exec_gemm_p2(ctx, vq, job, w_scratch);
        }
    }
}

velri_status velri_linear_typed_run_parallel(const velri_linear_typed_plan *plan,
                                             const velri_linear_run_args *args, velri_pool *pool,
                                             size_t active_workers) {
    bool use_pool = (pool != nullptr && active_workers > 0);
    velri_status s = velri_linear_validate_run(plan, args, !use_pool);
    if (s != VELRI_OK || args->rows == 0) {
        return s;
    }
    if (use_pool && args->rows >= 3 && velri_pool_scratch_bytes(pool) < plan->scratch_bytes) {
        return VELRI_ERR_INVALID;
    }

    const velri_lin_impl *impl = (const velri_lin_impl *)plan->impl;
    velri_lin_slice_fn fn = plan->prepared_weights ? impl->run_prepared : impl->run_rowmajor;

    if (!use_pool || active_workers == 1 || velri_pool_num_workers(pool) == 1) {
        void *sc = use_pool ? velri_pool_worker0_scratch(pool) : args->scratch;
        fn(plan, args, 0, args->rows, 0, plan->m, sc);
        return VELRI_OK;
    }

    size_t nw = active_workers < velri_pool_num_workers(pool) ? active_workers
                                                              : velri_pool_num_workers(pool);
    if (nw > 256) {
        nw = 256;
    }
    size_t num_tiles = (plan->m - 1) / plan->columns + 1;
    size_t l3_doms = (size_t)velri_get_cpu_info()->num_l3_domains;
    if (l3_doms < 1 || l3_doms > 64) {
        l3_doms = 1;
    }

    /* Path A: Decode GEMV / Small-Batch (`rows <= 16`) -> 1D contiguous tile split */
    if (args->rows <= 16) {
        if (num_tiles <= 1) {
            fn(plan, args, 0, args->rows, 0, plan->m, velri_pool_worker0_scratch(pool));
            return VELRI_OK;
        }
        size_t active_nw = nw < num_tiles ? nw : num_tiles;
        size_t weight_bytes = plan->m * plan->k * velri_dtype_bytes(plan->desc.w_dtype);
        lin_gemv_ctx ctx;
        ctx.plan = plan;
        ctx.args = args;
        ctx.fn = fn;
        ctx.tiles = num_tiles;
        ctx.domains = weight_bytes > velri_l3_domain_bytes() / 4
                          ? (l3_doms > 1 && active_nw >= 2 * l3_doms ? l3_doms : 1)
                          : 0;
        if (ctx.domains) {
            size_t each = active_nw / ctx.domains, extra = active_nw % ctx.domains;
            for (size_t d = 0; d <= ctx.domains; ++d) {
                ctx.domain_start[d] = d * each + (d < extra ? d : extra);
                if (d < ctx.domains)
                    atomic_init(&ctx.domain_claim[d], 0);
            }
            for (size_t i = 0; i < active_nw; ++i) {
                each = num_tiles / active_nw;
                extra = num_tiles % active_nw;
                size_t first = i * each + (i < extra ? i : extra);
                atomic_init(&ctx.queue[i].next, first);
                ctx.queue[i].end = first + each + (i < extra);
            }
        }
        velri_pool_parallel(pool, active_nw, lin_gemv_worker, &ctx);
        return VELRI_OK;
    }

    /* GEMM: implementation-owned job sizes, refined for the available work. */
    const velri_lin_blocking *blocking = plan->prepared_weights ? impl->prepared : impl->rowmajor;
    size_t mr_unit = blocking->row_grain;
    size_t mc1 = blocking->coarse_rows, tiles1 = blocking->coarse_tiles;
    size_t mc2 = blocking->fine_rows, tiles2 = blocking->fine_tiles;

    while (tiles1 > 1 && num_tiles < tiles1) {
        tiles1 >>= 1;
    }
    size_t min_mc1 = plan->prepared_weights ? (2 * mr_unit) : (4 * mr_unit);
    while (((num_tiles + tiles1 - 1) / tiles1) * ((args->rows + mc1 - 1) / mc1) < nw * 2) {
        if (mc1 >= 2 * min_mc1 && args->rows < mc1 * 4) {
            mc1 = (mc1 / (2 * mr_unit)) * mr_unit;
        } else if (tiles1 > 1) {
            tiles1 >>= 1;
        } else if (mc1 >= 2 * mr_unit) {
            mc1 = (mc1 / (2 * mr_unit)) * mr_unit;
        } else {
            break;
        }
    }

    size_t full_rc1 = args->rows / mc1;
    size_t n_rc1 = 0;
    size_t r_split = 0;
    if (full_rc1 >= 2) {
        n_rc1 = (full_rc1 * 3) / 4;
        if (n_rc1 == 0) {
            n_rc1 = 1;
        }
        r_split = n_rc1 * mc1;
    } else {
        mc2 = mr_unit;
        tiles2 = 1;
    }
    if (mc2 > mc1) {
        mc2 = mc1;
    }
    if (tiles2 > tiles1) {
        tiles2 = tiles1;
    }
    while (tiles2 > 1 && num_tiles < tiles2) {
        tiles2 >>= 1;
    }

    size_t total_jc1 = (num_tiles + tiles1 - 1) / tiles1;
    size_t doms = (l3_doms > 1 && nw >= l3_doms * 2 && total_jc1 >= l3_doms) ? l3_doms : 1;

    size_t w_dom_bytes =
        ((plan->m + doms - 1) / doms) * plan->k * velri_dtype_bytes(plan->desc.w_dtype);
    size_t x_band_bytes = mc1 * plan->k * velri_dtype_bytes(plan->desc.x_dtype);
    size_t l3_cap = velri_l3_domain_bytes();
    bool rc_outer = (w_dom_bytes + x_band_bytes <= (l3_cap * 3) / 4);

    size_t rem_rows = args->rows - r_split;
    size_t n_rc2 = rem_rows > 0 ? ((rem_rows + mc2 - 1) / mc2) : 0;

    lin_gemm_ctx ctx = {
        .plan = plan,
        .args = args,
        .fn = fn,
        .doms = doms,
        .mc1 = mc1,
        .tiles1 = tiles1,
        .n_rc1 = n_rc1,
        .r_split = r_split,
        .mc2 = mc2,
        .tiles2 = tiles2,
        .n_rc2 = n_rc2,
        .rc_outer = rc_outer,
    };

    size_t base_jc = total_jc1 / doms;
    size_t rem_jc = total_jc1 % doms;
    size_t off_jc = 0;
    size_t total_jobs = 0;
    for (size_t d = 0; d < doms; ++d) {
        size_t cnt_jc = base_jc + (d < rem_jc ? 1 : 0);
        size_t t0 = off_jc * tiles1;
        size_t t1 = (off_jc + cnt_jc) * tiles1;
        if (t1 > num_tiles) {
            t1 = num_tiles;
        }
        off_jc += cnt_jc;

        size_t dtiles = (t1 > t0) ? (t1 - t0) : 0;
        size_t n_jc1_d = dtiles > 0 ? ((dtiles + tiles1 - 1) / tiles1) : 0;
        size_t n_jc2_d = dtiles > 0 ? ((dtiles + tiles2 - 1) / tiles2) : 0;
        size_t p1_end = n_rc1 * n_jc1_d;
        size_t p2_end = n_rc2 * n_jc2_d;

        ctx.dom[d].t0 = t0;
        ctx.dom[d].t1 = t1;
        ctx.dom[d].n_jc1 = n_jc1_d;
        ctx.dom[d].n_jc2 = n_jc2_d;
        ctx.dom[d].p1_end = p1_end;
        ctx.dom[d].p2_end = p2_end;
        atomic_init(&ctx.dom[d].p1_cur, 0);
        atomic_init(&ctx.dom[d].p2_cur, 0);
        total_jobs += p1_end + p2_end;
    }

    if (total_jobs <= 1) {
        fn(plan, args, 0, args->rows, 0, plan->m, velri_pool_worker0_scratch(pool));
        return VELRI_OK;
    }

    ctx.active_nw = nw < total_jobs ? nw : total_jobs;
    velri_pool_parallel(pool, ctx.active_nw, lin_gemm_worker, &ctx);
    return VELRI_OK;
}

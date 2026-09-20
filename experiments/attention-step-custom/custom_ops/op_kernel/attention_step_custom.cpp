#include "kernel_operator.h"

using namespace AscendC;

// Keep the online-softmax state in Vector local memory.  This mirrors the
// all-vector IncreFlashAttention path in cann-ops-adv, but is intentionally
// restricted to the single-query-token decode shape used by llama.cpp.
constexpr SoftmaxConfig ATTENTION_SOFTMAX_FLASHV2_CFG = {false};

// The IFA source keeps these helpers in ifa_public_define.h.  For our single
// decode row, the 32-byte softmax state block contains the same fp32 scalar in
// all eight lanes, so src1BlkStride=0 broadcasts it over the output row.
template <bool DIVIDE>
__aicore__ inline void ApplySoftmaxStateToRow(LocalTensor<float> dst,
                                               LocalTensor<float> src,
                                               LocalTensor<float> state,
                                               uint32_t count) {
    constexpr uint32_t ELEMENTS_PER_REPEAT = 64;
    BinaryRepeatParams params;
    params.dstBlkStride = 1;
    params.src0BlkStride = 1;
    params.src1BlkStride = 0;
    params.dstRepStride = 8;
    params.src0RepStride = 8;
    params.src1RepStride = 0;

    const uint32_t repeats = count / ELEMENTS_PER_REPEAT;
    const uint32_t tail = count % ELEMENTS_PER_REPEAT;
    if (repeats != 0) {
        if constexpr (DIVIDE) {
            Div(dst, src, state, ELEMENTS_PER_REPEAT, repeats, params);
        } else {
            Mul(dst, src, state, ELEMENTS_PER_REPEAT, repeats, params);
        }
    }
    if (tail != 0) {
        const uint32_t offset = repeats * ELEMENTS_PER_REPEAT;
        if constexpr (DIVIDE) {
            Div(dst[offset], src[offset], state, tail, 1, params);
        } else {
            Mul(dst[offset], src[offset], state, tail, 1, params);
        }
    }
}

// One block per q-head. Each block handles one full attention head's
// q · K^T → softmax → · V for the current decode token (S_q = 1).
//
// Inputs (BSH-style memory, head_dim D fixed at compile-time via tiling):
//   query:   fp16 [num_q_heads * D]                — flat q for this token
//   k_cache: fp16 [max_seq, num_kv_heads * D]      — rows [0, context) valid
//   v_cache: fp16 [max_seq, num_kv_heads * D]      — rows [0, context) valid
//   actual_seq_lengths_kv: int32 [1]               — effective KV boundary
// Output:
//   out:     fp16 [num_q_heads * D]                — flat attn output
//
// Tiling supplies context, num_q_heads, num_kv_heads, head_dim, max_seq,
// and the inverse-sqrt scale.

class KernelAttentionStepCustom {
public:
    __aicore__ inline KernelAttentionStepCustom() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR mask,
                                GM_ADDR actual_seq_lengths_kv, GM_ADDR out,
                                uint32_t context, uint32_t numQHeads, uint32_t numKvHeads,
                                uint32_t headDim, uint32_t maxSeq, float scale) {
        this->context     = context;
        this->numQHeads   = numQHeads;
        this->numKvHeads  = numKvHeads;
        this->headDim     = headDim;
        this->maxSeq      = maxSeq;
        this->kvDim       = numKvHeads * headDim;
        this->qPerKv      = numQHeads / numKvHeads;
        this->scale       = scale;

        const int32_t bIdx = static_cast<int32_t>(GetBlockIdx());
        this->qHead  = bIdx;
        this->kvHead = bIdx / static_cast<int32_t>(qPerKv);

        qGm.SetGlobalBuffer((__gm__ half*)query + qHead * headDim, headDim);
        kGm.SetGlobalBuffer((__gm__ half*)k_cache, static_cast<uint64_t>(maxSeq) * kvDim);
        vGm.SetGlobalBuffer((__gm__ half*)v_cache, static_cast<uint64_t>(maxSeq) * kvDim);
        maskGm.SetGlobalBuffer((__gm__ half*)mask, maxSeq);
        actualSeqLengthsKvGm.SetGlobalBuffer((__gm__ int32_t*)actual_seq_lengths_kv, 1);
        outGm.SetGlobalBuffer((__gm__ half*)out + qHead * headDim, headDim);

        pipe.InitBuffer(qFp16Buf,  headDim * sizeof(half));
        pipe.InitBuffer(qFp32Buf,  headDim * sizeof(float));
        // Match the open-source IFA kernel's KV staging pattern: two VECIN
        // queue slots let MTE2 fill one tile while Vector consumes the other,
        // and TQue owns the cross-pipeline reuse events.
        pipe.InitBuffer(kvFp16Queue, 2, TILE_CONTEXT * headDim * sizeof(half));
        pipe.InitBuffer(kvFp32Buf, TILE_CONTEXT * headDim * sizeof(float));
        pipe.InitBuffer(prodBuf,   TILE_CONTEXT * sizeof(float));
        pipe.InitBuffer(outFp32Buf, headDim * sizeof(float));
        pipe.InitBuffer(outFp16Buf, headDim * sizeof(half));
        pipe.InitBuffer(maskFp16Buf, TILE_CONTEXT * sizeof(half));
        pipe.InitBuffer(weightsBuf, TILE_CONTEXT * 8 * sizeof(float));
        // SoftmaxFlashV2 uses one 32-byte block for each persistent online
        // state.  The temporary buffer size follows the official IFA allvec
        // implementation and is reused for every KV tile.
        pipe.InitBuffer(softmaxMaxBuf, 32);
        pipe.InitBuffer(softmaxSumBuf, 32);
        pipe.InitBuffer(softmaxExpBuf, 32);
        pipe.InitBuffer(softmaxTmpBuf, 32 * 1024);
    }

    __aicore__ inline void Process() {
        const int32_t actualContext = actualSeqLengthsKvGm.GetValue(0);
        if (actualContext > 0 && actualContext < static_cast<int32_t>(context)) {
            context = static_cast<uint32_t>(actualContext);
        }

        LocalTensor<half>  qFp16   = qFp16Buf.Get<half>();
        LocalTensor<float> qFp32   = qFp32Buf.Get<float>();
        LocalTensor<float> kvFp32  = kvFp32Buf.Get<float>();
        LocalTensor<float> prod    = prodBuf.Get<float>();
        LocalTensor<float> outFp32 = outFp32Buf.Get<float>();
        LocalTensor<half>  outFp16 = outFp16Buf.Get<half>();
        LocalTensor<half>  maskFp16 = maskFp16Buf.Get<half>();
        LocalTensor<float> weights = weightsBuf.Get<float>();
        LocalTensor<float> softmaxMax = softmaxMaxBuf.Get<float>();
        LocalTensor<float> softmaxSum = softmaxSumBuf.Get<float>();
        LocalTensor<float> softmaxExp = softmaxExpBuf.Get<float>();
        LocalTensor<uint8_t> softmaxTmp = softmaxTmpBuf.Get<uint8_t>();
        LocalTensor<float> softmaxTmpFp32 = softmaxTmpBuf.Get<float>();

        DataCopy(qFp16, qGm[0], headDim);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(qFp32, qFp16, RoundMode::CAST_NONE, headDim);

        // Online decode attention, following the official IFA all-vector
        // structure.  Each tile computes QK, updates the running softmax
        // max/sum, rescales the old PV accumulator, and consumes the matching
        // V tile.  No full scores[context] tensor is materialized.
        Duplicate(outFp32, 0.0f, headDim);
        Duplicate(softmaxMax, -2.0e38f, 8);
        Duplicate(softmaxSum, 0.0f, 8);
        PipeBarrier<PIPE_V>();

        for (uint32_t tileStart = 0; tileStart < context; tileStart += TILE_CONTEXT) {
            const uint32_t remaining = context - tileStart;
            const uint32_t tileSize = remaining < TILE_CONTEXT ? remaining : TILE_CONTEXT;
            const uint32_t tileSizeAlign = (tileSize + 7U) & ~7U;
            const uint64_t kOff = static_cast<uint64_t>(tileStart) * kvDim + kvHead * headDim;
            const DataCopyParams keyCopyParams {
                static_cast<uint16_t>(tileSize),
                static_cast<uint16_t>(headDim * sizeof(half) / 32),
                static_cast<uint16_t>((kvDim - headDim) * sizeof(half) / 32),
                0
            };
            LocalTensor<half> keyFp16 = kvFp16Queue.AllocTensor<half>();
            DataCopy(keyFp16, kGm[kOff], keyCopyParams);
            kvFp16Queue.EnQue(keyFp16);
            keyFp16 = kvFp16Queue.DeQue<half>();
            Cast(kvFp32, keyFp16, RoundMode::CAST_NONE, tileSize * headDim);
            kvFp16Queue.FreeTensor(keyFp16);
            PipeBarrier<PIPE_V>();
            BinaryRepeatParams dotParams;
            dotParams.dstBlkStride = 1;
            dotParams.src0BlkStride = 1;
            dotParams.src1BlkStride = 1;
            dotParams.dstRepStride = headDim / 8;
            dotParams.src0RepStride = 0;
            dotParams.src1RepStride = headDim / 8;
            Mul(kvFp32, qFp32, kvFp32, headDim, tileSize, dotParams);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(prod, kvFp32, headDim, tileSize, 1, 1, headDim / 8);
            PipeBarrier<PIPE_V>();
            Muls(prod, prod, scale, tileSize);
            PipeBarrier<PIPE_V>();

            const DataCopyParams maskCopyParams {
                1, static_cast<uint16_t>(tileSize * sizeof(half)), 0, 0
            };
            const DataCopyPadParams maskPadParams {false, 0, 0, 0};
            DataCopyPad(maskFp16, maskGm[tileStart], maskCopyParams, maskPadParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(kvFp32, maskFp16, RoundMode::CAST_NONE, tileSize);
            PipeBarrier<PIPE_V>();
            Add(prod, prod, kvFp32, tileSize);
            PipeBarrier<PIPE_V>();

            if (context <= SHORT_CONTEXT_THRESHOLD) {
                // A single KV tile does not need rolling online-softmax state.
                // The ordinary reduction path has lower fixed overhead for the
                // short decode contexts that dominate tg64.
                ReduceMax<float>(softmaxMax, prod, softmaxTmpFp32, tileSize);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const float maxScore = softmaxMax.GetValue(0);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Adds(prod, prod, -maxScore, tileSize);
                PipeBarrier<PIPE_V>();
                Exp(prod, prod, tileSize);
                PipeBarrier<PIPE_V>();
                ReduceSum<float>(softmaxSum, prod, softmaxTmpFp32, tileSize);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const float invSum = 1.0f / softmaxSum.GetValue(0);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Muls(prod, prod, invSum, tileSize);
                Duplicate(softmaxSum, 1.0f, 8);
                PipeBarrier<PIPE_V>();
            } else {
                SoftMaxShapeInfo softmaxShape = {1, tileSizeAlign, 1, tileSize};
                SoftMaxTiling softmaxTiling = SoftMaxFlashV2TilingFunc(
                    softmaxShape, sizeof(float), sizeof(float), softmaxTmp.GetSize(), true, false);
                SoftmaxFlashV2<float, true, true, false, false, ATTENTION_SOFTMAX_FLASHV2_CFG>(
                    prod, softmaxSum, softmaxMax, prod, softmaxExp,
                    softmaxSum, softmaxMax, softmaxTmp, softmaxTiling, softmaxShape);
                PipeBarrier<PIPE_V>();
            }

            if (tileStart != 0) {
                ApplySoftmaxStateToRow<false>(outFp32, outFp32, softmaxExp, headDim);
                PipeBarrier<PIPE_V>();
            }

            const uint64_t vOff = static_cast<uint64_t>(tileStart) * kvDim + kvHead * headDim;
            const DataCopyParams valueCopyParams {
                static_cast<uint16_t>(tileSize),
                static_cast<uint16_t>(headDim * sizeof(half) / 32),
                static_cast<uint16_t>((kvDim - headDim) * sizeof(half) / 32),
                0
            };
            LocalTensor<half> valueFp16 = kvFp16Queue.AllocTensor<half>();
            DataCopy(valueFp16, vGm[vOff], valueCopyParams);
            kvFp16Queue.EnQue(valueFp16);
            valueFp16 = kvFp16Queue.DeQue<half>();
            Cast(kvFp32, valueFp16, RoundMode::CAST_NONE, tileSize * headDim);
            kvFp16Queue.FreeTensor(valueFp16);
            PipeBarrier<PIPE_V>();

            Brcb(weights, prod, tileSizeAlign / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            BinaryRepeatParams valueParams;
            valueParams.dstBlkStride = 1;
            valueParams.src0BlkStride = 0;
            valueParams.src1BlkStride = 1;
            valueParams.dstRepStride = headDim / 8;
            valueParams.src0RepStride = 1;
            valueParams.src1RepStride = headDim / 8;
            Mul(kvFp32, weights, kvFp32, headDim, tileSize, valueParams);
            PipeBarrier<PIPE_V>();

            uint32_t rows = tileSize;
            while (rows > 1) {
                const uint32_t halfRows = rows / 2;
                Add(kvFp32, kvFp32, kvFp32[halfRows * headDim], halfRows * headDim);
                PipeBarrier<PIPE_V>();
                if ((rows & 1) != 0) {
                    Add(kvFp32, kvFp32, kvFp32[(rows - 1) * headDim], headDim);
                    PipeBarrier<PIPE_V>();
                }
                rows = halfRows;
            }
            Add(outFp32, outFp32, kvFp32, headDim);
            PipeBarrier<PIPE_V>();
        }

        // SoftmaxFlashV2 deliberately leaves the tile probabilities
        // unnormalised.  Divide once after all KV tiles have contributed.
        ApplySoftmaxStateToRow<true>(outFp32, outFp32, softmaxSum, headDim);
        PipeBarrier<PIPE_V>();

        Cast(outFp16, outFp32, RoundMode::CAST_RINT, headDim);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outGm[0], outFp16, headDim);
    }

private:
    // 240 stays below the 255-repeat hardware limit while making each queued
    // KV transfer large enough to amortize TQue bookkeeping and overlap well.
    static constexpr uint32_t TILE_CONTEXT = 240;
    // The ordinary one-tile softmax wins for very short decode contexts; above
    // this measured crossover, retain the online IFA path even within one tile.
    static constexpr uint32_t SHORT_CONTEXT_THRESHOLD = 64;

    uint32_t context;
    uint32_t numQHeads;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t kvDim;
    uint32_t qPerKv;
    uint32_t maxSeq;
    float    scale;
    int32_t  qHead;
    int32_t  kvHead;

    GlobalTensor<half> qGm;
    GlobalTensor<half> kGm;
    GlobalTensor<half> vGm;
    GlobalTensor<half> maskGm;
    GlobalTensor<int32_t> actualSeqLengthsKvGm;
    GlobalTensor<half> outGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> qFp16Buf;
    TBuf<TPosition::VECCALC> qFp32Buf;
    TQue<QuePosition::VECIN, 2> kvFp16Queue;
    TBuf<TPosition::VECCALC> kvFp32Buf;
    TBuf<TPosition::VECCALC> prodBuf;
    TBuf<TPosition::VECCALC> outFp32Buf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> maskFp16Buf;
    TBuf<TPosition::VECCALC> weightsBuf;
    TBuf<TPosition::VECCALC> softmaxMaxBuf;
    TBuf<TPosition::VECCALC> softmaxSumBuf;
    TBuf<TPosition::VECCALC> softmaxExpBuf;
    TBuf<TPosition::VECCALC> softmaxTmpBuf;
};

// Experimental GQA route: one logical block owns a group of Q heads which
// share one KV head. K and V are staged once per context tile and reused by
// every Q head in the group. The softmax state and output accumulator remain
// independent for every Q head.
class KernelAttentionStepGqaReuse {
public:
    __aicore__ inline KernelAttentionStepGqaReuse() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR mask,
                                GM_ADDR actual_seq_lengths_kv, GM_ADDR out,
                                uint32_t context, uint32_t numQHeads, uint32_t numKvHeads,
                                uint32_t headDim, uint32_t maxSeq, float scale,
                                uint32_t qHeadTile, uint32_t groupsPerKv) {
        this->context = context;
        this->numQHeads = numQHeads;
        this->numKvHeads = numKvHeads;
        this->headDim = headDim;
        this->maxSeq = maxSeq;
        this->kvDim = numKvHeads * headDim;
        this->scale = scale;
        this->qHeadTile = qHeadTile;
        this->groupsPerKv = groupsPerKv;

        const uint32_t blockIdx = GetBlockIdx();
        this->kvHead = blockIdx / groupsPerKv;
        const uint32_t groupInKv = blockIdx % groupsPerKv;
        const uint32_t qBegin = kvHead * numQHeads / numKvHeads;
        const uint32_t qEnd = (kvHead + 1U) * numQHeads / numKvHeads;
        this->qGroupBegin = qBegin + groupInKv * qHeadTile;
        const uint32_t proposedEnd = qGroupBegin + qHeadTile;
        this->qGroupEnd = proposedEnd < qEnd ? proposedEnd : qEnd;
        this->activeHeads = qGroupBegin < qGroupEnd ? qGroupEnd - qGroupBegin : 0U;

        qGm.SetGlobalBuffer((__gm__ half*)query, static_cast<uint64_t>(numQHeads) * headDim);
        kGm.SetGlobalBuffer((__gm__ half*)k_cache, static_cast<uint64_t>(maxSeq) * kvDim);
        vGm.SetGlobalBuffer((__gm__ half*)v_cache, static_cast<uint64_t>(maxSeq) * kvDim);
        maskGm.SetGlobalBuffer((__gm__ half*)mask, maxSeq);
        actualSeqLengthsKvGm.SetGlobalBuffer((__gm__ int32_t*)actual_seq_lengths_kv, 1);
        outGm.SetGlobalBuffer((__gm__ half*)out, static_cast<uint64_t>(numQHeads) * headDim);

        pipe.InitBuffer(qFp16Buf, MAX_Q_HEAD_TILE * headDim * sizeof(half));
        pipe.InitBuffer(qFp32Buf, MAX_Q_HEAD_TILE * headDim * sizeof(float));
        pipe.InitBuffer(kvFp16Queue, 2, TILE_CONTEXT * headDim * sizeof(half));
        pipe.InitBuffer(sharedKvFp32Buf, TILE_CONTEXT * headDim * sizeof(float));
        pipe.InitBuffer(workFp32Buf, TILE_CONTEXT * headDim * sizeof(float));
        pipe.InitBuffer(prodBuf, MAX_Q_HEAD_TILE * TILE_CONTEXT * sizeof(float));
        pipe.InitBuffer(outFp32Buf, MAX_Q_HEAD_TILE * headDim * sizeof(float));
        pipe.InitBuffer(outFp16Buf, MAX_Q_HEAD_TILE * headDim * sizeof(half));
        pipe.InitBuffer(maskFp16Buf, TILE_CONTEXT * sizeof(half));
        pipe.InitBuffer(maskFp32Buf, TILE_CONTEXT * sizeof(float));
        pipe.InitBuffer(weightsBuf, TILE_CONTEXT * 8 * sizeof(float));
        pipe.InitBuffer(softmaxMaxBuf, MAX_Q_HEAD_TILE * 32);
        pipe.InitBuffer(softmaxSumBuf, MAX_Q_HEAD_TILE * 32);
        pipe.InitBuffer(softmaxExpBuf, MAX_Q_HEAD_TILE * 32);
        pipe.InitBuffer(softmaxTmpBuf, 32 * 1024);
    }

    __aicore__ inline void Process() {
        if (activeHeads == 0U) {
            return;
        }

        const int32_t actualContext = actualSeqLengthsKvGm.GetValue(0);
        if (actualContext > 0 && actualContext < static_cast<int32_t>(context)) {
            context = static_cast<uint32_t>(actualContext);
        }

        LocalTensor<half> qFp16 = qFp16Buf.Get<half>();
        LocalTensor<float> qFp32 = qFp32Buf.Get<float>();
        LocalTensor<float> sharedKvFp32 = sharedKvFp32Buf.Get<float>();
        LocalTensor<float> workFp32 = workFp32Buf.Get<float>();
        LocalTensor<float> prod = prodBuf.Get<float>();
        LocalTensor<float> outFp32 = outFp32Buf.Get<float>();
        LocalTensor<half> outFp16 = outFp16Buf.Get<half>();
        LocalTensor<half> maskFp16 = maskFp16Buf.Get<half>();
        LocalTensor<float> maskFp32 = maskFp32Buf.Get<float>();
        LocalTensor<float> weights = weightsBuf.Get<float>();
        LocalTensor<float> softmaxMax = softmaxMaxBuf.Get<float>();
        LocalTensor<float> softmaxSum = softmaxSumBuf.Get<float>();
        LocalTensor<float> softmaxExp = softmaxExpBuf.Get<float>();
        LocalTensor<uint8_t> softmaxTmp = softmaxTmpBuf.Get<uint8_t>();
        LocalTensor<float> softmaxTmpFp32 = softmaxTmpBuf.Get<float>();

        DataCopy(qFp16, qGm[static_cast<uint64_t>(qGroupBegin) * headDim], activeHeads * headDim);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(qFp32, qFp16, RoundMode::CAST_NONE, activeHeads * headDim);
        Duplicate(outFp32, 0.0f, activeHeads * headDim);
        Duplicate(softmaxMax, -2.0e38f, activeHeads * 8U);
        Duplicate(softmaxSum, 0.0f, activeHeads * 8U);
        PipeBarrier<PIPE_V>();

        for (uint32_t tileStart = 0; tileStart < context; tileStart += TILE_CONTEXT) {
            const uint32_t remaining = context - tileStart;
            const uint32_t tileSize = remaining < TILE_CONTEXT ? remaining : TILE_CONTEXT;
            const uint32_t tileSizeAlign = (tileSize + 7U) & ~7U;

            const DataCopyParams kvCopyParams {
                static_cast<uint16_t>(tileSize),
                static_cast<uint16_t>(headDim * sizeof(half) / 32),
                static_cast<uint16_t>((kvDim - headDim) * sizeof(half) / 32),
                0
            };
            const uint64_t kOff = static_cast<uint64_t>(tileStart) * kvDim + kvHead * headDim;
            LocalTensor<half> kvFp16 = kvFp16Queue.AllocTensor<half>();
            DataCopy(kvFp16, kGm[kOff], kvCopyParams);
            kvFp16Queue.EnQue(kvFp16);
            kvFp16 = kvFp16Queue.DeQue<half>();
            Cast(sharedKvFp32, kvFp16, RoundMode::CAST_NONE, tileSize * headDim);
            kvFp16Queue.FreeTensor(kvFp16);
            PipeBarrier<PIPE_V>();

            const DataCopyParams maskCopyParams {
                1, static_cast<uint16_t>(tileSize * sizeof(half)), 0, 0
            };
            const DataCopyPadParams maskPadParams {false, 0, 0, 0};
            DataCopyPad(maskFp16, maskGm[tileStart], maskCopyParams, maskPadParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(maskFp32, maskFp16, RoundMode::CAST_NONE, tileSize);
            PipeBarrier<PIPE_V>();

            for (uint32_t localHead = 0; localHead < activeHeads; ++localHead) {
                LocalTensor<float> qHead = qFp32[localHead * headDim];
                LocalTensor<float> prodHead = prod[localHead * TILE_CONTEXT];
                LocalTensor<float> outHead = outFp32[localHead * headDim];
                LocalTensor<float> maxHead = softmaxMax[localHead * 8U];
                LocalTensor<float> sumHead = softmaxSum[localHead * 8U];
                LocalTensor<float> expHead = softmaxExp[localHead * 8U];

                BinaryRepeatParams dotParams;
                dotParams.dstBlkStride = 1;
                dotParams.src0BlkStride = 1;
                dotParams.src1BlkStride = 1;
                dotParams.dstRepStride = headDim / 8;
                dotParams.src0RepStride = 0;
                dotParams.src1RepStride = headDim / 8;
                Mul(workFp32, qHead, sharedKvFp32, headDim, tileSize, dotParams);
                PipeBarrier<PIPE_V>();
                WholeReduceSum(prodHead, workFp32, headDim, tileSize, 1, 1, headDim / 8);
                PipeBarrier<PIPE_V>();
                Muls(prodHead, prodHead, scale, tileSize);
                PipeBarrier<PIPE_V>();
                Add(prodHead, prodHead, maskFp32, tileSize);
                PipeBarrier<PIPE_V>();

                if (context <= SHORT_CONTEXT_THRESHOLD) {
                    ReduceMax<float>(maxHead, prodHead, softmaxTmpFp32, tileSize);
                    SetFlag<HardEvent::V_S>(EVENT_ID0);
                    WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const float maxScore = maxHead.GetValue(0);
                    SetFlag<HardEvent::S_V>(EVENT_ID0);
                    WaitFlag<HardEvent::S_V>(EVENT_ID0);
                    Adds(prodHead, prodHead, -maxScore, tileSize);
                    PipeBarrier<PIPE_V>();
                    Exp(prodHead, prodHead, tileSize);
                    PipeBarrier<PIPE_V>();
                    ReduceSum<float>(sumHead, prodHead, softmaxTmpFp32, tileSize);
                    SetFlag<HardEvent::V_S>(EVENT_ID0);
                    WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const float invSum = 1.0f / sumHead.GetValue(0);
                    SetFlag<HardEvent::S_V>(EVENT_ID0);
                    WaitFlag<HardEvent::S_V>(EVENT_ID0);
                    Muls(prodHead, prodHead, invSum, tileSize);
                    Duplicate(sumHead, 1.0f, 8);
                    PipeBarrier<PIPE_V>();
                } else {
                    SoftMaxShapeInfo softmaxShape = {1, tileSizeAlign, 1, tileSize};
                    SoftMaxTiling softmaxTiling = SoftMaxFlashV2TilingFunc(
                        softmaxShape, sizeof(float), sizeof(float), softmaxTmp.GetSize(), true, false);
                    SoftmaxFlashV2<float, true, true, false, false, ATTENTION_SOFTMAX_FLASHV2_CFG>(
                        prodHead, sumHead, maxHead, prodHead, expHead,
                        sumHead, maxHead, softmaxTmp, softmaxTiling, softmaxShape);
                    PipeBarrier<PIPE_V>();
                }

                if (tileStart != 0) {
                    ApplySoftmaxStateToRow<false>(outHead, outHead, expHead, headDim);
                    PipeBarrier<PIPE_V>();
                }
            }

            const uint64_t vOff = static_cast<uint64_t>(tileStart) * kvDim + kvHead * headDim;
            kvFp16 = kvFp16Queue.AllocTensor<half>();
            DataCopy(kvFp16, vGm[vOff], kvCopyParams);
            kvFp16Queue.EnQue(kvFp16);
            kvFp16 = kvFp16Queue.DeQue<half>();
            Cast(sharedKvFp32, kvFp16, RoundMode::CAST_NONE, tileSize * headDim);
            kvFp16Queue.FreeTensor(kvFp16);
            PipeBarrier<PIPE_V>();

            for (uint32_t localHead = 0; localHead < activeHeads; ++localHead) {
                LocalTensor<float> prodHead = prod[localHead * TILE_CONTEXT];
                LocalTensor<float> outHead = outFp32[localHead * headDim];
                Brcb(weights, prodHead, tileSizeAlign / 8, {1, 8});
                PipeBarrier<PIPE_V>();

                BinaryRepeatParams valueParams;
                valueParams.dstBlkStride = 1;
                valueParams.src0BlkStride = 0;
                valueParams.src1BlkStride = 1;
                valueParams.dstRepStride = headDim / 8;
                valueParams.src0RepStride = 1;
                valueParams.src1RepStride = headDim / 8;
                Mul(workFp32, weights, sharedKvFp32, headDim, tileSize, valueParams);
                PipeBarrier<PIPE_V>();

                uint32_t rows = tileSize;
                while (rows > 1) {
                    const uint32_t halfRows = rows / 2;
                    Add(workFp32, workFp32, workFp32[halfRows * headDim], halfRows * headDim);
                    PipeBarrier<PIPE_V>();
                    if ((rows & 1U) != 0U) {
                        Add(workFp32, workFp32, workFp32[(rows - 1U) * headDim], headDim);
                        PipeBarrier<PIPE_V>();
                    }
                    rows = halfRows;
                }
                Add(outHead, outHead, workFp32, headDim);
                PipeBarrier<PIPE_V>();
            }
        }

        for (uint32_t localHead = 0; localHead < activeHeads; ++localHead) {
            LocalTensor<float> outHead = outFp32[localHead * headDim];
            LocalTensor<float> sumHead = softmaxSum[localHead * 8U];
            ApplySoftmaxStateToRow<true>(outHead, outHead, sumHead, headDim);
            PipeBarrier<PIPE_V>();
        }
        Cast(outFp16, outFp32, RoundMode::CAST_RINT, activeHeads * headDim);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outGm[static_cast<uint64_t>(qGroupBegin) * headDim], outFp16, activeHeads * headDim);
    }

private:
    static constexpr uint32_t TILE_CONTEXT = 240;
    static constexpr uint32_t SHORT_CONTEXT_THRESHOLD = 64;
    static constexpr uint32_t MAX_Q_HEAD_TILE = 7;

    uint32_t context;
    uint32_t numQHeads;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t kvDim;
    uint32_t maxSeq;
    float scale;
    uint32_t qHeadTile;
    uint32_t groupsPerKv;
    uint32_t kvHead;
    uint32_t qGroupBegin;
    uint32_t qGroupEnd;
    uint32_t activeHeads;

    GlobalTensor<half> qGm;
    GlobalTensor<half> kGm;
    GlobalTensor<half> vGm;
    GlobalTensor<half> maskGm;
    GlobalTensor<int32_t> actualSeqLengthsKvGm;
    GlobalTensor<half> outGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> qFp16Buf;
    TBuf<TPosition::VECCALC> qFp32Buf;
    TQue<QuePosition::VECIN, 2> kvFp16Queue;
    TBuf<TPosition::VECCALC> sharedKvFp32Buf;
    TBuf<TPosition::VECCALC> workFp32Buf;
    TBuf<TPosition::VECCALC> prodBuf;
    TBuf<TPosition::VECCALC> outFp32Buf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> maskFp16Buf;
    TBuf<TPosition::VECCALC> maskFp32Buf;
    TBuf<TPosition::VECCALC> weightsBuf;
    TBuf<TPosition::VECCALC> softmaxMaxBuf;
    TBuf<TPosition::VECCALC> softmaxSumBuf;
    TBuf<TPosition::VECCALC> softmaxExpBuf;
    TBuf<TPosition::VECCALC> softmaxTmpBuf;
};

extern "C" __global__ __aicore__ void attention_step_custom(GM_ADDR query,
                                                              GM_ADDR k_cache,
                                                              GM_ADDR v_cache,
                                                              GM_ADDR mask,
                                                              GM_ADDR actual_seq_lengths_kv,
                                                              GM_ADDR out,
                                                              GM_ADDR workspace,
                                                              GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.useGqaReuse != 0U) {
        KernelAttentionStepGqaReuse op;
        op.Init(query, k_cache, v_cache, mask, actual_seq_lengths_kv, out,
                tiling_data.context, tiling_data.numQHeads, tiling_data.numKvHeads,
                tiling_data.headDim, tiling_data.maxSeq, tiling_data.scale,
                tiling_data.qHeadTile, tiling_data.groupsPerKv);
        op.Process();
    } else {
        KernelAttentionStepCustom op;
        op.Init(query, k_cache, v_cache, mask, actual_seq_lengths_kv, out,
                tiling_data.context, tiling_data.numQHeads, tiling_data.numKvHeads,
                tiling_data.headDim, tiling_data.maxSeq, tiling_data.scale);
        op.Process();
    }
}

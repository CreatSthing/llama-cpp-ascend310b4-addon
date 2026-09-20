#include "kernel_operator.h"

using namespace AscendC;

class KernelW8a8QuantizeCustom {
public:
    __aicore__ inline KernelW8a8QuantizeCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR xq, GM_ADDR scale, uint32_t K) {
        this->K = K;
        xGm.SetGlobalBuffer((__gm__ half *) x, K);
        xqGm.SetGlobalBuffer((__gm__ int8_t *) xq, K);
        scaleGm.SetGlobalBuffer((__gm__ half *) scale, 1);
        pipe.InitBuffer(xBuf, K * sizeof(half));
        pipe.InitBuffer(xqBuf, K * sizeof(int8_t));
        pipe.InitBuffer(fp32Buf, K * sizeof(float));
        pipe.InitBuffer(tempBuf, K * sizeof(float));
    }

    __aicore__ inline void Process() {
        LocalTensor<half> x = xBuf.Get<half>();
        LocalTensor<int8_t> xq = xqBuf.Get<int8_t>();
        LocalTensor<float> fp32 = fp32Buf.Get<float>();
        LocalTensor<float> temp = tempBuf.Get<float>();
        LocalTensor<int32_t> tempInt32 = tempBuf.Get<int32_t>();
        LocalTensor<half> tempHalf = xBuf.Get<half>();

        DataCopy(x, xGm, K);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(fp32, x, RoundMode::CAST_NONE, K);
        PipeBarrier<PIPE_V>();
        Abs(temp, fp32, K);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(temp, K);
        const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event);
        WaitFlag<HardEvent::V_S>(event);

        float maxAbs = temp.GetValue(0);
        if (maxAbs < 0.00000127f) {
            maxAbs = 0.00000127f;
        }
        const float scale = maxAbs / 127.0f;
        const float inverseScale = 127.0f / maxAbs;
        scaleGm.SetValue(0, static_cast<half>(scale));

        Muls(fp32, fp32, inverseScale, K);
        PipeBarrier<PIPE_V>();
        Cast(tempInt32, fp32, RoundMode::CAST_RINT, K);
        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0));
        PipeBarrier<PIPE_V>();
        Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, K);
        PipeBarrier<PIPE_V>();
        Cast(xq, tempHalf, RoundMode::CAST_TRUNC, K);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(xqGm, xq, K);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

private:
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> & values, uint32_t count) {
        constexpr uint32_t elemsPerRepeat = 64;
        const uint32_t repeats = count / elemsPerRepeat;
        const uint32_t offset = repeats * elemsPerRepeat;
        const uint32_t remainder = count % elemsPerRepeat;

        if (repeats > 1) {
            Max(values, values[elemsPerRepeat], values, elemsPerRepeat, repeats - 1,
                {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        if (remainder > 0 && offset > 0) {
            Max(values, values[offset], values, remainder, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        const uint32_t mask = repeats > 0 ? elemsPerRepeat : count;
        WholeReduceMax(values, values, mask, 1, 8, 1, 8);
    }

    uint32_t K{0};
    GlobalTensor<half> xGm;
    GlobalTensor<int8_t> xqGm;
    GlobalTensor<half> scaleGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> xBuf;
    TBuf<TPosition::VECCALC> xqBuf;
    TBuf<TPosition::VECCALC> fp32Buf;
    TBuf<TPosition::VECCALC> tempBuf;
};

extern "C" __global__ __aicore__ void w8a8_quantize_custom(GM_ADDR x,
                                                            GM_ADDR xq,
                                                            GM_ADDR scale,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelW8a8QuantizeCustom op;
    op.Init(x, xq, scale, tiling_data.K);
    op.Process();
}

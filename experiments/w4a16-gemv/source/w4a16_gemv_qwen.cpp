#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kCoreCount = 8;
constexpr uint32_t kInputSize = 1536;
constexpr uint32_t kOutputSize = 8960;
constexpr uint32_t kGroupSize = 32;
constexpr uint32_t kGroupCount = kInputSize / kGroupSize;
constexpr uint32_t kOutputTile = 448;
constexpr uint32_t kHalfTile = kOutputTile / 2;
constexpr uint32_t kPackedRowBytes = kOutputTile / 2;
constexpr uint32_t kPackedGroupBytes = kGroupSize * kPackedRowBytes;
constexpr uint32_t kPackedGroupWords = kPackedGroupBytes / sizeof(uint16_t);
constexpr uint32_t kHalfGroupElements = kGroupSize * kHalfTile;
constexpr uint32_t kInputGroupElements = kGroupSize * kOutputTile;
constexpr uint32_t kTileCount = kOutputSize / kOutputTile;
constexpr uint32_t kBaseTilesPerCore = kTileCount / kCoreCount;
constexpr uint32_t kExtraTileCores = kTileCount % kCoreCount;
constexpr uint32_t kMaxTilesPerCore = kBaseTilesPerCore + 1;
}

class KernelW4A16GemvQwen {
public:
    __aicore__ inline void Init(GM_ADDR expandedInput, GM_ADDR packedWeight,
                                GM_ADDR scales, GM_ADDR output)
    {
        core_ = GetBlockIdx();
        tileCount_ = kBaseTilesPerCore + (core_ < kExtraTileCores ? 1 : 0);
        tileStart_ = core_ * kBaseTilesPerCore +
                     (core_ < kExtraTileCores ? core_ : kExtraTileCores);
        inputGm_.SetGlobalBuffer((__gm__ half *)expandedInput,
                                 kInputSize * kOutputTile);
        weightGm_.SetGlobalBuffer((__gm__ uint16_t *)packedWeight,
                                  kTileCount * kGroupCount * kPackedGroupWords);
        scaleGm_.SetGlobalBuffer((__gm__ half *)scales,
                                 kTileCount * kGroupCount * kOutputTile);
        outputGm_.SetGlobalBuffer((__gm__ half *)output, kOutputSize);

        pipe_.InitBuffer(weightQueue_, 2, kPackedGroupBytes);
        pipe_.InitBuffer(inputQueue_, 2, kInputGroupElements * sizeof(half));
        pipe_.InitBuffer(scaleQueue_, 2, kOutputTile * sizeof(half));
        pipe_.InitBuffer(outputQueue_, 1, kOutputTile * sizeof(half));
        pipe_.InitBuffer(maskBuffer_, kPackedGroupBytes);
        pipe_.InitBuffer(lowBuffer_, kPackedGroupBytes);
        pipe_.InitBuffer(highBuffer_, kPackedGroupBytes);
        pipe_.InitBuffer(weightHalfBuffer_,
                         2 * kHalfGroupElements * sizeof(half));
        pipe_.InitBuffer(groupTotalBuffer_, kOutputTile * sizeof(half));
        pipe_.InitBuffer(productBuffer_, kOutputTile * sizeof(half));
        pipe_.InitBuffer(totalBuffer_,
                         kMaxTilesPerCore * kOutputTile * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<uint16_t> mask = maskBuffer_.Get<uint16_t>();
        LocalTensor<half> totals = totalBuffer_.Get<half>();
        Duplicate(mask, (uint16_t)0x0F0F, kPackedGroupWords);
        Duplicate(totals, (half)0.0, tileCount_ * kOutputTile);

        // Keep one input group in UB while all local output tiles consume it.
        for (uint32_t group = 0; group < kGroupCount; ++group) {
            LocalTensor<half> input = inputQueue_.AllocTensor<half>();
            DataCopy(input, inputGm_[group * kInputGroupElements],
                     kInputGroupElements);
            inputQueue_.EnQue(input);
            input = inputQueue_.DeQue<half>();

            for (uint32_t localTile = 0; localTile < tileCount_; ++localTile) {
                const uint32_t tile = tileStart_ + localTile;
                LocalTensor<uint16_t> packed =
                    weightQueue_.AllocTensor<uint16_t>();
                LocalTensor<half> scale = scaleQueue_.AllocTensor<half>();
                const uint32_t weightOffset =
                    (tile * kGroupCount + group) * kPackedGroupWords;
                const uint32_t scaleOffset =
                    (tile * kGroupCount + group) * kOutputTile;
                DataCopy(packed, weightGm_[weightOffset], kPackedGroupWords);
                DataCopy(scale, scaleGm_[scaleOffset], kOutputTile);
                weightQueue_.EnQue(packed);
                scaleQueue_.EnQue(scale);
                packed = weightQueue_.DeQue<uint16_t>();
                scale = scaleQueue_.DeQue<half>();

                ComputeGroup(totals[localTile * kOutputTile], packed,
                             input, scale, mask);

                weightQueue_.FreeTensor(packed);
                scaleQueue_.FreeTensor(scale);
            }
            inputQueue_.FreeTensor(input);
        }

        for (uint32_t localTile = 0; localTile < tileCount_; ++localTile) {
            LocalTensor<half> out = outputQueue_.AllocTensor<half>();
            DataCopy(out, totals[localTile * kOutputTile], kOutputTile);
            outputQueue_.EnQue(out);
            out = outputQueue_.DeQue<half>();
            DataCopy(outputGm_[(tileStart_ + localTile) * kOutputTile],
                     out, kOutputTile);
            outputQueue_.FreeTensor(out);
        }
    }

private:
    __aicore__ inline void ComputeGroup(LocalTensor<half> total,
                                        LocalTensor<uint16_t> packed,
                                        LocalTensor<half> input,
                                        LocalTensor<half> scale,
                                        LocalTensor<uint16_t> mask)
    {
        LocalTensor<uint16_t> low = lowBuffer_.Get<uint16_t>();
        LocalTensor<uint16_t> high = highBuffer_.Get<uint16_t>();
        LocalTensor<half> weight = weightHalfBuffer_.Get<half>();
        LocalTensor<half> weightLow = weight;
        LocalTensor<half> weightHigh = weight[kHalfGroupElements];
        LocalTensor<half> groupTotal = groupTotalBuffer_.Get<half>();
        LocalTensor<half> product = productBuffer_.Get<half>();
        LocalTensor<half> groupLow = groupTotal;
        LocalTensor<half> groupHigh = groupTotal[kHalfTile];
        LocalTensor<half> productLow = product;
        LocalTensor<half> productHigh = product[kHalfTile];

        // Unpack a whole 32x64 group with wide vector instructions.
        And(low, packed, mask, kPackedGroupWords);
        ShiftRight(high, packed, (uint16_t)4, kPackedGroupWords);
        And(high, high, mask, kPackedGroupWords);
        Cast(weightLow, low.ReinterpretCast<uint8_t>(),
             RoundMode::CAST_NONE, kHalfGroupElements);
        Cast(weightHigh, high.ReinterpretCast<uint8_t>(),
             RoundMode::CAST_NONE, kHalfGroupElements);
        Adds(weightLow, weightLow, (half)-8.0, kHalfGroupElements);
        Adds(weightHigh, weightHigh, (half)-8.0, kHalfGroupElements);
        Duplicate(groupTotal, (half)0.0, kOutputTile);

        for (uint32_t k = 0; k < kGroupSize; ++k) {
            LocalTensor<half> inputRow = input[k * kOutputTile];
            Mul(productLow, weightLow[k * kHalfTile], inputRow, kHalfTile);
            Mul(productHigh, weightHigh[k * kHalfTile],
                inputRow[kHalfTile], kHalfTile);
            Add(groupLow, groupLow, productLow, kHalfTile);
            Add(groupHigh, groupHigh, productHigh, kHalfTile);
        }
        // Scale is constant for a K group, so apply it once after accumulation.
        Mul(groupTotal, groupTotal, scale, kOutputTile);
        Add(total, total, groupTotal, kOutputTile);
    }

private:
    uint32_t core_;
    uint32_t tileStart_;
    uint32_t tileCount_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> weightQueue_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECIN, 2> scaleQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> maskBuffer_;
    TBuf<QuePosition::VECCALC> lowBuffer_;
    TBuf<QuePosition::VECCALC> highBuffer_;
    TBuf<QuePosition::VECCALC> weightHalfBuffer_;
    TBuf<QuePosition::VECCALC> groupTotalBuffer_;
    TBuf<QuePosition::VECCALC> productBuffer_;
    TBuf<QuePosition::VECCALC> totalBuffer_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<uint16_t> weightGm_;
    GlobalTensor<half> scaleGm_;
    GlobalTensor<half> outputGm_;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR expandedInput, GM_ADDR packedWeight, GM_ADDR scales,
    GM_ADDR output)
{
    KernelW4A16GemvQwen op;
    op.Init(expandedInput, packedWeight, scales, output);
    op.Process();
}

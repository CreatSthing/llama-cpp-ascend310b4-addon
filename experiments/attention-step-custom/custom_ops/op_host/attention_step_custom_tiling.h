#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AttentionStepCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, context);
    TILING_DATA_FIELD_DEF(uint32_t, numQHeads);
    TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, maxSeq);
    TILING_DATA_FIELD_DEF(float, scale);
    TILING_DATA_FIELD_DEF(uint32_t, useGqaReuse);
    TILING_DATA_FIELD_DEF(uint32_t, qHeadTile);
    TILING_DATA_FIELD_DEF(uint32_t, groupsPerKv);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AttentionStepCustom, AttentionStepCustomTilingData)
}  // namespace optiling

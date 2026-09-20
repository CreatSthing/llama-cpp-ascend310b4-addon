import os

import numpy as np


SEED = 20260916
K = 1536
N = 8960
GROUP_SIZE = 32
OUTPUT_TILE = 448

rng = np.random.default_rng(SEED)
x = rng.normal(0.0, 0.08, K).astype(np.float16)
q = rng.integers(-8, 8, size=(N, K), dtype=np.int8)
scales = rng.uniform(0.002, 0.02, size=(N, K // GROUP_SIZE)).astype(np.float16)

tile_count = N // OUTPUT_TILE
group_count = K // GROUP_SIZE
packed = np.empty((tile_count, group_count, GROUP_SIZE, OUTPUT_TILE // 2),
                  dtype=np.uint8)
scale_layout = np.empty((tile_count, group_count, OUTPUT_TILE), dtype=np.float16)

for tile in range(tile_count):
    n0 = tile * OUTPUT_TILE
    for group in range(group_count):
        k0 = group * GROUP_SIZE
        half_tile = OUTPUT_TILE // 2
        low = q[n0:n0 + half_tile, k0:k0 + GROUP_SIZE].T.astype(np.int16) + 8
        high = q[n0 + half_tile:n0 + OUTPUT_TILE, k0:k0 + GROUP_SIZE].T.astype(np.int16) + 8
        packed[tile, group] = (low | (high << 4)).astype(np.uint8)
        scale_layout[tile, group, :half_tile] = scales[n0:n0 + half_tile, group]
        scale_layout[tile, group, half_tile:] = scales[n0 + half_tile:n0 + OUTPUT_TILE, group]

golden = np.zeros(N, dtype=np.float16)
for k in range(K):
    group = k // GROUP_SIZE
    weight = (q[:, k].astype(np.float16) * scales[:, group]).astype(np.float16)
    product = (weight * x[k]).astype(np.float16)
    golden = (golden + product).astype(np.float16)

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
np.repeat(x[:, None], OUTPUT_TILE, axis=1).tofile("input/input_x.bin")
packed.tofile("input/input_weight.bin")
scale_layout.tofile("input/input_scales.bin")
golden.tofile("output/golden.bin")
print(f"x={x.shape}, packed={packed.shape}, scales={scale_layout.shape}, y={golden.shape}")

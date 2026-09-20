import numpy as np


actual = np.fromfile("output/output_y.bin", dtype=np.float16)
golden = np.fromfile("output/golden.bin", dtype=np.float16)
if actual.shape != golden.shape:
    raise SystemExit(f"shape mismatch: actual={actual.shape}, golden={golden.shape}")

actual32 = actual.astype(np.float32)
golden32 = golden.astype(np.float32)
abs_error = np.abs(actual32 - golden32)
cosine = float(np.dot(actual32, golden32) /
               (np.linalg.norm(actual32) * np.linalg.norm(golden32)))
print(f"max_abs_error={abs_error.max():.6f} mean_abs_error={abs_error.mean():.6f} cosine={cosine:.8f}")
if not np.allclose(actual, golden, rtol=2e-2, atol=2e-2):
    raise SystemExit("W4A16_GEMV_VERIFY_FAILED")
print("W4A16_GEMV_VERIFY_OK")

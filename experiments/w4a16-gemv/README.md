# Qwen2.5-1.5B single-token W4A16 GEMV probe

This fixed-shape probe validates the first Qwen FFN projection:
`[1,1536] x [1536,8960] -> [1,8960]` on Ascend 310B4.

It follows the official CANN FFN AntiQuant structure where it fits the target:

- split the output axis across AI cores;
- move a full K-group instead of issuing one global-memory copy per element;
- load one FP16 scale vector per Q4 group;
- use queues for MTE2-to-vector and vector-to-MTE3 ordering;
- keep two input buffers for later copy/compute overlap.

The official A2 implementation expands quantized weights to an FP16 global
workspace and then calls Cube Matmul. That path cannot be copied directly:
310B4 lacks the official INT4-to-FP16 conversion, and writing the expanded
weight back to global memory removes much of the decode-time bandwidth gain.
This probe unpacks each Q4 tile in UB and immediately performs the FP16 vector
multiply-accumulate without writing expanded weights to global memory.

The first gate is correctness. Once it passes, add event timing and compare it
with the same fixed-shape FP16 GEMV. Integration into llama.cpp starts only if
the standalone W4A16 operator is faster.

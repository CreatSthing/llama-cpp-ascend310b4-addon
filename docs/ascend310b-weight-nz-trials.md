# Ascend 310B4 weight prepack trials

This log tracks attempts to pre-convert constant matrix weights on Ascend
310B4 so that decode does not repeatedly pay for ND/private-format `TransData`.
The experimental path is disabled by default and is enabled with:

```bash
GGML_CANN_WEIGHT_NZ=on llama-bench ...
```

## 2026-09-14: CANN 8.5 API capability probe

Environment:

- Device: Ascend 310B4
- Driver: 23.0.0
- CANN: 8.5.0 with the 310B ops package
- Model test shape: FP16 `[896, 896]`

Results:

| API | Result | Detail |
|---|---|---|
| `aclnnCalculateMatmulWeightSizeV2` | FAIL | returns `361001` and size `0` |
| `aclnnTransMatmulWeightGetWorkspaceSize` | PASS | returns a valid executor and 1,606,144-byte workspace |
| `aclnnTransMatmulWeight` | PASS | executes and synchronizes successfully |

Conclusion: the earlier assumption that the complete NZ path is absent on
310B4 is too broad. The conversion operation works after the CANN 8.5 upgrade;
only its allocation-size helper is missing from the installed 310B kernel set.

The current prototype computes the required storage conservatively as FP16
`ceil(K/16) * ceil(N/16)` elements, matching FRACTAL_NZ's 16x16 blocking, and
keeps the feature opt-in. Before enabling it by default, the following gates
must all pass:

1. model load without an overwrite or allocation error;
2. deterministic output equality against the ND path;
3. lower `TransData` time/count in `msprof`;
4. improved `pp512` and/or `tg32` in repeated warm benchmarks.

## 2026-09-14: single-layout model test

The first opt-in implementation replaced every eligible FP16 weight with its
NZ representation. It established that the converted data is numerically
usable, but also exposed a shape limit in the 310B Matmul kernels.

| Test | ND | NZ | Result |
|---|---:|---:|---|
| `pp8`, one repetition | 26.26 t/s | 77.82 t/s | NZ runs and is 2.96x faster |
| prompt M=1/2/4/8 | PASS | PASS | NZ kernel available |
| prompt M=16/32/64 | PASS | FAIL | NZ Matmul path aborts |
| `tg32`, Flash Attention off | 3.35 t/s | 10.94 t/s | NZ is 3.27x faster |
| 8-token greedy continuation | `I am trying to create a simple` | same | output matches |

The Flash Attention path failed after `pp8` with both NZ enabled and disabled,
so that failure is independent of weight packing.

### Revised design

The second prototype retains two copies of each eligible FP16 matrix weight:

- ND at the original tensor address for prefill (`M > 8`);
- FRACTAL_NZ at a 128-byte-aligned offset for decode/small batches (`M <= 8`).

This costs roughly one additional FP16 model-weight copy, but it allows one
model instance to serve both prefill and decode. The feature remains opt-in
until broader model and shape coverage is complete.

## 2026-09-14: dual-layout end-to-end validation

The mixed prefill/decode run passes on the 310B4. Matmuls with `M > 8` use
the preserved ND weight, while `M <= 8` uses the prepacked NZ copy.

| Test | ND | Dual ND/NZ | Change |
|---|---:|---:|---:|
| `pp64`, two repetitions | 196.29 t/s | 246.87 t/s | 1.26x |
| `tg32`, two repetitions | 3.35 t/s | 10.84 t/s | 3.24x |
| 20-token prompt, greedy 8-token continuation | 3.22 t/s | 9.61 t/s | 2.98x |

The greedy continuation was identical in both modes:

```text
 Edge AI inference can reduce network latency for
```

This validates the layout selection and numerical result for the tested
Qwen2.5 0.5B FP16 model. It does not yet establish support for quantized
weights, every model shape, or production use. At the time of this test, Flash
Attention remained off because the separate experimental 310B custom attention
path failed with both ND and NZ weights.

## 2026-09-14: custom attention compatibility recheck

The installed `AttentionStepCustom` operator passes its standalone two-head,
four-position numerical probe. With the current CANN 8.5 build and the same
Qwen2.5 0.5B FP16 model, the earlier `-fa on` failure did not reproduce:

- `pp64` and `tg32` completed with `GGML_CANN_WEIGHT_NZ=on` and `-fa on`.
- Two fixed-prompt greedy continuations (12 and 24 tokens) matched `-fa off`.
- The CANN backend now routes only the tested single-token, 64-wide, FP16-KV
  shape to the custom operator. Unsupported shapes fall back to another backend.

This is a narrow correctness check, not proof of support for other models or
long contexts. The previous failure's exact cause remains unknown. Flash
Attention is still optional; no performance claim is made here.

## 2026-09-15: Qwen2.5-1.5B decode recovery

The dual ND/NZ layout was retested with Qwen2.5-1.5B-Instruct F16 on the
Ascend 310B4. Flash Attention was disabled so this test isolates weight
packing. Each case used three repetitions and offloaded all 29 layers.

```bash
export GGML_CANN_WEIGHT_NZ=on
./build-cann85/bin/llama-bench \
  -m ./qwen2.5-1.5b-instruct-f16.gguf \
  -p 0 -n 64 -r 3 -ngl 99 -fa off
```

| Test | ND baseline | Dual ND/NZ | Speedup |
|---|---:|---:|---:|
| `tg64`, three repetitions | 1.42 t/s | 3.40 t/s | 2.39x |
| `tg256`, three repetitions | 1.42 t/s | 3.40 t/s | 2.39x |

The longer run matches `tg64`, so the recovered performance is sustained and
is not a short-generation timing artifact. Keep `GGML_CANN_WEIGHT_NZ=on` for
the Qwen2.5-1.5B decode baseline while the feature remains opt-in.

## 2026-09-15: double-buffered temporary lifetime fix

Large-M FP32-to-FP16 Matmul needs temporary input and output buffers. Returning
them immediately to the pool races asynchronous CANN work. Per-buffer events
were tested first, but event completion was intermittent on this 310B4 runtime:
independent `pp512` launches failed 2 out of 5 times.

The stable implementation retains two temporary-buffer groups. When a third
group is needed, it drains the stream once and releases both completed groups.
This preserves limited overlap without exceeding the board's 16 GB memory.
Eight retained groups exhausted memory and failed consistently.

Validation with `GGML_CANN_WEIGHT_NZ=on`, `-ngl 99`, and `-fa off`:

| Test | Result |
|---|---:|
| `pp512 -r 3` | 121.24 +/- 0.03 token/s |
| five independent `pp512 -r 1` launches | 5/5 passed |
| `tg64 -r 3` | 3.39 +/- 0.00 token/s |

## 2026-09-15: Qwen2.5-1.5B tied output weight

The Qwen2.5-1.5B F16 GGUF contains `token_embd.weight` with shape
`[1536, 151936]`, but no separate `output.weight`. Qwen2 model loading
reuses the embedding tensor for the final output projection. The existing NZ
weight classifier recognized `output.weight`, not this tied tensor, so decode
repacked the 151936-by-1536 output matrix on every generated token.

In the prior `tg64` profile, the 151936-by-1536 ND-to-FRACTAL_NZ conversion
appeared 65 times and took 4,806,279.96 us in total (73.94 ms/call). A separate
`GGML_CANN_TIED_OUTPUT_NZ=on` experiment switch now selects only this exact
FP16 tied-weight shape for the existing dual ND/NZ layout. ND remains available
for embedding lookup and large-M prefill; NZ serves small-M output Matmul.
Both `GGML_CANN_WEIGHT_NZ=on` and the new switch are required.

| Test, CANN 8.5, 310B4, `-ngl 99`, `-fa off` | Previous dual ND/NZ | Tied output NZ |
|---|---:|---:|
| `tg64 -r 3` | 3.39–3.40 t/s | 4.55 +/- 0.00 t/s |
| `tg256 -r 3` | 3.40 t/s | 4.55 +/- 0.00 t/s |
| `pp512 -r 3` | 121.24 t/s | 123.49 +/- 0.01 t/s |

The decode gain is about 34% over the prior NZ baseline. Two independent
16-token greedy completions of the same fixed prompt, with the tied-weight
switch off and on, produced byte-identical generated text. This checks the
tested prompt only, not general numerical equivalence.
An additional fixed Qwen chat prompt asking `2 + 3` returned `5` in both
modes and then stopped at end-of-text.

The valid `tg8` profile with the new switch shows the 151936-by-1536 weight
converted ND-to-FRACTAL_NZ once at load time (85,670.42 us). The
`aclnnMatmul_TransData_TransData` conversion with that large input shape occurs
zero times. Output-format conversion after the Matmul still runs per token;
that is a separate, much smaller operation.

Profile data: `../llama-cpp-310b-workspace/profiling/tied-output-nz-tg8-valid/`.
Board logs: `/tmp/tied-output-nz-tg64-r3.log`,
`/tmp/tied-output-nz-tg256-r3.log`, `/tmp/tied-output-nz-pp512-r3.log`, and
`/tmp/tied-output-answer-{off,on}.log`, and
`/tmp/tied-output-chat-{off,on}.log`.

## 2026-09-15: post-fix hotspot and Q4 probe

A new optimized `tg32` profile attributes 82.77% of device task time, after
excluding one-time weight packing, to MatMul. The three FFN projections account
for 61.77%, and the output projection accounts for 11.90%. The dominant FFN
kernels report MTE2 execution ratios around 0.906-0.912, consistent with a
weight-bandwidth-bound decode workload.

A pure Q4_0 probe reduced the model from 2944.68 MiB to 828.59 MiB; pure Q8_0
reduced it to 1564.62 MiB. Both GGUF formats use weight-only W4A16/W8A16
inference in the current CANN backend. `aclnnWeightQuantBatchMatmulV2` returns
error 361001 (`support for Ascend310B is not implemented`) for this path, and
V3 returns the same error. The V3 probe was reverted after confirmation.

This does not mean that 310B lacks quantized MatMul. The documented Atlas
inference-series route through `aclnnQuantMatmulV3` is INT8 x INT8 (A8W8), which
requires dynamic activation quantization and dequantization parameters. The
current llama.cpp CANN backend does not implement that route. The next probe
should benchmark A8W8 on M=1 FFN shapes 1536x8960 and 8960x1536 before graph
integration. A custom W4A16 kernel remains a later option if A8W8 activation
quantization overhead is too high.

## 2026-09-16: Qwen2.5-1.5B dynamic A8W8 decode experiment

The system `aclnnQuantMatmulV3` binary is unavailable for A8W8 on this 310B4
installation, so the experiment uses the installed custom operator chain:

1. `W8a8QuantizeCustom`: FP16 activation to per-token INT8 and FP16 scale.
2. `MatmulW8a8I32Custom`: INT8 by INT8 MatMul with INT32 accumulation.
3. `W8a8DequantCustom`: multiply the accumulator by activation and per-column
   weight scales, producing FP16.

The scalar activation quantizer took 245 us for K=1536 and 1,400 us for K=8960.
Replacing its scalar loops with AscendC vector operations reduced these medians
to 8.38 us and 15.65 us. The first ND-weight MatMul looked fast in a repeated
microbenchmark, but an end-to-end profile exposed cold-weight times of 10.25 ms
and 4.26 ms. Prepacking weights into the 310B INT8 FRACTAL_NZ 16-by-32 layout
reduced the end-to-end MatMul medians to 1.10 ms and 1.07 ms.

The path is opt-in with `GGML_CANN_A8W8=on`. It only selects Qwen FFN gate, up,
and down FP16 weights for single-token decode. The original FP16 data remains
available for prefill. `GGML_CANN_WEIGHT_NZ=on` and
`GGML_CANN_TIED_OUTPUT_NZ=on` remain enabled for the other decode weights.

| Test, three repetitions | FP16/NZ control | Dynamic A8W8 plus NZ | Change |
| --- | ---: | ---: | ---: |
| `tg64` | 3.06 t/s | 4.70 t/s | +53.6% |
| `tg256` | 3.06 t/s | 4.67 t/s | +52.6% |

A deterministic `2 + 3` prompt produced `5` on both paths. This is a smoke
check, not a complete accuracy evaluation. The full A8W8 msprof archive and
benchmark log are stored under the companion workspace `profiling` directory.

Profile archive:
`../llama-cpp-310b-workspace/profiling/qwen15b-after-tied-nz-tg32.tar.gz`.

## 2026-09-16: tied output dynamic A8W8

The current-best profile still spent about 48 ms per generated token in the
151936-column tied output projection. `GGML_CANN_TIED_OUTPUT_A8W8=on` now gives
the exact Qwen2.5-1.5B `token_embd.weight` shape an INT8 FRACTAL_NZ side copy.
Embedding lookup and prefill continue to use the original FP16 data; only M=1
output projection uses the dynamic A8W8 chain.

| Test, three repetitions | FFN A8W8 plus tied-output NZ | Tied-output A8W8 | Change |
| --- | ---: | ---: | ---: |
| `tg64` | 4.70 t/s | 5.44 t/s | +15.7% |
| `tg256` | 4.67 t/s | 5.47 t/s | +17.1% |
| `pp512` | 110.30 t/s | 109.99 t/s | -0.3% |

The fixed `2 + 3` smoke prompt still produced `5`. In the new profile, the
output MatMul median fell from about 48.0 ms to 18.7 ms. Its one-time INT8
packing increases model startup time but is not repeated during generation.
The next largest optimization target is now the per-layer FFN A8W8 MatMul.

Profile archive:
`../llama-cpp-310b-workspace/profiling/qwen15b-output-a8w8-tg4-msprof.tar.gz`.

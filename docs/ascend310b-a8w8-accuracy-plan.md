# Qwen2.5-1.5B A8W8 accuracy experiment

## Question answered

The experiment measures accuracy loss from the two approximate optimizations:

1. dynamic A8W8 for the FFN weights;
2. dynamic A8W8 for the tied output projection.

The NZ layout optimization only rearranges FP16 data and is retained in the
reference. It should not introduce quantization loss.

## Controlled comparison

All three runs use the same F16 GGUF, text, tokenization, context size, seed,
device, and CANN build.

| Group | FFN | tied output | Purpose |
| --- | --- | --- | --- |
| A | FP16/NZ | FP16/NZ | reference |
| B | dynamic A8W8 | FP16/NZ | isolate FFN loss |
| C | dynamic A8W8 | dynamic A8W8 | measure total and incremental output loss |

The command forces `-b 1 -ub 1`. This is essential because the experimental
A8W8 path is selected only for an M=1 decode matrix multiplication. A normal
large-batch perplexity run would silently test the FP16 prefill path instead.

## Metrics and gates

The primary test uses llama.cpp's KL-divergence mode on WikiText-2. It reports
PPL, KL divergence from FP16, correct-token probability change, RMS probability
change, and top-token agreement.

Use these initial acceptance gates for group C versus A:

- relative PPL increase no more than 1%;
- mean KL divergence no more than 0.01;
- same top token at least 98%;
- mean correct-token probability loss no worse than 0.5 percentage points.

Group B versus A locates FFN loss. Group C versus B locates the extra loss from
the output projection. A failure in C alone means the output layer should return
to FP16/NZ while retaining FFN A8W8.

## Stages

Run `scripts/ascend310b-a8w8-accuracy.sh smoke` first (128-token context), then
`quick` (512-token context). If quick passes, run `full`, which evaluates eight
512-token chunks. The reference logits are stored with the logs, so disk use is
roughly 40 MB for smoke, 160 MB for quick, and 1.2 GB for full.

After the numerical gates pass, run a small deterministic Chinese/English task
set in all three modes. It should include arithmetic, factual QA, instruction
following, JSON formatting, and short code generation. Use exact-match or a
task-specific checker rather than judging prose by eye. This catches behavioral
changes that corpus perplexity may miss.

## Initial smoke result

The 128-token smoke run completed successfully with the intended M=1 path:

| Group | PPL | PPL change | KL divergence | same top token |
| --- | ---: | ---: | ---: | ---: |
| A: FP16/NZ | 5.6811 | reference | reference | reference |
| B: FFN A8W8 | 5.9434 | +4.6% | 0.02520 | 93.651% |
| C: FFN + output A8W8 | 6.0944 | +7.3% | 0.03600 | 90.476% |

This sample is too small for a final quality decision: its PPL uncertainty is
large. It nevertheless fails all provisional numerical gates and demonstrates
that the decode A8W8 path has measurable loss. Run `quick` next to confirm the
direction. If confirmed, disable output A8W8 first; its incremental speed gain
does not justify the observed extra divergence. Then test finer FFN weight
scales or selective FP16 layers before accepting FFN A8W8.

## 512-token quick result and decision

| Group | PPL | PPL ratio | KL divergence | mean correct-token Δp | same top token |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: FP16/NZ | 6.9207 | reference | reference | reference | reference |
| B: FFN A8W8 | 6.8753 | 0.9934 | 0.02083 | -0.213 pp | 96.078% |
| C: FFN + output A8W8 | 6.8669 | 0.9922 | 0.03164 | -0.229 pp | 94.510% |

The small PPL decrease is inside the reported uncertainty and is not evidence
of improved quality. Both A8W8 groups fail the KL and top-token gates. Adding
output A8W8 increases KL by 0.01080 and reduces top-token agreement by another
1.568 percentage points, so it should not be the quality-default despite its
generation-speed gain.

Do not spend board time on the eight-chunk run for this implementation. Keep
FP16/NZ as the reference/default, keep output A8W8 experimental, and improve
FFN accuracy before repeating `quick`. The next controlled experiment should
keep the output in FP16/NZ and retain selected sensitive FFN layers in FP16,
testing layer blocks from the beginning and end of the network. If that cannot
meet the gates at useful speed, replace the single per-token activation scale
with grouped activation scales and repeat the same protocol.

## Selective FFN fallback quick gate (2026-09-16)

The 28 FFN layers were screened in blocks of four. The final candidate keeps
FFN layers 0–7, 13, and 16–27 in FP16/NZ; only layers 8–12 and 14–15 use
A8W8. The tied output projection stays FP16/NZ. Every corpus run used
`-b 1 -ub 1`; the fixed-question runner also reads the prompt one token at a
time and generates greedily.

| Gate | FP16/NZ | selective A8W8 | limit | outcome |
| --- | ---: | ---: | ---: | --- |
| English 256 scored tokens, PPL | 6.9207 | 6.9792 (+0.85%) | +1% | pass |
| English KL / top-token match | reference | 0.00153 / 99.22% | ≤0.01 / ≥98% | pass |
| Chinese 256 scored tokens, PPL | 13.9591 | 13.9616 (+0.02%) | +1% | pass |
| Chinese KL / top-token match | reference | 0.00184 / 98.43% | ≤0.01 / ≥98% | pass |
| fixed 20 questions | 16/20 | 16/20 | lose at most one | pass |
| `tg64`, three repeats | 3.10 t/s | 3.35 t/s (+8.1%) | clear gain | provisional |

The English corpus was WikiText-2, including its first 128 tokens used during
layer selection. The Chinese quick corpus was a small hand-written text repeated
to satisfy the tool's minimum input length. Therefore this quick gate is a
screening result, not independent final evidence. The 20 questions were answered
in one combined prompt; both modes used the same prompt and score parser.
The table labels the initial one-chunk run as 256 tokens, but the tool scores
only the final 255 tokens of a 512-token chunk. The corrected quick script uses
`-c 514` to score exactly 256 tokens and must be run before claiming that this
specific short gate passed as written.

The speed benefit is much smaller than the unrestricted FFN A8W8 path. Do not
make this configuration the default until independent 4096–8192-token bilingual
corpora and C-Eval, GSM8K, HellaSwag, IFEval, and HumanEval comparisons meet
the formal gates. With roughly 3.35 generated tokens per second and
single-token corpus evaluation, the formal campaign will take hours rather than
the five-to-ten-minute quick estimate. Keep both accuracy and performance logs.
The original `tg64` command generated one token at a time but used llama-bench's
default context batch settings. Therefore the +8.1% figure is provisional;
`scripts/ascend310b-a8w8-holdout-speed.sh` explicitly sets `-b 1 -ub 1` and
repeats both `tg64` and `tg256` three times after the corpus run completes.

## Independent bilingual holdout in progress (2026-09-16)

`scripts/ascend310b-a8w8-holdout.sh` uses 16 chunks of 514 tokens per language,
scoring exactly 4096 tokens per corpus (256 scored tokens per chunk). It keeps
`-b 1 -ub 1` for both FP16 and selective
A8W8, records PPL and full-vocabulary KL/top-token comparison, and keeps the
tied output in FP16/NZ. English uses WikiText-2 `wiki.valid.raw`, rather than the
`wiki.test.raw` split used for layer selection. Chinese uses the official C-Eval
validation questions and answer choices, generated by
`scripts/ascend310b-ceval-corpus.py`; it is exam-style text, not natural prose.

The run started in the background on the board with logs in
`accuracy-a8w8/holdout-4096` and a controller log in
`accuracy-a8w8/holdout-run.log`. Do not report a formal quality pass until both
languages finish and the task benchmarks and speed gate are checked. Quick-gate
logs (excluding large `.kld` files) are archived locally in the companion
workspace's `accuracy/bilingual-256` directory.
After a successful holdout run, a follow-on process runs the strict
`-b 1 -ub 1` `tg64`/`tg256` speed comparisons and repeats the corrected quick
gate at context 514 in `accuracy-a8w8/quick-exact-256`. The follow-on controller
log is `accuracy-a8w8/holdout-follow-run.log`.

## Edge candidate and layer-output analysis (2026-09-16)

The formal holdout was deliberately stopped before completion because running
it before selecting the final layer set would require repeating several hours
of work. Existing block, 255-token bilingual, and new independent 128-token
results are now used as successive filters.

`scripts/ascend310b-a8w8-edge-screen.sh` compared the current seven-layer A8W8
set with candidates that add exactly one of layers 13, 16, 17, 18, or 19. On
the independent WikiText validation prefix, adding layer 16 gave the best PPL
ratio (0.99980), KL 0.00136, and 99.22% top-token agreement. This is a ranking
sample rather than a quality pass. `scripts/ascend310b-a8w8-edge-confirm.sh`
therefore checks the eight-layer candidate (8–12, 14–16) on independent English
and Chinese 256-token samples, then runs strict `-b 1 -ub 1` `tg64` and `tg256`
tests with three repeats.

The eval-callback example now supports binary layer dumps through
`LLAMA_LAYER_DUMP`. It records FP32-normalized `ffn_out-N` and `l_out-N` values;
`scripts/compare-layer-dumps.py` reports reference-versus-candidate cosine,
relative L2 error, and the per-layer change in cumulative `l_out` cosine error.
`scripts/ascend310b-layer-cosine.sh` runs this comparison after the add16 short
confirmation. Local `ffn_out` error distinguishes the layer's own quantization
damage, while cumulative `l_out` error detects amplification of errors from
earlier layers. These values rank candidates only; PPL, KL, top-token agreement,
fixed questions, task accuracy, and speed remain the acceptance criteria.

Only after the eight-layer candidate meets the short gates (targeting PPL drift
no greater than 0.95% for margin and at least 10% speed gain) should the exact
4096-token bilingual holdout run once with
`GGML_CANN_A8W8_DISABLE_LAYERS=0-7,13,17-27`. Otherwise fall back to the
seven-layer candidate or test the next ranked single-layer addition.

### Eight-layer add16 result

The candidate quantizing FFN layers 8–12 and 14–16 passed the independent
256-token bilingual quality screen:

| Gate | English | Chinese | limit |
| --- | ---: | ---: | ---: |
| PPL ratio | 1.00255 (+0.26%) | 1.00362 (+0.36%) | ≤1.01 |
| mean KL | 0.00156 | 0.00130 | ≤0.01 |
| same top token | 99.22% | 98.83% | ≥98% |

With strict `-b 1 -ub 1`, FP16/NZ measured 4.45 t/s and the candidate measured
4.85 t/s for both `tg64` and `tg256`, an 8.99% gain. This is a real improvement
but narrowly misses the experimental 10% target. The layer dump showed tiny
cumulative residual-stream error through layer 25, followed by a clear increase
at layer 26: `l_out-25` cosine 0.99999738 / relative L2 0.00229 versus
`l_out-26` cosine 0.99980619 / relative L2 0.01978. This indicates late-layer
amplification of accumulated differences even though layer 26 remains FP16;
it does not by itself prove that layer 26 is locally sensitive.

To test the requested boundary safely, one final short candidate adds layer 18
as well (A8W8 layers 8–12, 14–16, 18). Layer 18 was the next best independent
128-token PPL candidate (+0.094%, KL 0.00170). It must pass the same bilingual
256-token gates and exceed the eight-layer speed result before any long run.

### Nine-layer add16+add18 result

The final boundary candidate quantizes FFN layers 8–12, 14–16, and 18. It also
passed the independent bilingual 256-token screen:

| Gate | English | Chinese | limit |
| --- | ---: | ---: | ---: |
| PPL ratio | 1.00405 (+0.40%) | 1.00447 (+0.45%) | ≤1.01 |
| mean KL | 0.00178 | 0.00144 | ≤0.01 |
| same top token | 99.22% | 98.83% | ≥98% |

Strict `-b 1 -ub 1` speed increased from 4.45 to 4.90 t/s for both `tg64` and
`tg256`, a 10.11% gain. This exceeds the eight-layer candidate's 4.85 t/s while
retaining substantial short-gate margin. A layer-cosine confirmation is queued,
followed by the one-time exact 4096-token bilingual holdout using
`GGML_CANN_A8W8_DISABLE_LAYERS=0-7,13,17,19-27`. The candidate remains
experimental until the long holdout and task checks pass.

The 4096-token English holdout rejected the nine-layer candidate before the
last chunk completed: after 15 of 16 equal chunks, top-token agreement was
97.474%. Even a perfect final chunk could raise the aggregate only to about
97.63%, below the 98% gate. The run was stopped to save board time. Its PPL and
KL remained inside their limits, showing why top-token agreement is required in
addition to average-distribution metrics.

The eight-layer candidate is now the formal candidate. The completed English
FP16 reference is reused, and `scripts/ascend310b-a8w8-holdout-resume.sh` runs
the eight-layer English comparison plus the complete Chinese FP16/A8W8 pair.
This avoids repeating the roughly 35-minute English reference pass.

The eight-layer English holdout also failed the strict top-token gate, but only
by 0.026 percentage points: PPL ratio 1.001662 (+0.166%), mean KL 0.001550,
and top-token agreement 97.974%. Because the declared limit is 98%, this is a
failure rather than a rounded pass. The Chinese eight-layer run was stopped to
save time, and the formal holdout fell back to the seven-layer candidate
(A8W8 layers 8–12 and 14–15). The same completed English FP16 reference is
reused before running the seven-layer Chinese pair.

The user then explicitly selected the eight-layer candidate despite the
0.026-point top-token miss. It is therefore the intended deployment candidate,
not a strict-gate pass. The seven-layer fallback run was stopped. The
eight-layer Chinese 4096-token pair resumed with `SKIP_EN=1`; after it finishes,
`scripts/ascend310b-a8w8-fixed20-final.sh` automatically compares FP16 and the
eight-layer candidate on the fixed 20 questions. Reports must preserve the
distinction between user acceptance and satisfying the original 98% gate.

The Chinese 4096-token holdout passed: PPL ratio 1.002054 (+0.2054%), mean KL
0.001766, and top-token agreement 98.413%. The fixed 20 questions scored 16/20
for both FP16 and the eight-layer candidate.

### Realistic prefill/decode PPL mode

Using a larger logical batch with `ubatch=1` produced exactly the same metrics
as strict `-b 1 -ub 1`, but was 2–4% slower. The physical graph still processed
one token at a time, so the larger logical submission did not remove the main
cost.

`LLAMA_PPL_REALISTIC_DECODE=1` now models deployed inference explicitly. The
first half of each context is processed as a batched FP16 prefill. Its KV cache
is transferred to a separate context reserved with `ubatch=1`; the scored half
then runs token by token through the M=1 A8W8 path. On the 514-token Chinese
check, candidate time fell from 117.78 to 71.12 seconds (39.6% shorter). The
realistic result was PPL ratio 1.009510 (+0.9510%), mean KL 0.001173, and
top-token agreement 99.219%, passing all three declared gates. Formal PPL
scripts use this mode; generation benchmarks remain strict `-b 1 -ub 1`.

### Fast iteration gate and final-eight profiling

During kernel iteration, use a reduced engineering gate to shorten turnaround:
PPL drift at most 2%, mean KL at most 0.02, top-token agreement at least 97%,
no more than two lost answers on the fixed 20 questions, and at least 3%
repeatable generation-speed improvement. English and Chinese use 256 scored
tokens each. The longer holdout is repeated only after the main optimization
has stabilized.

The final eight-layer `tg16` msProf run measured a 216.95 ms median token step.
Device tasks occupied about 202.9 ms (93.5%), leaving about 14 ms (6.5%) in
dispatch gaps. FP16 MatMulV2 used 144.7 ms/token, the eight-layer custom A8W8
quant/matmul/dequant chain used about 20.1 ms/token, and the tied output matmul
alone used about 26.45 ms/token. Both FP16 and INT8 matrix paths were dominated
by MTE2 weight reads. Gate/Up fusion is therefore a bounded probe: retain it
only if the combined GEMV improves end-to-end speed by at least 3%; sharing
input quantization alone cannot meet that threshold.

### Fast 12-layer candidate (relaxed engineering gate)

Reusing Gate's input quantization for Up changed `tg` from 4.86 to 4.87 t/s
(about 0.2%). The guarded experiment was removed from source because it did
not reach the 3% end-to-end improvement gate. The output projection A8W8 route
also remains rejected: although it was faster, its earlier KL of 0.03164 and
top-token agreement of 94.510% fail even this relaxed quality gate.

The next bounded experiment increased the number of FFN layers using A8W8.
All comparisons used realistic FP16 prefill and token-by-token scoring on
256 English and 256 Chinese tokens. Generation speed used `-b 1 -ub 1` with
three repeats. The fixed 20 questions were checked separately.

| A8W8 FFN layers | English PPL drift | Chinese PPL drift | English / Chinese KL | English / Chinese same top | `tg64` / `tg256` | Fixed 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8–12, 14–16, 18–19 (10) | +1.165% | +0.768% | 0.001744 / 0.001719 | 99.219% / 99.219% | 4.93 / 4.93 t/s | 16/20 |
| 8–16, 18–19 (11) | +0.833% | −0.195% | 0.002083 / 0.001798 | 99.609% / 100% | 4.99 / 4.99 t/s | 16/20 |
| 8–19 (12) | +0.973% | +0.553% | 0.002119 / 0.002055 | 99.219% / 99.219% | 5.06 / 5.05 t/s | 16/20 |

The 12-layer candidate passes the relaxed engineering gate. Against the
4.45 t/s FP16/NZ baseline, `tg256` improves by about 13.5%. Select it with
`GGML_CANN_A8W8_DISABLE_LAYERS='0-7,20-27'`. This is a fast iteration result,
not a replacement for the longer bilingual and task holdout before deployment.
Stop expanding layers for now. Profile this candidate next, then focus on the
single-token fused A8W8 GEMV path if operator time remains the bottleneck.

The subsequent 4096-token English holdout measured PPL ratio 1.002103
(+0.2103%), mean KL 0.002245, and 97.363% same-top agreement. It comfortably
passes the relaxed 97% iteration gate but fails the original 98% deployment
gate. This confirms the candidate is usable for fast iteration and interactive
evaluation, while strict deployment should retain the eight-layer choice or
add targeted mixed-precision protection before promoting the 12-layer version.

The `tg16` msProf capture for the 12-layer candidate is archived at
`../llama-cpp-310b-workspace/profiling/qwen15b-fast12-tg16-msprof.tar.zst`.
Across the 17 measured passes, FP16 `MatMulV2` time fell from 2460.22 ms
(eight layers) to 2125.54 ms; custom INT8 matmul rose from 311.13 to
466.51 ms as four more layers moved to A8W8. The quant/matmul/dequant chain
for all 12 layers totaled 510.76 ms, or 30.04 ms/pass, with quantization and
dequantization only 44.25 ms of that (2.60 ms/pass). The tied output FP16
matmul still takes about 26.4 ms/pass. MTE2 accounts for 97.2% of the custom
matmul's measured core activity, indicating that weight movement dominates.
Therefore, a single-token GEMV layout experiment was a bounded next test;
combining only the existing quant/dequant launches has limited upside. These percentages are
profiled core metrics, not a claim that the entire token time is memory I/O.

An isolated AscendC GEMV-mode test changed the M=1 input A layout from ND to
VECTOR in the custom INT8 matmul, following the [official GEMV guidance](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_10019.html). Both
standalone shapes produced the expected result, but their short-run chain
latencies moved only from 888.19 to 877.47 us and 854.45 to 852.07 us.
Paired model `tg64` runs at the same time measured 3.59 t/s with the original
operator and 3.60 t/s with GEMV mode; absolute throughput was lower than the
earlier 5.06 t/s session for both variants. This is no defensible end-to-end
gain, so the experimental operator remains isolated and the original installed
operator is retained. Further work should target weight-transfer efficiency
or a better single-token kernel, with paired full-model validation.

### Current-state slowdown probe

A later clean `tg16` msProf capture reproduced the interactive slowdown at
3.37 t/s under profiling, versus 4.68 t/s in the earlier 12-layer capture.
Both reports contain the same operator counts, tensor shapes and A8W8 layer
selection. FP16 `MatMulV2` increased from 2125.54 to 3350.16 ms across the 17
passes, custom INT8 matmul from 466.51 to 660.68 ms, and `TensorMove` from
263.39 to 350.01 ms. Small vector operations stayed nearly unchanged.
`MatMulV2` MTE2 activity rose from 92.4% to 95.1% while its MAC share fell
from 13.0% to 8.3%; the custom INT8 matmul remained 98.0% MTE2 dominated.
This points to degraded weight-transfer throughput or device operating state,
not a graph-path regression. The current report is archived at
`../llama-cpp-310b-workspace/profiling/qwen15b-fast12-current-tg16-msprof.tar.zst`.

The board reports health codes `80E3A203` (LPM current-reading fault) and
`80F18003` (DDRA internal configuration error). No competing user NPU process
was present during the capture. Do not tune quality-sensitive layer selection
against this degraded absolute speed; compare kernel variants in paired runs,
and investigate board firmware/EEPROM or reset state separately.

# Qwen2.5-1.5B：第一份 llama.cpp 基线

目的：在任何新优化前，保存一次可重复的 310B4 运行结果。每组由
`llama-bench -r 3` 测 3 次；同一组结果中的平均值和波动由工具报告。

## 板端运行

先确认 FP16 GGUF 已在板端，且当前 `llama-completion` 和 `llama-bench` 可执行。
用固定提示词生成一次短文本，保存输出，确认模型不仅能加载，还能
正常生成：

```bash
source /usr/local/Ascend/cann/set_env.sh
GGML_CANN_WEIGHT_NZ=off ./build-cann85/bin/llama-completion \
  -m ./qwen2.5-1.5b-instruct-f16.gguf -ngl 99 -fa off \
  -p '请用一句话介绍你自己。' -n 32 --temp 0 -st \
  2>&1 | tee ./qwen25-15b-smoke.log
```

确认输出正常后，在板端 `llama-cpp-310b` 目录运行基线：

```bash
source /usr/local/Ascend/cann/set_env.sh
GGML_CANN_WEIGHT_NZ=off ./scripts/ascend310b-qwen25-15b-baseline.sh \
  ./qwen2.5-1.5b-instruct-f16.gguf
```

如果模型或二进制不在上述位置，可显式传入三个参数：

```bash
./scripts/ascend310b-qwen25-15b-baseline.sh \
  /path/to/model.gguf /path/to/llama-bench /path/to/output-dir
```

脚本保存 `environment.txt` 和四个原始日志：`pp128`、`pp512`、
`tg64`、`tg256`。`pp` 是读入提示词，`tg` 是逐字生成。它们的速度
不能混为一个数字。脚本固定 `-ngl 99`、`-fa off` 和每组 3 次；
默认权重布局 NZ 关闭。后续实验必须使用相同参数对照，并在日志中
说明改动了哪一个变量。

当前 1.5B 模型的 head dimension 是 128，而自定义 Attention
只支持 64，因此这份基线先使用 `-fa off`。基线完成后，再单独
验证 128 维 Attention 和 NZ 权重布局；不要把此前 0.5B 的结果
视作 1.5B 的性能数据。

完成条件：四组运行均成功；日志显示 CANN 后端和 NPU offload；
没有 NPU 错误；保存每组 3 次结果和环境信息。若某组失败，保留
原始日志并记录失败位置，不以其他组的结果代替。

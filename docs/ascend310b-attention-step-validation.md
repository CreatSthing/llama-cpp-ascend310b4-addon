# Ascend 310B4 decode Attention 适配与验证

验证日期：2026-09-14

## 目标与范围

本次工作为 llama.cpp 的 CANN 后端补充面向 Ascend 310B4 的单 token decode Attention 原型，目标模型为 Qwen2.5-0.5B FP16：

- Query heads：14
- KV heads：2
- Head dimension：64
- 最大上下文：8192
- 当前只在上述 `head_dim = 64` 条件下启用；不满足条件时继续使用原有 CANN 普通路径。

该算子对应增量推理阶段的 `Q × Kᵀ → mask → softmax → P × V`。设计参考 Ascend 官方开源 `cann-ops-adv` 中的 IncreFlashAttention 数据搬运方式，但本实现仍是面向 310B4 单 AI Core 的轻量原型，不宣称等同于完整 IFA 实现。

官方参考：

- [IFA 算子设计介绍](https://gitee.com/ascend/cann-ops-adv/blob/master/docs/common/IFA%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md)
- [IncreFlashAttention 文档](https://gitee.com/ascend/cann-ops-adv/blob/9c9579a4ea3f69a8c24ed1cc58f4a517119136c8/docs/IncreFlashAttention.md)
- 参考源码提交：`e0f6933182bcc757ebcde538fbd635bbc7deadfb`

## 第一阶段：正确性闭环

完成内容：

1. 使用 mask 识别真实有效上下文，避免把 llama.cpp 为复用计算图保留的 KV 存储长度当作有效 token 数。
2. 修复 MTE2 搬运覆盖临时 KV 缓冲区、而 Vector Cast 尚未读取完成的竞争问题。
3. 将 Reduce 临时空间扩展到覆盖 8192 token，避免长上下文越界。
4. 独立探针同时执行 NPU 算子和 CPU Attention 参考实现，并逐元素比较结果。
5. llama.cpp 端只允许 `head_dim = 64`、单 token Query、上下文不超过 8192 的图走自定义算子，其余配置自动回退。

Qwen 实际 head 配置下的验证结果：

| 有效上下文 | 最大绝对误差 | 结果 |
| ---: | ---: | :---: |
| 1 | 0 | PASS |
| 128 | 3.78676e-06 | PASS |
| 129 | 3.08827e-06 | PASS |
| 257 | 1.30292e-06 | PASS |
| 1024 | 4.60772e-07 | PASS |
| 2048 | 2.31666e-07 | PASS |
| 4096 | 5.32018e-08 | PASS |
| 8192 | 3.84898e-08 | PASS |

此外，127、255、256、511、512、513 均通过。257 和 8192 分别连续运行 5 次，误差完全一致。GQA 组合 `8/2`、`16/4`、`32/8`（Q heads/KV heads）也分别连续运行 3 次通过。

端到端验证使用同一提示词、温度、随机种子和输出长度，对比自定义路径开启/关闭后的文本文件，`cmp` 返回 0，输出逐字节一致。

## 第二阶段：KV 分块搬运与双缓冲

原实现使用一块 `TBuf` 配合手写事件同步。当前实现改为 AscendC 队列生命周期：

```text
AllocTensor → DataCopy → EnQue → DeQue → Vector compute → FreeTensor
```

具体调整：

- KV 输入使用 `TQue<QuePosition::VECIN, 2>`，队列深度为 2。
- K 和 V 均按 tile 搬入 Local Memory 后再转 FP32 计算。
- tile 从 128 调整为 240：保持 8 元素对齐并处于 Reduce repeat 限制内，同时减少长上下文的分块次数。
- 不再依赖 K/V 路径中的手写 MTE2/Vector 事件，缓冲区所有权由队列管理。

边界 239、240、241，以及 257、1024、8192 均连续运行 3 次通过 CPU 对照，端到端输出仍与普通路径逐字节一致。

## 性能结果

统一使用 Qwen2.5-0.5B FP16、Ascend 310B4、CANN 8.5、`llama-bench` generation 128 tokens、重复 5 次：

| 路径 | token/s | 说明 |
| --- | ---: | --- |
| CANN 普通 Attention | 11.35 | 当前对照基线 |
| 正确但使用早期 fence 的自定义版本 | 10.71 | 第一阶段稳定基线 |
| TQue 双缓冲，tile 128 | 10.35 | 队列开销未被充分摊薄 |
| TQue 双缓冲，tile 240 | 11.10 | 当前第二阶段版本 |

第二阶段相对第一阶段稳定版本提升约 3.64%，但仍比普通 Attention 慢约 2.20%。因此当前结论是：适配已经正确、稳定并接近普通路径，尚不能声称性能领先。

一次 32-token profiling 中，`AttentionStepCustom` 执行 792 次，总耗时 260185.833 us，平均 328.517 us，占采样总耗时 7.589%。板端原始报告位于：

```text
/home/HwHiAiUser/profiling_llamacpp_tque240_20260914/
```

## 当前边界与下一步

- 当前原型只覆盖目标模型需要的 `head_dim = 64`；80/128 未作为本阶段适配目标。
- 双缓冲队列已经建立，但当前 K/V 的搬运与计算仍在单个逻辑流程中，尚未达到成熟 Flash Attention 的充分流水重叠。
- 下一阶段优先分析单次 328 us 内部的 Vector、搬运和同步占比，再决定是否做更深的分块流水、减少标量同步或改用 Cube 计算。
- 在性能超过普通路径之前，建议保留回退能力，并把该版本标记为“可运行原型、性能尚未优于基线”。

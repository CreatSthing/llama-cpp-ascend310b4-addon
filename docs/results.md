# 实测结果

测试环境：Qwen2.5-1.5B-Instruct、Ascend 310B4、CANN 8.5.0、单用户 Decode、`batch=1`、`ubatch=1`，性能结果重复 3 次。

| 阶段 | 结果 |
| --- | ---: |
| 初始 FP16 Decode | 1.42 tokens/s |
| 常规权重 NZ 预排布 | 3.40 tokens/s |
| 共享输出权重 ND/NZ 双布局 | 4.55 tokens/s |
| W8A8 + SmoothQuant 候选 | 5.40 tokens/s |
| FP16 `pp512` | 121.24 tokens/s，独立启动 5/5 通过 |

W8A8 + SmoothQuant 快速精度门槛：

- 英文首选 Token 一致率：99.22%。
- 中文首选 Token 一致率：98.04%。
- 英文 KL：0.001509；中文 KL：0.002075。
- 固定 20 题：16/20，与对照候选一致。

W4A16 单 Token GEMV 探针由 140.80 ms 优化到 8.06 ms，余弦相似度为 0.99998218；但仍慢于同形状 FP16 的 4.26 ms，因此保留为实验代码，没有接入整模型主线。

这些结果只适用于上述模型、版本和测试条件。长语料与完整任务集验证尚未完成。

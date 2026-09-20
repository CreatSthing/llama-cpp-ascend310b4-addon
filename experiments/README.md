# 实验算子

- `a8w8/`：动态激活量化、INT8 矩阵乘和反量化链路探针。
- `attention-step-custom/`：单步 Attention/GQA 自定义算子源码。
- `w4a16-gemv/`：Qwen2.5-1.5B FFN 形状的 W4A16 单 Token GEMV 探针。

这些目录用于说明和复现实验，不会由补丁安装脚本自动编译。A8W8 和 Attention 源码需要放入完整的 AscendC 自定义算子工程中生成 OPP；W4A16 尚未接入 llama.cpp。

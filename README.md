# llama.cpp Ascend 310B4 Add-on

这是一个面向 Ascend 310B4 的 llama.cpp 适配包。仓库只保存补丁、安装脚本、验证资料和实验算子，不包含完整的 llama.cpp 源码。

## 已实现

- 修复大批量 Prefill 中临时缓冲区提前复用导致的崩溃。
- 为 Decode 增加 FP16 权重的 FRACTAL_NZ 预排布和共享输出权重双布局。
- 增加 310B4 单 Token A8W8 混合量化、逐层控制及 SmoothQuant 接口。
- 增加 Attention/GQA 实验接口、精度评测脚本和性能基线脚本。
- 附带 A8W8、Attention 和 W4A16 的 AscendC 实验源码。

实测 Qwen2.5-1.5B 的 Decode 速度由 1.42 提升至 5.40 tokens/s。详细条件和精度结果见 [docs/results.md](docs/results.md)。

## 安装

当前补丁固定适配官方 llama.cpp 提交：

```text
37b3a9e0ccba261d1cc245a971deae0b18c201ab
```

先克隆官方仓库并切换到该版本：

```bash
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 37b3a9e0ccba261d1cc245a971deae0b18c201ab
git clone https://github.com/CreatSthing/llama-cpp-ascend310b4-addon.git
```

Linux：

```bash
./llama-cpp-ascend310b4-addon/scripts/install.sh ./llama.cpp
```

Windows PowerShell：

```powershell
pwsh ./llama-cpp-ascend310b4-addon/scripts/install.ps1 -LlamaCppPath ./llama.cpp
```

安装脚本会检查目标版本和工作区状态，再将补丁应用到 llama.cpp。安装后可自行检查差异并提交到个人分支。

## 构建与运行

在 310B4 环境中加载 CANN 后构建：

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -B build-cann85 -DGGML_CANN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cann85 -j
```

启用已验证的 FP16 NZ 路径：

```bash
GGML_CANN_WEIGHT_NZ=on \
GGML_CANN_TIED_OUTPUT_NZ=on \
./build-cann85/bin/llama-bench -m MODEL.gguf -p 0 -n 64 -r 3 -ngl 99 -fa off
```

A8W8 路径还需要预先构建对应的自定义 OPP，并通过 `GGML_CANN_CUSTOM_OPS_DIR` 传给 CMake。`experiments/` 中保存了关键算子和探针源码，但不是预编译安装包。

## 卸载

目标工作区未继续修改时，可反向应用同一补丁：

```bash
./llama-cpp-ascend310b4-addon/scripts/uninstall.sh ./llama.cpp
```

或：

```powershell
pwsh ./llama-cpp-ascend310b4-addon/scripts/uninstall.ps1 -LlamaCppPath ./llama.cpp
```

## 目录

```text
patches/       llama.cpp 适配补丁
scripts/       安装与卸载脚本
tools/         SmoothQuant 标定工具
experiments/   AscendC 算子及探针源码
docs/          实测结果和适用范围
```

## 限制

- 补丁针对固定的 llama.cpp 基线；较新版本可能需要重新合并。
- W4A16 当前是独立算子验证，尚未接入 llama.cpp 整模型。
- 模型权重、编译产物和性能日志不包含在本仓库中。

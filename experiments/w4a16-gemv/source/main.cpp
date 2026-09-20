#include "data_utils.h"
#include "acl/acl.h"

#include <chrono>
#include <cstdio>

namespace {
constexpr size_t kInputBytes = 1536 * 448 * sizeof(uint16_t);
constexpr size_t kWeightBytes = 8960 * 1536 / 2;
constexpr size_t kScaleBytes = 8960 * (1536 / 32) * sizeof(uint16_t);
constexpr size_t kOutputBytes = 8960 * sizeof(uint16_t);
constexpr int kMeasuredRuns = 3;
}

int32_t main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t * inputHost;
    uint8_t * weightHost;
    uint8_t * scaleHost;
    uint8_t * outputHost;
    uint8_t * inputDevice;
    uint8_t * weightDevice;
    uint8_t * scaleDevice;
    uint8_t * outputDevice;
    CHECK_ACL(aclrtMallocHost((void **)&inputHost, kInputBytes));
    CHECK_ACL(aclrtMallocHost((void **)&weightHost, kWeightBytes));
    CHECK_ACL(aclrtMallocHost((void **)&scaleHost, kScaleBytes));
    CHECK_ACL(aclrtMallocHost((void **)&outputHost, kOutputBytes));
    CHECK_ACL(aclrtMalloc((void **)&inputDevice, kInputBytes,
                         ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&weightDevice, kWeightBytes,
                         ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&scaleDevice, kScaleBytes,
                         ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outputDevice, kOutputBytes,
                         ACL_MEM_MALLOC_HUGE_FIRST));

    size_t inputBytes = kInputBytes;
    size_t weightBytes = kWeightBytes;
    size_t scaleBytes = kScaleBytes;
    ReadFile("./input/input_x.bin", inputBytes, inputHost, kInputBytes);
    ReadFile("./input/input_weight.bin", weightBytes, weightHost, kWeightBytes);
    ReadFile("./input/input_scales.bin", scaleBytes, scaleHost, kScaleBytes);
    CHECK_ACL(aclrtMemcpy(inputDevice, kInputBytes, inputHost, kInputBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(weightDevice, kWeightBytes, weightHost, kWeightBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(scaleDevice, kScaleBytes, scaleHost, kScaleBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtBinHandle binHandle = nullptr;
    aclrtFuncHandle funcHandle = nullptr;
    aclrtArgsHandle argsHandle = nullptr;
    aclrtParamHandle paramHandle = nullptr;
    const char * filePath = "./out/fatbin/ascendc_kernels/ascendc_kernels.o";
    CHECK_ACL(aclrtBinaryLoadFromFile(filePath, nullptr, &binHandle));
    CHECK_ACL(aclrtBinaryGetFunction(binHandle, "add_custom", &funcHandle));
    CHECK_ACL(aclrtKernelArgsInit(funcHandle, &argsHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&inputDevice,
                                   sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&weightDevice,
                                   sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&scaleDevice,
                                   sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&outputDevice,
                                   sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsFinalize(argsHandle));

    for (int run = 0; run <= kMeasuredRuns; ++run) {
        const auto start = std::chrono::steady_clock::now();
        CHECK_ACL(aclrtLaunchKernelWithConfig(funcHandle, 8, stream, nullptr,
                                             argsHandle, nullptr));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        const auto end = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        if (run > 0) {
            std::printf("run=%d kernel_ms=%.3f\n", run, ms);
        }
    }

    CHECK_ACL(aclrtMemcpy(outputHost, kOutputBytes, outputDevice, kOutputBytes,
                         ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_y.bin", outputHost, kOutputBytes);

    CHECK_ACL(aclrtBinaryUnLoad(binHandle));
    CHECK_ACL(aclrtFree(inputDevice));
    CHECK_ACL(aclrtFree(weightDevice));
    CHECK_ACL(aclrtFree(scaleDevice));
    CHECK_ACL(aclrtFree(outputDevice));
    CHECK_ACL(aclrtFreeHost(inputHost));
    CHECK_ACL(aclrtFreeHost(weightHost));
    CHECK_ACL(aclrtFreeHost(scaleHost));
    CHECK_ACL(aclrtFreeHost(outputHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}

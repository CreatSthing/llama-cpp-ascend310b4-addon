#include <acl/acl.h>
#include <aclnnop/aclnn_quant_matmul_v3.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_ACL(expr) do { const auto status = (expr); if (status != ACL_SUCCESS) { std::fprintf(stderr, "%s failed: %d (%s)\n", #expr, status, aclGetRecentErrMsg()); std::exit(1); } } while (0)

struct DeviceBuffer {
    void * ptr = nullptr;
    explicit DeviceBuffer(size_t size) { CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST)); }
    ~DeviceBuffer() { if (ptr) { aclrtFree(ptr); } }
};

static aclTensor * make_tensor(void * data, aclDataType type, const std::vector<int64_t> & shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int) shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), type, strides.data(), 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), data);
}

static void run_shape(int64_t k, int64_t n, aclrtStream stream) {
    const int64_t m = 1;
    std::vector<int8_t> host_x(m * k, 1);
    std::vector<int8_t> host_w(k * n, 1);
    std::vector<uint64_t> host_quant_param(n, UINT64_C(0x3f800000));
    std::vector<uint16_t> host_out(m * n);

    DeviceBuffer x_bytes(host_x.size());
    DeviceBuffer w_bytes(host_w.size());
    DeviceBuffer quant_param_bytes(host_quant_param.size() * sizeof(uint64_t));
    DeviceBuffer out_bytes(host_out.size() * sizeof(uint16_t));
    CHECK_ACL(aclrtMemcpy(x_bytes.ptr, host_x.size(), host_x.data(), host_x.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(w_bytes.ptr, host_w.size(), host_w.data(), host_w.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(quant_param_bytes.ptr, host_quant_param.size() * sizeof(uint64_t), host_quant_param.data(),
                          host_quant_param.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor * x = make_tensor(x_bytes.ptr, ACL_INT8, {m, k});
    aclTensor * w = make_tensor(w_bytes.ptr, ACL_INT8, {k, n});
    aclTensor * quant_param = make_tensor(quant_param_bytes.ptr, ACL_UINT64, {n});
    aclTensor * out = make_tensor(out_bytes.ptr, ACL_FLOAT16, {m, n});

    uint64_t workspace_size = 0;
    aclOpExecutor * executor = nullptr;

    void * matmul_workspace = nullptr;
    size_t matmul_workspace_capacity = 0;
    auto execute = [&]() {
        workspace_size = 0;
        executor = nullptr;
        CHECK_ACL(aclnnQuantMatmulV3GetWorkspaceSize(x, w, quant_param, nullptr, nullptr, false, false,
                                                      out, &workspace_size, &executor));
        if (workspace_size > matmul_workspace_capacity) {
            if (matmul_workspace) { CHECK_ACL(aclrtFree(matmul_workspace)); }
            CHECK_ACL(aclrtMalloc(&matmul_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
            matmul_workspace_capacity = workspace_size;
        }
        CHECK_ACL(aclnnQuantMatmulV3(matmul_workspace, workspace_size, executor, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    };

    for (int i = 0; i < 3; ++i) { execute(); }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) { execute(); }
    const auto stop = std::chrono::steady_clock::now();

    CHECK_ACL(aclrtMemcpy(host_out.data(), host_out.size() * sizeof(uint16_t), out_bytes.ptr,
                          host_out.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    const double average_us = std::chrono::duration<double, std::micro>(stop - start).count() / 20.0;
    std::printf("shape=[1,%lld]x[%lld,%lld] average=%.2f us first_fp16_bits=0x%04x expected=%lld\n",
                (long long) k, (long long) k, (long long) n, average_us, host_out[0], (long long) k);

    if (matmul_workspace) { CHECK_ACL(aclrtFree(matmul_workspace)); }
    aclDestroyTensor(out);
    aclDestroyTensor(quant_param);
    aclDestroyTensor(w);
    aclDestroyTensor(x);
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    run_shape(1536, 8960, stream);
    run_shape(8960, 1536, stream);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}

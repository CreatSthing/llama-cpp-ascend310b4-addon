#include <acl/acl.h>
#include <aclnn_matmul_w8a8_i32_custom.h>
#include <aclnn_w8a8_dequant_custom.h>
#include <aclnn_w8a8_quantize_custom.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_ACL(expr) do { \
    const auto status = (expr); \
    if (status != ACL_SUCCESS) { \
        std::fprintf(stderr, "%s failed: %d (%s)\n", #expr, status, aclGetRecentErrMsg()); \
        std::exit(1); \
    } \
} while (0)

struct DeviceBuffer {
    void * ptr = nullptr;
    size_t size = 0;

    explicit DeviceBuffer(size_t bytes) : size(bytes) {
        if (bytes > 0) {
            CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        }
    }

    ~DeviceBuffer() {
        if (ptr) {
            aclrtFree(ptr);
        }
    }
};

struct Workspace {
    void * ptr = nullptr;
    uint64_t capacity = 0;

    void reserve(uint64_t bytes) {
        if (bytes <= capacity) {
            return;
        }
        if (ptr) {
            CHECK_ACL(aclrtFree(ptr));
        }
        CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        capacity = bytes;
    }

    ~Workspace() {
        if (ptr) {
            aclrtFree(ptr);
        }
    }
};

static aclTensor * make_tensor(void * data, aclDataType type, const std::vector<int64_t> & shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), type, strides.data(), 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), data);
}

static float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
    uint32_t exponent = (h >> 10) & 0x1fU;
    uint32_t mantissa = h & 0x03ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffU;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static void run_shape(int64_t k, int64_t n, aclrtStream stream) {
    const int64_t m = 1;
    const uint16_t fp16_one = 0x3c00;
    std::vector<uint16_t> host_x(m * k, fp16_one);
    std::vector<int8_t> host_w(k * n);
    std::vector<int8_t> host_w_nz(k * n);
    int64_t expected_first = 0;
    for (int64_t row = 0; row < k; ++row) {
        for (int64_t column = 0; column < n; ++column) {
            const int8_t value = static_cast<int8_t>((row % 7) - 3 + (column % 3));
            host_w[row * n + column] = value;
            const size_t nz_index = ((((size_t) column / 32) * ((size_t) k / 16) + (size_t) row / 16) * 16 +
                                     (size_t) row % 16) * 32 + (size_t) column % 32;
            host_w_nz[nz_index] = value;
            if (column == 0) {
                expected_first += value;
            }
        }
    }
    std::vector<uint16_t> host_w_scale(n, fp16_one);
    std::vector<int8_t> host_xq(m * k);
    std::vector<uint16_t> host_x_scale(1);
    std::vector<uint16_t> host_out(m * n);

    DeviceBuffer x_bytes(host_x.size() * sizeof(uint16_t));
    DeviceBuffer xq_bytes(host_xq.size());
    DeviceBuffer x_scale_bytes(sizeof(uint16_t));
    DeviceBuffer w_bytes(host_w.size());
    DeviceBuffer w_nz_bytes(host_w.size());
    DeviceBuffer w_scale_bytes(host_w_scale.size() * sizeof(uint16_t));
    DeviceBuffer acc_bytes(m * n * sizeof(int32_t));
    DeviceBuffer out_bytes(host_out.size() * sizeof(uint16_t));

    CHECK_ACL(aclrtMemcpy(x_bytes.ptr, x_bytes.size, host_x.data(), x_bytes.size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(w_bytes.ptr, w_bytes.size, host_w.data(), w_bytes.size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(w_nz_bytes.ptr, w_nz_bytes.size, host_w_nz.data(), w_nz_bytes.size,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(w_scale_bytes.ptr, w_scale_bytes.size, host_w_scale.data(), w_scale_bytes.size,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor * x = make_tensor(x_bytes.ptr, ACL_FLOAT16, {m, k});
    aclTensor * xq = make_tensor(xq_bytes.ptr, ACL_INT8, {m, k});
    aclTensor * x_scale = make_tensor(x_scale_bytes.ptr, ACL_FLOAT16, {1});
    aclTensor * w = make_tensor(w_nz_bytes.ptr, ACL_INT8, {k, n});
    aclTensor * w_scale = make_tensor(w_scale_bytes.ptr, ACL_FLOAT16, {n});
    aclTensor * acc = make_tensor(acc_bytes.ptr, ACL_INT32, {m, n});
    aclTensor * out = make_tensor(out_bytes.ptr, ACL_FLOAT16, {m, n});


    Workspace quant_workspace;
    Workspace matmul_workspace;
    Workspace dequant_workspace;

    auto execute = [&]() {
        uint64_t workspace_size = 0;
        aclOpExecutor * executor = nullptr;
        CHECK_ACL(aclnnW8a8QuantizeCustomGetWorkspaceSize(x, xq, x_scale, &workspace_size, &executor));
        quant_workspace.reserve(workspace_size);
        CHECK_ACL(aclnnW8a8QuantizeCustom(quant_workspace.ptr, workspace_size, executor, stream));

        workspace_size = 0;
        executor = nullptr;
        CHECK_ACL(aclnnMatmulW8a8I32CustomGetWorkspaceSize(xq, w, acc, &workspace_size, &executor));
        matmul_workspace.reserve(workspace_size);
        CHECK_ACL(aclnnMatmulW8a8I32Custom(matmul_workspace.ptr, workspace_size, executor, stream));

        workspace_size = 0;
        executor = nullptr;
        CHECK_ACL(aclnnW8a8DequantCustomGetWorkspaceSize(acc, x_scale, w_scale, out, &workspace_size, &executor));
        dequant_workspace.reserve(workspace_size);
        CHECK_ACL(aclnnW8a8DequantCustom(dequant_workspace.ptr, workspace_size, executor, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    };

    for (int i = 0; i < 3; ++i) {
        execute();
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) {
        execute();
    }
    const auto stop = std::chrono::steady_clock::now();

    CHECK_ACL(aclrtMemcpy(host_xq.data(), host_xq.size(), xq_bytes.ptr, xq_bytes.size, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(host_x_scale.data(), sizeof(uint16_t), x_scale_bytes.ptr, sizeof(uint16_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(host_out.data(), host_out.size() * sizeof(uint16_t), out_bytes.ptr, out_bytes.size,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    const double average_us = std::chrono::duration<double, std::micro>(stop - start).count() / 20.0;
    const float x_scale_value = half_to_float(host_x_scale[0]);
    const float out_value = half_to_float(host_out[0]);
    const bool correct = host_xq[0] == 127 && std::fabs(out_value - static_cast<float>(expected_first)) <= 8.0f;
    std::printf("shape=[1,%lld]x[%lld,%lld] chain_average=%.2f us xq=%d x_scale=%.8f out=%.1f expected=%lld %s\n",
                static_cast<long long>(k), static_cast<long long>(k), static_cast<long long>(n), average_us,
                static_cast<int>(host_xq[0]), x_scale_value, out_value, static_cast<long long>(expected_first),
                correct ? "PASS" : "FAIL");

    aclDestroyTensor(out);
    aclDestroyTensor(acc);
    aclDestroyTensor(w_scale);
    aclDestroyTensor(w);
    aclDestroyTensor(x_scale);
    aclDestroyTensor(xq);
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

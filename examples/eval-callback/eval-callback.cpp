#include "arg.h"
#include "common.h"
#include "debug.h"
#include "ggml-backend.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct layer_dump_data {
    FILE * file = nullptr;
    std::vector<uint8_t> raw;
    std::vector<float> values;

    explicit layer_dump_data(const char * path) {
        file = std::fopen(path, "wb");
        if (file == nullptr) {
            throw std::runtime_error(std::string("cannot open layer dump: ") + path);
        }
        const char magic[8] = {'L', 'L', 'D', 'U', 'M', 'P', '1', '\0'};
        std::fwrite(magic, sizeof(magic), 1, file);
    }

    ~layer_dump_data() {
        if (file != nullptr) {
            std::fclose(file);
        }
    }
};

static bool is_layer_output(const char * name) {
    return std::strncmp(name, "ffn_out-", 8) == 0 || std::strncmp(name, "l_out-", 6) == 0;
}

static bool layer_dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool wanted = is_layer_output(t->name);
    if (ask || !wanted) {
        return wanted;
    }
    if (!ggml_is_contiguous(t)) {
        LOG_WRN("skipping non-contiguous layer tensor %s\n", t->name);
        return true;
    }

    auto & dump = *(layer_dump_data *) user_data;
    const int64_t count = ggml_nelements(t);
    dump.raw.resize(ggml_nbytes(t));
    const uint8_t * src = nullptr;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        src = (const uint8_t *) t->data;
    } else {
        ggml_backend_tensor_get(t, dump.raw.data(), 0, dump.raw.size());
        src = dump.raw.data();
    }

    dump.values.resize(count);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(dump.values.data(), src, count * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const auto * data = (const ggml_fp16_t *) src;
        for (int64_t i = 0; i < count; ++i) dump.values[i] = ggml_fp16_to_fp32(data[i]);
    } else if (t->type == GGML_TYPE_BF16) {
        const auto * data = (const ggml_bf16_t *) src;
        for (int64_t i = 0; i < count; ++i) dump.values[i] = ggml_bf16_to_fp32(data[i]);
    } else {
        LOG_WRN("skipping unsupported layer tensor type %s for %s\n", ggml_type_name(t->type), t->name);
        return true;
    }

    const uint16_t name_len = (uint16_t) std::strlen(t->name);
    const uint64_t n_values = (uint64_t) count;
    std::fwrite(&name_len, sizeof(name_len), 1, dump.file);
    std::fwrite(t->name, name_len, 1, dump.file);
    std::fwrite(&n_values, sizeof(n_values), 1, dump.file);
    std::fwrite(dump.values.data(), sizeof(float), count, dump.file);
    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    const int n_batch = std::max(1, params.n_batch);
    for (size_t pos = 0; pos < tokens.size(); pos += n_batch) {
        const int count = std::min<int>(n_batch, tokens.size() - pos);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + pos, count))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_debug_cb_user_data cb_data;
    std::unique_ptr<layer_dump_data> layer_dump;

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    const char * dump_path = std::getenv("LLAMA_LAYER_DUMP");
    if (dump_path != nullptr && dump_path[0] != '\0') {
        layer_dump = std::make_unique<layer_dump_data>(dump_path);
        params.cb_eval = layer_dump_cb;
        params.cb_eval_user_data = layer_dump.get();
    } else {
        params.cb_eval = common_debug_cb_eval;
        params.cb_eval_user_data = &cb_data;
    }
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}

#if !defined(_WIN32) && !defined(_FILE_OFFSET_BITS)
#  define _FILE_OFFSET_BITS 64
#endif

#include "quantize.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#  include <sys/types.h>
#endif

namespace {

gqz_log_callback g_log_cb        = nullptr;
void *           g_log_user_data = nullptr;

void log_msg(gqz_log_level level, const char * fmt, ...) {
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (g_log_cb) {
        g_log_cb(level, buf, g_log_user_data);
        return;
    }
    const char * prefix = level == GQZ_LOG_ERROR ? "error" : level == GQZ_LOG_WARN ? "warning" : "info";
    fprintf(stderr, "%s: %s\n", prefix, buf);
}

#define LOG_INFO(...)  log_msg(GQZ_LOG_INFO, __VA_ARGS__)
#define LOG_WARN(...)  log_msg(GQZ_LOG_WARN, __VA_ARGS__)
#define LOG_ERROR(...) log_msg(GQZ_LOG_ERROR, __VA_ARGS__)

std::string to_lower(const std::string & s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::tolower(c); });
    return r;
}

std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return "";
    }
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t pos = s.find(sep, start);
        if (pos == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

// ggml_type values that ggml_quantize_chunk() knows how to *produce*.
// (everything else in the enum is either an intermediate-only format
// like Q8_K/Q8_1, an integer/F64 type, or a removed legacy variant.)
bool is_valid_quantize_target(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// best-effort: can we dequantize (to_float) this source type at all?
bool can_dequantize(ggml_type t) {
    if (t == GGML_TYPE_F32) {
        return true;
    }
    const ggml_type_traits * tt = ggml_get_type_traits(t);
    return tt != nullptr && tt->to_float != nullptr;
}

ggml_type parse_ggml_type(const std::string & name_in) {
    std::string name = to_lower(trim(name_in));
    if (name.empty()) {
        return GGML_TYPE_COUNT;
    }
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        ggml_type t = (ggml_type) i;
        const char * tn = ggml_type_name(t);
        if (tn == nullptr) {
            continue;
        }
        std::string tns = to_lower(tn);
        if (tns.find("removed") != std::string::npos || tns.find("deprecated") != std::string::npos) {
            continue;
        }
        if (tns == name) {
            return t;
        }
    }
    return GGML_TYPE_COUNT;
}

struct TensorRule {
    std::string pattern_str;
    std::regex  pattern;
    ggml_type   type;
};

std::vector<TensorRule> parse_tensor_type_rules(const std::string & rules_str) {
    std::vector<TensorRule> rules;
    for (const std::string & item : split(rules_str, ',')) {
        std::string trimmed = trim(item);
        if (trimmed.empty()) {
            continue;
        }
        size_t eq = trimmed.find('=');
        if (eq == std::string::npos || eq == 0 || eq == trimmed.size() - 1) {
            LOG_WARN("ignoring malformed --tensor-type-rules entry '%s'", trimmed.c_str());
            continue;
        }
        std::string pattern_str = trimmed.substr(0, eq);
        std::string type_str    = trimmed.substr(eq + 1);
        ggml_type   type        = parse_ggml_type(type_str);
        if (type == GGML_TYPE_COUNT || !is_valid_quantize_target(type)) {
            LOG_WARN("ignoring --tensor-type-rules entry with unknown type '%s'", type_str.c_str());
            continue;
        }
        try {
            TensorRule rule;
            rule.pattern_str = pattern_str;
            rule.pattern     = std::regex(pattern_str);
            rule.type        = type;
            rules.push_back(std::move(rule));
        } catch (const std::regex_error & e) {
            LOG_WARN("ignoring --tensor-type-rules entry with invalid regex '%s': %s", pattern_str.c_str(), e.what());
        }
    }
    return rules;
}

ggml_type decide_target_type(const std::string & name, int n_dims, int64_t ne0, ggml_type default_type,
                              const std::vector<TensorRule> & rules, bool & explicit_match) {
    explicit_match = false;
    ggml_type target = default_type;
    for (const TensorRule & rule : rules) {
        if (std::regex_search(name, rule.pattern)) {
            target        = rule.type;
            explicit_match = true;
            break;
        }
    }

    auto block_ok = [](int64_t ne0_, ggml_type t) {
        return !ggml_is_quantized(t) || (ne0_ % ggml_blck_size(t) == 0);
    };

    if (explicit_match) {
        return target;
    }

    // default (--type) path: only touch >=2D tensors when quantizing to a
    // block-quantized type, mirroring llama.cpp/stable-diffusion.cpp convention
    // of leaving 1D norm/bias vectors at full precision unless explicitly overridden.
    if (ggml_is_quantized(target) && (n_dims < 2 || !block_ok(ne0, target))) {
        return GGML_TYPE_COUNT; // signal: keep original type
    }
    return target;
}

// dequantize `n` elements of `src_type` into a float buffer.
bool to_f32(ggml_type src_type, const void * src, float * dst, int64_t n) {
    if (src_type == GGML_TYPE_F32) {
        memcpy(dst, src, n * sizeof(float));
        return true;
    }
    const ggml_type_traits * tt = ggml_get_type_traits(src_type);
    if (tt == nullptr || tt->to_float == nullptr) {
        return false;
    }
    tt->to_float(src, dst, n);
    return true;
}

// convert a single tensor's raw bytes from src_type to dst_type.
// returns the number of bytes written to `dst`, or 0 on failure.
size_t convert_tensor_data(ggml_type src_type, const void * src, ggml_type dst_type, void * dst, int64_t nrows,
                            int64_t n_per_row, int n_threads) {
    const size_t dst_row_size = ggml_row_size(dst_type, n_per_row);
    const size_t dst_bytes    = dst_row_size * nrows;

    if (src_type == dst_type) {
        memcpy(dst, src, dst_bytes);
        return dst_bytes;
    }

    n_threads = std::max(1, n_threads);

    std::vector<float> f32_buf;
    const float *       f32_src;
    if (src_type == GGML_TYPE_F32) {
        f32_src = (const float *) src;
    } else {
        const int64_t n = nrows * n_per_row;
        f32_buf.resize(n);
        if ((int64_t) n_threads > nrows) {
            n_threads = (int) std::max<int64_t>(1, nrows);
        }
        std::vector<std::thread> pool;
        const int64_t            rows_per_thread = (nrows + n_threads - 1) / n_threads;
        const ggml_type_traits * tt              = ggml_get_type_traits(src_type);
        for (int t = 0; t < n_threads; ++t) {
            const int64_t row0 = t * rows_per_thread;
            const int64_t row1 = std::min(nrows, row0 + rows_per_thread);
            if (row0 >= row1) {
                break;
            }
            pool.emplace_back([=, &f32_buf]() {
                tt->to_float((const char *) src + row0 * ggml_row_size(src_type, n_per_row),
                             f32_buf.data() + row0 * n_per_row, (row1 - row0) * n_per_row);
            });
        }
        for (auto & th : pool) {
            th.join();
        }
        f32_src = f32_buf.data();
    }

    // dummy (uniform) importance matrix: needed by a few quant kernels that
    // require a non-null imatrix, harmless no-op weighting for the rest.
    std::vector<float> imatrix(n_per_row, 1.0f);

    if ((int64_t) n_threads > nrows) {
        n_threads = (int) std::max<int64_t>(1, nrows);
    }
    std::vector<std::thread> pool;
    const int64_t            rows_per_thread = (nrows + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        const int64_t row0 = t * rows_per_thread;
        const int64_t row1 = std::min(nrows, row0 + rows_per_thread);
        if (row0 >= row1) {
            break;
        }
        pool.emplace_back([=, &imatrix]() {
            ggml_quantize_chunk(dst_type, f32_src, dst, row0 * n_per_row, row1 - row0, n_per_row, imatrix.data());
        });
    }
    for (auto & th : pool) {
        th.join();
    }

    return dst_bytes;
}

struct FileCloser {
    FILE * f;
    ~FileCloser() {
        if (f) {
            fclose(f);
        }
    }
};

// 64-bit-safe seek: plain fseek() takes a `long`, which is only 32 bits on
// Windows (both MSVC and MinGW) even in 64-bit builds, silently truncating
// offsets beyond ~2GB.
bool seek64(FILE * f, int64_t offset) {
#if defined(_WIN32)
    return _fseeki64(f, offset, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t) offset, SEEK_SET) == 0;
#endif
}

// fread()/fwrite() are allowed to transfer fewer bytes than requested in a
// single call for reasons other than EOF/error (observed in practice on
// Windows for large single transfers - antivirus/cloud-filter drivers,
// network/exFAT volumes, etc.). Loop until the full amount is moved, or a
// real EOF/error condition is hit.
bool read_exact(FILE * f, void * buf, size_t n) {
    size_t done = 0;
    while (done < n) {
        size_t chunk = fread((char *) buf + done, 1, n - done, f);
        if (chunk == 0) {
            return false; // real EOF or error
        }
        done += chunk;
    }
    return true;
}

bool write_exact(FILE * f, const void * buf, size_t n) {
    size_t done = 0;
    while (done < n) {
        size_t chunk = fwrite((const char *) buf + done, 1, n - done, f);
        if (chunk == 0) {
            return false;
        }
        done += chunk;
    }
    return true;
}

void write_padding(FILE * f, size_t nbytes, size_t alignment) {
    size_t aligned = ((nbytes + alignment - 1) / alignment) * alignment;
    size_t pad     = aligned - nbytes;
    if (pad == 0) {
        return;
    }
    static const char zeros[64] = {0};
    while (pad > 0) {
        size_t chunk = std::min(pad, sizeof(zeros));
        fwrite(zeros, 1, chunk, f);
        pad -= chunk;
    }
}

} // namespace

extern "C" struct gqz_params gqz_default_params(void) {
    gqz_params p;
    memset(&p, 0, sizeof(p));
    p.n_threads = 0;
    return p;
}

extern "C" int gqz_quantize(const gqz_params * params) {
    if (params == nullptr || params->input_path == nullptr || params->output_path == nullptr ||
        params->default_type == nullptr) {
        LOG_ERROR("missing required parameter (input/output/type)");
        return 1;
    }

    if (strcmp(params->input_path, params->output_path) == 0) {
        LOG_ERROR("input and output paths must be different ('%s')", params->input_path);
        return 1;
    }

    g_log_cb        = params->log_cb;
    g_log_user_data = params->log_user_data;

    ggml_type default_type = parse_ggml_type(params->default_type);
    if (default_type == GGML_TYPE_COUNT || !is_valid_quantize_target(default_type)) {
        LOG_ERROR("invalid --type '%s'", params->default_type);
        return 1;
    }

    std::vector<TensorRule> rules;
    if (params->tensor_type_rules != nullptr && params->tensor_type_rules[0] != '\0') {
        rules = parse_tensor_type_rules(params->tensor_type_rules);
    }

    int n_threads = params->n_threads;
    if (n_threads <= 0) {
        n_threads = (int) std::thread::hardware_concurrency();
        if (n_threads <= 0) {
            n_threads = 4;
        }
    }

    ggml_context * meta_ctx = nullptr;
    gguf_init_params gguf_params = { /*no_alloc =*/ true, /*ctx =*/ &meta_ctx };
    gguf_context * in_ctx = gguf_init_from_file(params->input_path, gguf_params);
    if (in_ctx == nullptr) {
        LOG_ERROR("failed to load gguf file '%s'", params->input_path);
        return 1;
    }

    const int64_t n_tensors      = gguf_get_n_tensors(in_ctx);
    const size_t  data_offset_in = gguf_get_data_offset(in_ctx);

    struct TensorPlan {
        std::string name;
        ggml_type   src_type;
        ggml_type   dst_type;
        int         n_dims;
        int64_t     ne[GGML_MAX_DIMS];
        size_t      src_offset;
        size_t      src_size;
    };

    std::vector<TensorPlan> plans;
    plans.reserve(n_tensors);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(in_ctx, i);
        ggml_tensor * t    = ggml_get_tensor(meta_ctx, name);
        if (t == nullptr) {
            LOG_ERROR("internal error: tensor '%s' missing from metadata context", name);
            gguf_free(in_ctx);
            ggml_free(meta_ctx);
            return 1;
        }

        TensorPlan plan;
        plan.name       = name;
        plan.src_type   = t->type;
        plan.n_dims     = ggml_n_dims(t);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            plan.ne[d] = t->ne[d];
        }
        plan.src_offset = data_offset_in + gguf_get_tensor_offset(in_ctx, i);
        plan.src_size   = gguf_get_tensor_size(in_ctx, i);

        bool explicit_match = false;
        ggml_type target =
            decide_target_type(plan.name, plan.n_dims, plan.ne[0], default_type, rules, explicit_match);

        if (target == GGML_TYPE_COUNT) {
            target = plan.src_type; // default-path decided to leave this tensor untouched
        } else if (explicit_match && ggml_is_quantized(target) && plan.ne[0] % ggml_blck_size(target) != 0) {
            LOG_WARN("tensor '%s': row size %lld not divisible by block size of %s, keeping original type %s",
                     plan.name.c_str(), (long long) plan.ne[0], ggml_type_name(target), ggml_type_name(plan.src_type));
            target = plan.src_type;
        } else if (target != plan.src_type && !can_dequantize(plan.src_type)) {
            LOG_WARN("tensor '%s': cannot dequantize source type %s, keeping original type", plan.name.c_str(),
                     ggml_type_name(plan.src_type));
            target = plan.src_type;
        }

        plan.dst_type = target;
        plans.push_back(std::move(plan));
    }

    // build output gguf context: copy all metadata, register tensors with their target types/shapes
    gguf_context * out_ctx = gguf_init_empty();
    gguf_set_kv(out_ctx, in_ctx);

    size_t out_meta_mem = (size_t) n_tensors * ggml_tensor_overhead() + ggml_graph_overhead() + 4096;
    ggml_init_params out_meta_params = { out_meta_mem, nullptr, /*no_alloc=*/ true };
    ggml_context * out_meta_ctx = ggml_init(out_meta_params);

    for (const TensorPlan & plan : plans) {
        ggml_tensor * t = ggml_new_tensor(out_meta_ctx, plan.dst_type, plan.n_dims, plan.ne);
        ggml_set_name(t, plan.name.c_str());
        gguf_add_tensor(out_ctx, t);
    }

    if (!gguf_write_to_file(out_ctx, params->output_path, /*only_meta=*/ true)) {
        LOG_ERROR("failed to write gguf header to '%s'", params->output_path);
        gguf_free(in_ctx);
        ggml_free(meta_ctx);
        gguf_free(out_ctx);
        ggml_free(out_meta_ctx);
        return 1;
    }

    const size_t alignment = gguf_get_alignment(out_ctx);

    FILE * fin = fopen(params->input_path, "rb");
    if (fin == nullptr) {
        LOG_ERROR("failed to reopen input file '%s'", params->input_path);
        gguf_free(in_ctx);
        ggml_free(meta_ctx);
        gguf_free(out_ctx);
        ggml_free(out_meta_ctx);
        return 1;
    }
    FileCloser fin_closer{ fin };

    FILE * fout = fopen(params->output_path, "ab");
    if (fout == nullptr) {
        LOG_ERROR("failed to reopen output file '%s'", params->output_path);
        gguf_free(in_ctx);
        ggml_free(meta_ctx);
        gguf_free(out_ctx);
        ggml_free(out_meta_ctx);
        return 1;
    }
    FileCloser fout_closer{ fout };

    size_t total_src_bytes = 0;
    size_t total_dst_bytes = 0;
    size_t n_converted      = 0;
    size_t n_copied         = 0;

    std::vector<char> src_buf;
    std::vector<char> dst_buf;

    for (const TensorPlan & plan : plans) {
        src_buf.resize(plan.src_size);
        if (!seek64(fin, (int64_t) plan.src_offset)) {
            LOG_ERROR("failed to seek to tensor '%s' (offset %llu) in input file", plan.name.c_str(),
                      (unsigned long long) plan.src_offset);
            return 1;
        }
        if (!read_exact(fin, src_buf.data(), plan.src_size)) {
            LOG_ERROR("failed to read tensor '%s' from input file (expected %llu bytes at offset %llu)",
                      plan.name.c_str(), (unsigned long long) plan.src_size, (unsigned long long) plan.src_offset);
            return 1;
        }

        const int64_t n_per_row = plan.ne[0];
        int64_t       nrows     = 1;
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            nrows *= plan.ne[d];
        }

        const size_t dst_size = ggml_row_size(plan.dst_type, n_per_row) * nrows;
        dst_buf.resize(dst_size);

        convert_tensor_data(plan.src_type, src_buf.data(), plan.dst_type, dst_buf.data(), nrows, n_per_row,
                             n_threads);

        if (!write_exact(fout, dst_buf.data(), dst_size)) {
            LOG_ERROR("failed to write tensor '%s' to output file", plan.name.c_str());
            return 1;
        }
        write_padding(fout, dst_size, alignment);

        total_src_bytes += plan.src_size;
        total_dst_bytes += dst_size;
        if (plan.dst_type != plan.src_type) {
            n_converted++;
            LOG_INFO("%-60s %10s -> %-10s (%8.2f MB -> %8.2f MB)", plan.name.c_str(), ggml_type_name(plan.src_type),
                     ggml_type_name(plan.dst_type), plan.src_size / 1024.0 / 1024.0, dst_size / 1024.0 / 1024.0);
        } else {
            n_copied++;
        }
    }

    fflush(fout);

    gguf_free(in_ctx);
    ggml_free(meta_ctx);
    gguf_free(out_ctx);
    ggml_free(out_meta_ctx);

    LOG_INFO("done: %lld tensors quantized, %lld copied unchanged", (long long) n_converted, (long long) n_copied);
    LOG_INFO("total size: %.2f MB -> %.2f MB", total_src_bytes / 1024.0 / 1024.0, total_dst_bytes / 1024.0 / 1024.0);

    return 0;
}

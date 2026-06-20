#pragma once

#include <stdint.h>

#if defined(_WIN32) && defined(GGUF_QUANTIZER_SHARED)
#  ifdef GGUF_QUANTIZER_BUILD
#    define GQZ_API __declspec(dllexport)
#  else
#    define GQZ_API __declspec(dllimport)
#  endif
#elif defined(GGUF_QUANTIZER_SHARED)
#  define GQZ_API __attribute__((visibility("default")))
#else
#  define GQZ_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// log levels passed to the optional log callback
enum gqz_log_level {
    GQZ_LOG_INFO = 0,
    GQZ_LOG_WARN = 1,
    GQZ_LOG_ERROR = 2,
};

typedef void (*gqz_log_callback)(enum gqz_log_level level, const char * message, void * user_data);

struct gqz_params {
    const char * input_path;          // -m
    const char * output_path;         // -o
    const char * default_type;        // --type, e.g. "q4_k", "iq4_xs", "mxfp4" (case-insensitive)
    const char * tensor_type_rules;   // --tensor-type-rules "pattern=type,pattern=type,..." (regex, first match wins)
    int          n_threads;           // 0 = auto-detect

    gqz_log_callback log_cb;          // optional, NULL = log to stderr
    void *            log_user_data;
};

GQZ_API struct gqz_params gqz_default_params(void);

// Returns 0 on success, non-zero on failure.
GQZ_API int gqz_quantize(const struct gqz_params * params);

#ifdef __cplusplus
}
#endif

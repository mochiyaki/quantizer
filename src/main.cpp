#include "quantize.h"

#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(const char * argv0) {
    printf("standalone GGUF quantizer\n\n");
    printf("usage: %s -m <input.gguf> -o <output.gguf> --type <type> [options]\n\n", argv0);
    printf("required:\n");
    printf("  -m, --model <file>           input GGUF file\n");
    printf("  -o, --output <file>          output GGUF file\n");
    printf("  --type <type>                default weight type applied to all (eligible) tensors\n\n");
    printf("options:\n");
    printf("  --tensor-type-rules <rules>  comma-separated \"<regex>=<type>\" overrides, matched (in order,\n");
    printf("                                first match wins) against each tensor name with std::regex_search\n");
    printf("  -t, --threads <n>            number of threads to use (default: hardware concurrency)\n");
    printf("  -h, --help                   show this help\n\n");
    printf("supported types:\n");
    printf("  f32, f16, bf16\n");
    printf("  q4_0, q4_1, q5_0, q5_1, q8_0, q1_0\n");
    printf("  q2_k, q3_k, q4_k, q5_k, q6_k\n");
    printf("  iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs\n");
    printf("  tq1_0, tq2_0, mxfp4, nvfp4\n\n");
    printf("examples:\n");
    printf("  %s -m model-f32.gguf -o model-q2_k.gguf --type q2_k\n", argv0);
    printf("  %s -m model-f16.gguf -o model-iq4_xs.gguf --type iq4_xs\n", argv0);
    printf("  %s -m model-bf16.gguf -o model-q4_0.gguf --type q4_0\n", argv0);
    printf("  %s -m model.gguf -o model-mixed.gguf --type q8_0 \\\n"
           "      --tensor-type-rules \"layers.*adaln_modulation.*weight=q8_0,layers.*attention.o.*weight=q4_0,"
           "layers.*attention.qkv.*weight=q5_0,layers.*feed_forward.*weight=q8_0\"\n",
           argv0);
}

static void log_to_stderr(gqz_log_level level, const char * message, void * /*user_data*/) {
    const char * prefix = level == GQZ_LOG_ERROR ? "error" : level == GQZ_LOG_WARN ? "warning" : "info";
    fprintf(stderr, "%s: %s\n", prefix, message);
}

int main(int argc, char ** argv) {
    std::string input_path;
    std::string output_path;
    std::string type_str;
    std::string rules_str;
    int         n_threads = 0;

    auto next_arg = [&](int & i) -> const char * {
        if (i + 1 >= argc) {
            fprintf(stderr, "error: missing value for argument '%s'\n", argv[i]);
            exit(1);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            input_path = next_arg(i);
        } else if (arg == "-o" || arg == "--output") {
            output_path = next_arg(i);
        } else if (arg == "--type") {
            type_str = next_arg(i);
        } else if (arg == "--tensor-type-rules") {
            rules_str = next_arg(i);
        } else if (arg == "-t" || arg == "--threads") {
            n_threads = atoi(next_arg(i));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (input_path.empty() || output_path.empty() || type_str.empty()) {
        fprintf(stderr, "error: -m, -o and --type are all required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    gqz_params params       = gqz_default_params();
    params.input_path       = input_path.c_str();
    params.output_path      = output_path.c_str();
    params.default_type     = type_str.c_str();
    params.tensor_type_rules = rules_str.empty() ? nullptr : rules_str.c_str();
    params.n_threads        = n_threads;
    params.log_cb           = log_to_stderr;

    return gqz_quantize(&params);
}

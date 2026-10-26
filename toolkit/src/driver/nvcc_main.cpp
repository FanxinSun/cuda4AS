#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <mach-o/dyld.h>
#include <libgen.h>
#include <sys/stat.h>

#ifndef CUDA4AS_VERSION
#define CUDA4AS_VERSION "0.1.0"
#endif

#ifndef CUDA_COMPAT_VERSION
#define CUDA_COMPAT_VERSION "12.0.0"
#endif

std::string get_executable_dir() {
    char path[1024];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        char real[1024];
        if (realpath(path, real) != nullptr) {
            char* dir = dirname(real);
            return std::string(dir);
        }
    }
    return "/Users/masshj/system-env/install/cuda4AS-toolkit-0.1.0/bin";
}

std::string get_toolkit_root() {
    const char* env_tk = std::getenv("CUDA4AS_TOOLKIT");
    if (env_tk != nullptr && env_tk[0] != '\0') {
        return std::string(env_tk);
    }
    const char* env_ctk = std::getenv("CUDAToolkit_ROOT");
    if (env_ctk != nullptr && env_ctk[0] != '\0') {
        return std::string(env_ctk);
    }
    std::string bin_dir = get_executable_dir();
    char* copy = strdup(bin_dir.c_str());
    char* parent = dirname(copy);
    std::string root(parent);
    free(copy);
    return root;
}

std::string find_host_cxx(const std::string& explicit_ccbin) {
    if (!explicit_ccbin.empty()) {
        return explicit_ccbin;
    }
    const char* env_cxx = std::getenv("CUDAHOSTCXX");
    if (env_cxx != nullptr && env_cxx[0] != '\0') {
        return std::string(env_cxx);
    }
    const char* env_llvm = std::getenv("CUDA4AS_LLVM");
    if (env_llvm != nullptr && env_llvm[0] != '\0') {
        std::string llvm_clang = std::string(env_llvm) + "/bin/clang++";
        if (access(llvm_clang.c_str(), X_OK) == 0) {
            return llvm_clang;
        }
    }
    if (access("/opt/homebrew/opt/llvm/bin/clang++", X_OK) == 0) {
        return "/opt/homebrew/opt/llvm/bin/clang++";
    }
    return "/usr/bin/clang++";
}

void print_version() {
    std::cout << "nvcc: cuda4AS Cuda compiler driver (Apple Silicon Native)\n"
              << "cuda4AS Version " << CUDA4AS_VERSION << "\n"
              << "Cuda compilation tools, release 12.0, V" << CUDA_COMPAT_VERSION << "\n";
}

void print_help() {
    std::cout << "Usage: nvcc [options] <inputfile>\n\n"
              << "Options:\n"
              << "  -V, --version           Print version information\n"
              << "  -v, --verbose           Print verbose compilation and link commands\n"
              << "  -c                      Compile and assemble, but do not link\n"
              << "  -o <file>               Specify output file name\n"
              << "  -I<dir>                 Add include directory\n"
              << "  -D<macro>               Define macro\n"
              << "  -ccbin <path>           Specify host compiler\n"
              << "  -h, --help              Print this help summary\n";
}

int run_command(const std::vector<std::string>& cmd_args, bool verbose) {
    if (verbose) {
        std::string cmd_str;
        for (size_t i = 0; i < cmd_args.size(); ++i) {
            if (i > 0) cmd_str += " ";
            cmd_str += cmd_args[i];
        }
        std::cerr << "#$ " << cmd_str << std::endl;
    }
    std::vector<char*> argv;
    argv.reserve(cmd_args.size() + 1);
    for (const auto& a : cmd_args) {
        argv.push_back(const_cast<char*>(a.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid == 0) {
        execvp(argv[0], argv.data());
        std::cerr << "nvcc fatal: failed to execute " << argv[0] << std::endl;
        _exit(127);
    } else if (pid > 0) {
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        }
        return 1;
    } else {
        std::cerr << "nvcc fatal: fork failed" << std::endl;
        return 1;
    }
}

bool file_exists_and_not_empty(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return st.st_size > 0;
}

bool file_contains(const std::string& path, const std::string& needle) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find(needle) != std::string::npos) return true;
    }
    return false;
}

int compile_cuda_to_object(
    const std::string& host_cxx,
    const std::string& cu_file,
    const std::string& out_obj,
    const std::string& bin_dir,
    const std::string& inc_dir,
    const std::vector<std::string>& forward_flags,
    bool verbose
) {
    std::string translator = bin_dir + "/cuda4as-translate";
    if (access(translator.c_str(), X_OK) != 0) {
        translator = "/Users/masshj/YFCE/cuda4AS/toolkit/src/translator/cuda4as_translate.py";
    }

    std::string temp_reg_cpp = out_obj + ".cuda4as_reg.cpp";
    std::string temp_reg_o = out_obj + ".cuda4as_reg.o";
    std::string temp_host_ll = out_obj + ".cuda4as_host.ll";
    std::string temp_host_stitched_ll = out_obj + ".cuda4as_host_stitched.ll";
    std::string temp_dev_ll = out_obj + ".cuda4as_dev.ll";
    std::string temp_host_o = out_obj + ".cuda4as_host.o";
    std::string temp_dev_o = out_obj + ".cuda4as_dev.o";

    std::vector<std::string> trans_cmd = {
        "/usr/bin/python3",
        translator,
        cu_file,
        "-r", temp_reg_cpp
    };
    int ret = run_command(trans_cmd, verbose);
    if (ret != 0) {
        unlink(temp_reg_cpp.c_str());
        return ret;
    }

    std::vector<std::string> host_cmd = {
        host_cxx,
        "--cuda-host-only", "-nocudainc", "-nocudalib", "-x", "cuda",
        "-Wno-unknown-cuda-version", "-Wno-deprecated-attributes",
        "-Xclang", "-fcuda-allow-variadic-functions",
        "-include", "cuda_runtime.h",
        "-S", "-emit-llvm", cu_file, "-o", temp_host_ll,
        "-D__CUDACC__=1", "-D__NVCC__=1",
        "-D__CUDACC_VER_MAJOR__=12", "-D__CUDACC_VER_MINOR__=0", "-D__CUDACC_VER_BUILD__=0",
        "-DCUDART_VERSION=12000",
        "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1",
        "-DCCCL_DISABLE_NVTX=1", "-DNVTX_DISABLE=1",
        "-D_CCCL_GRID_CONSTANT=",
        "-I" + inc_dir
    };
    for (const auto& f : forward_flags) host_cmd.push_back(f);
    ret = run_command(host_cmd, verbose);
    if (ret != 0) {
        unlink(temp_reg_cpp.c_str());
        unlink(temp_host_ll.c_str());
        return ret;
    }

    bool has_dev_o = false;
    if (file_contains(temp_host_ll, "__device_stub__")) {
        std::vector<std::string> dev_cmd = {
            host_cxx,
            "--cuda-host-only", "-Xclang", "-fcuda-is-device",
            "-Xclang", "-fcuda-allow-variadic-functions",
            "-nocudainc", "-nocudalib", "-x", "cuda",
            "-Wno-unknown-cuda-version", "-Wno-deprecated-attributes", "-Wno-macro-redefined",
            "-include", "cuda_runtime.h",
            "-S", "-emit-llvm", cu_file, "-o", temp_dev_ll,
            "-D__CUDACC__=1", "-D__NVCC__=1",
            "-D__CUDACC_VER_MAJOR__=12", "-D__CUDACC_VER_MINOR__=0", "-D__CUDACC_VER_BUILD__=0",
            "-DCUDART_VERSION=12000",
            "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1",
            "-DCCCL_DISABLE_NVTX=1", "-DNVTX_DISABLE=1",
            "-D_CCCL_GRID_CONSTANT=",
            "-U__CUDA_ARCH__", "-D__CUDA_ARCH__=800",
            "-U__NVPTX__",
            "-I" + inc_dir
        };
        for (size_t i = 0; i < forward_flags.size(); ++i) {
            const std::string& f = forward_flags[i];
            if (f == "-MD" || f == "-MMD") continue;
            if (f == "-MT" || f == "-MF" || f == "-MQ") {
                ++i;
                continue;
            }
            dev_cmd.push_back(f);
        }
        if (run_command(dev_cmd, verbose) == 0) {
            std::vector<std::string> cp_cmd = { "cp", temp_host_ll, temp_host_stitched_ll };
            run_command(cp_cmd, false);
            std::vector<std::string> stitch_cmd = {
                "/usr/bin/python3",
                translator,
                "--stitch-host-ll", temp_host_stitched_ll,
                "--stitch-dev-ll", temp_dev_ll
            };
            if (run_command(stitch_cmd, verbose) == 0) {
                std::vector<std::string> asm_dev_cmd = {
                    host_cxx, "-O2", "-Wno-override-module", "-c", temp_dev_ll, "-o", temp_dev_o
                };
                if (run_command(asm_dev_cmd, verbose) == 0) {
                    has_dev_o = true;
                    rename(temp_host_stitched_ll.c_str(), temp_host_ll.c_str());
                }
            }
            unlink(temp_host_stitched_ll.c_str());
        }
        unlink(temp_dev_ll.c_str());
    }

    std::vector<std::string> asm_host_cmd = {
        host_cxx, "-O2", "-Wno-override-module", "-c", temp_host_ll, "-o", temp_host_o
    };
    ret = run_command(asm_host_cmd, verbose);
    unlink(temp_host_ll.c_str());
    if (ret != 0) {
        unlink(temp_reg_cpp.c_str());
        unlink(temp_host_o.c_str());
        if (has_dev_o) unlink(temp_dev_o.c_str());
        return ret;
    }

    bool has_reg_o = false;
    if (file_exists_and_not_empty(temp_reg_cpp)) {
        std::vector<std::string> reg_cmd = {
            host_cxx, "-O3", "-std=c++17", "-c", temp_reg_cpp, "-o", temp_reg_o
        };
        ret = run_command(reg_cmd, verbose);
        if (ret != 0) {
            unlink(temp_reg_cpp.c_str());
            unlink(temp_host_o.c_str());
            if (has_dev_o) unlink(temp_dev_o.c_str());
            unlink(temp_reg_o.c_str());
            return ret;
        }
        has_reg_o = true;
    }
    unlink(temp_reg_cpp.c_str());

    if (has_dev_o || has_reg_o) {
        std::vector<std::string> ld_cmd = { "ld", "-r", temp_host_o };
        if (has_dev_o) ld_cmd.push_back(temp_dev_o);
        if (has_reg_o) ld_cmd.push_back(temp_reg_o);
        ld_cmd.push_back("-o");
        ld_cmd.push_back(out_obj);
        ret = run_command(ld_cmd, verbose);
        unlink(temp_host_o.c_str());
        if (has_dev_o) unlink(temp_dev_o.c_str());
        if (has_reg_o) unlink(temp_reg_o.c_str());
        return ret;
    } else {
        if (rename(temp_host_o.c_str(), out_obj.c_str()) != 0) {
            std::vector<std::string> cp_cmd = { "cp", temp_host_o, out_obj };
            run_command(cp_cmd, verbose);
            unlink(temp_host_o.c_str());
        }
        return 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "nvcc fatal   : No input files specified; use option --help for more information\n";
        return 1;
    }

    std::string toolkit_root = get_toolkit_root();
    std::string bin_dir = toolkit_root + "/bin";
    std::string inc_dir = toolkit_root + "/include";
    std::string lib_dir = toolkit_root + "/lib";
    std::string libdevice_dir = toolkit_root + "/nvvm/libdevice";

    bool verbose = false;
    bool compile_only = false;
    bool preprocess_only = false;
    bool dep_only = false;
    std::string explicit_ccbin;
    std::string output_file;
    std::vector<std::string> input_files;
    std::vector<std::string> forward_flags;
    std::vector<std::string> link_flags;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-V" || arg == "--version") {
            print_version();
            return 0;
        }
        if (arg == "-h" || arg == "--help") {
            print_help();
            return 0;
        }
        if (arg == "__cmake_determine_cuda") {
            std::cout << "#$ TOP=" << toolkit_root << "\n"
                      << "#$ NVVMIR_LIBRARY_DIR=" << libdevice_dir << "\n"
                      << "#$ PATH=" << bin_dir << "\n"
                      << "#$ INCLUDES= -I" << inc_dir << "\n"
                      << "#$ LIBRARIES= -L" << lib_dir << " -lcudart\n"
                      << "#$ -arch compute_80\n"
                      << "#$ " << find_host_cxx("") << " -o a.out -L" << lib_dir << " -lcudart\n";
            return 0;
        }
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
            continue;
        }
        if (arg == "-c") {
            compile_only = true;
            continue;
        }
        if (arg == "-E") {
            preprocess_only = true;
            forward_flags.push_back(arg);
            continue;
        }
        if (arg == "-M" || arg == "-MM") {
            dep_only = true;
            forward_flags.push_back(arg);
            continue;
        }
        if (arg == "-o") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
            continue;
        }
        if (arg.rfind("-o=", 0) == 0) {
            output_file = arg.substr(3);
            continue;
        }
        if (arg == "-ccbin") {
            if (i + 1 < argc) {
                explicit_ccbin = argv[++i];
            }
            continue;
        }
        if (arg.rfind("-ccbin=", 0) == 0) {
            explicit_ccbin = arg.substr(7);
            continue;
        }
        // Multi-token flags that take an argument
        if (arg == "-MT" || arg == "-MF" || arg == "-MQ" || arg == "-isystem" ||
            arg == "-I" || arg == "-D" || arg == "-U" || arg == "-target" || arg == "--target") {
            forward_flags.push_back(arg);
            if (i + 1 < argc) {
                forward_flags.push_back(argv[++i]);
            }
            continue;
        }
        if (arg == "-x") {
            if (i + 1 < argc) {
                std::string next = argv[++i];
                if (next != "cu") {
                    forward_flags.push_back("-x");
                    forward_flags.push_back(next);
                }
            }
            continue;
        }
        if (arg == "-Xcompiler" || arg == "-Xlinker") {
            if (i + 1 < argc) {
                std::string flags = argv[++i];
                std::stringstream ss(flags);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    if (!item.empty()) {
                        if (arg == "-Xlinker") link_flags.push_back(item);
                        else forward_flags.push_back(item);
                    }
                }
            }
            continue;
        }
        if (arg.rfind("-Xcompiler=", 0) == 0) {
            std::string flags = arg.substr(11);
            std::stringstream ss(flags);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) forward_flags.push_back(item);
            }
            continue;
        }
        if (arg.rfind("-Xlinker=", 0) == 0) {
            std::string flags = arg.substr(9);
            std::stringstream ss(flags);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) link_flags.push_back(item);
            }
            continue;
        }
        // Ignored nvcc specific options with possible arguments
        if (arg == "--keep-dir" || arg == "-arch" || arg == "-gencode" ||
            arg == "--generate-code" || arg == "--default-stream" ||
            arg == "--diag-suppress") {
            if (i + 1 < argc) ++i;
            continue;
        }
        if (arg.rfind("--keep-dir=", 0) == 0 || arg.rfind("-arch=", 0) == 0 ||
            arg.rfind("--cuda-gpu-arch", 0) == 0 || arg.rfind("-gencode=", 0) == 0 ||
            arg.rfind("--generate-code=", 0) == 0 || arg.rfind("--default-stream=", 0) == 0 ||
            arg.rfind("--diag-suppress=", 0) == 0) {
            continue;
        }
        // Ignored nvcc boolean flags
        if (arg == "--keep" || arg == "-rdc=true" || arg == "-rdc=false" ||
            arg == "-dlto" || arg == "-forward-unknown-to-host-compiler" ||
            arg == "--expt-relaxed-constexpr" || arg == "--expt-extended-lambda" ||
            arg == "--extended-lambda") {
            continue;
        }
        // General flags to forward
        if (arg[0] == '-') {
            forward_flags.push_back(arg);
        } else {
            input_files.push_back(arg);
        }
    }

    std::string host_cxx = find_host_cxx(explicit_ccbin);

    if (verbose) {
        std::cerr << "#$ TOP=" << toolkit_root << "\n"
                  << "#$ NVVMIR_LIBRARY_DIR=" << libdevice_dir << "\n"
                  << "#$ PATH=" << bin_dir << "\n"
                  << "#$ INCLUDES= -I" << inc_dir << "\n"
                  << "#$ LIBRARIES= -L" << lib_dir << " -lcudart\n"
                  << "#$ -arch compute_80\n";
    }

    if (input_files.empty()) {
        std::cerr << "nvcc fatal   : No input files specified; use option --help for more information\n";
        return 1;
    }

    bool has_cuda_input = false;
    for (const auto& file : input_files) {
        size_t dot = file.rfind('.');
        if (dot != std::string::npos) {
            std::string ext = file.substr(dot);
            if (ext == ".cu" || ext == ".cuh") {
                has_cuda_input = true;
                break;
            }
        }
    }

    bool is_linking = !compile_only && !preprocess_only && !dep_only;

    if (is_linking) {
        // Direct link execution
        if (output_file.empty()) {
            output_file = "a.out";
        }

        std::vector<std::string> temp_objs;
        std::vector<std::string> link_inputs;

        for (const auto& file : input_files) {
            size_t dot = file.rfind('.');
            std::string ext = (dot != std::string::npos) ? file.substr(dot) : "";
            if (ext == ".cu") {
                std::string temp_o = output_file + "_" + std::to_string(temp_objs.size()) + ".o";
                int ret = compile_cuda_to_object(host_cxx, file, temp_o, bin_dir, inc_dir, forward_flags, verbose);
                if (ret != 0) {
                    for (const auto& to : temp_objs) unlink(to.c_str());
                    unlink(temp_o.c_str());
                    return ret;
                }
                temp_objs.push_back(temp_o);
                link_inputs.push_back(temp_o);
            } else {
                link_inputs.push_back(file);
            }
        }

        std::vector<std::string> link_cmd = { host_cxx };
        for (const auto& inp : link_inputs) link_cmd.push_back(inp);
        link_cmd.push_back("-o");
        link_cmd.push_back(output_file);
        for (const auto& f : forward_flags) link_cmd.push_back(f);
        for (const auto& lf : link_flags) link_cmd.push_back(lf);

        link_cmd.push_back("-L" + lib_dir);
        link_cmd.push_back("-lcudart");
        link_cmd.push_back("-framework");
        link_cmd.push_back("Metal");
        link_cmd.push_back("-framework");
        link_cmd.push_back("Foundation");
        link_cmd.push_back("-Wl,-rpath," + lib_dir);

        int ret = run_command(link_cmd, verbose);
        for (const auto& to : temp_objs) {
            unlink(to.c_str());
        }
        return ret;
    } else if (compile_only) {
        for (const auto& file : input_files) {
            size_t dot = file.rfind('.');
            std::string ext = (dot != std::string::npos) ? file.substr(dot) : "";
            std::string cur_out = output_file;
            if (cur_out.empty()) {
                size_t slash = file.rfind('/');
                std::string base = (slash != std::string::npos) ? file.substr(slash + 1) : file;
                size_t bdot = base.rfind('.');
                cur_out = (bdot != std::string::npos ? base.substr(0, bdot) : base) + ".o";
            }
            if (ext == ".cu") {
                int ret = compile_cuda_to_object(host_cxx, file, cur_out, bin_dir, inc_dir, forward_flags, verbose);
                if (ret != 0) return ret;
            } else {
                std::vector<std::string> c_cmd = { host_cxx, "-c", file, "-o", cur_out };
                for (const auto& f : forward_flags) c_cmd.push_back(f);
                int ret = run_command(c_cmd, verbose);
                if (ret != 0) return ret;
            }
        }
        return 0;
    } else {
        // Preprocess only / dep generation
        std::vector<std::string> comp_cmd = { host_cxx };
        if (has_cuda_input) {
            comp_cmd.push_back("--cuda-host-only");
            comp_cmd.push_back("-nocudainc");
            comp_cmd.push_back("-nocudalib");
            comp_cmd.push_back("-x");
            comp_cmd.push_back("cuda");
            comp_cmd.push_back("-include");
            comp_cmd.push_back("cuda_runtime.h");
            comp_cmd.push_back("-D__CUDACC__=1");
            comp_cmd.push_back("-D__NVCC__=1");
            comp_cmd.push_back("-D__CUDACC_VER_MAJOR__=12");
            comp_cmd.push_back("-D__CUDACC_VER_MINOR__=0");
            comp_cmd.push_back("-D__CUDACC_VER_BUILD__=0");
            comp_cmd.push_back("-DCUDART_VERSION=12000");
            comp_cmd.push_back("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1");
            comp_cmd.push_back("-DCCCL_DISABLE_NVTX=1");
            comp_cmd.push_back("-DNVTX_DISABLE=1");
            comp_cmd.push_back("-I" + inc_dir);
        }
        if (!output_file.empty()) {
            comp_cmd.push_back("-o");
            comp_cmd.push_back(output_file);
        }
        for (const auto& f : forward_flags) comp_cmd.push_back(f);
        for (const auto& inp : input_files) comp_cmd.push_back(inp);

        return run_command(comp_cmd, verbose);
    }
}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {
namespace fs = std::filesystem;

/**
 * @brief Quote one shell argument for POSIX-compatible command execution.
 * @param value Raw argument text.
 * @return Escaped argument that can be safely appended to a shell command.
 */
std::string shell_quote(const std::string_view value) {
  std::string out;
  out.reserve(value.size() + 2U);
  out.push_back('\'');
  for (const char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out.push_back(ch);
    }
  }
  out.push_back('\'');
  return out;
}

/**
 * @brief Execute a command from an argument vector.
 * @param args Command and arguments where `args[0]` is the executable.
 * @return Exit code returned by `std::system`.
 */
int run_command(const std::vector<std::string> &args) {
  if (args.empty()) {
    throw std::runtime_error("run_command requires a non-empty argument list");
  }

  std::string command;
  for (const std::string &arg : args) {
    if (!command.empty()) {
      command.push_back(' ');
    }
    command += shell_quote(arg);
  }
  return std::system(command.c_str());
}

/**
 * @brief Create a unique temporary directory for one test case.
 * @param label Prefix label included in the directory name.
 * @return Created unique temporary directory path.
 */
fs::path make_temp_dir(const std::string_view label) {
  const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 rng(static_cast<std::uint64_t>(now));
  std::error_code ec;

  for (std::size_t attempt = 0; attempt < 32U; ++attempt) {
    const fs::path candidate = fs::temp_directory_path() / (std::string("varre_shader_tests_") + std::string(label) + "_" + std::to_string(rng()));
    if (fs::create_directories(candidate, ec)) {
      return candidate;
    }
    if (ec) {
      throw std::runtime_error("failed creating temporary directory: " + ec.message());
    }
  }

  throw std::runtime_error("failed to create a unique temporary directory");
}

/**
 * @brief Write a fake `slangc` wrapper that copies prebuilt SPIR-V fixtures.
 * @param script_path Output script path.
 */
void write_fake_slangc_script(const fs::path &script_path) {
  const std::string script = std::string("#!/usr/bin/env bash\n") +
                             "set -euo pipefail\n"
                             "src=\"\"\n"
                             "out=\"\"\n"
                             "prev=\"\"\n"
                             "for arg in \"$@\"; do\n"
                             "  if [[ -z \"$src\" && \"$arg\" != -* ]]; then\n"
                             "    src=\"$arg\"\n"
                             "  fi\n"
                             "  if [[ \"$prev\" == \"-o\" ]]; then\n"
                             "    out=\"$arg\"\n"
                             "    break\n"
                             "  fi\n"
                             "  prev=\"$arg\"\n"
                             "done\n"
                             "if [[ -z \"$src\" || -z \"$out\" ]]; then\n"
                             "  echo \"fake slangc: missing source or -o argument\" >&2\n"
                             "  exit 2\n"
                             "fi\n"
                             "mkdir -p \"$(dirname \"$out\")\"\n"
                             "base=\"$(basename \"$src\")\"\n"
                             "fixtures=" +
                             shell_quote(VARRE_SHADERS_FIXTURES_DIR) +
                             "\n"
                             "case \"$base\" in\n"
                             "  with_bindings.slang)\n"
                             "    cp \"$fixtures/with_bindings.spv\" \"$out\"\n"
                             "    ;;\n"
                             "  no_bindings.slang)\n"
                             "    cp \"$fixtures/no_bindings.spv\" \"$out\"\n"
                             "    ;;\n"
                             "  *)\n"
                             "    cp \"$fixtures/invalid.spv\" \"$out\"\n"
                             "    ;;\n"
                             "esac\n";

  std::ofstream out(script_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("failed writing fake slangc script: " + script_path.string());
  }
  out << script;
  out.close();

#ifndef _WIN32
  std::error_code ec;
  fs::permissions(script_path,
                  fs::perms::owner_read | fs::perms::owner_write | fs::perms::owner_exec | fs::perms::group_read | fs::perms::group_exec |
                    fs::perms::others_read | fs::perms::others_exec,
                  fs::perm_options::replace, ec);
  if (ec) {
    throw std::runtime_error("failed setting executable permission on fake slangc script: " + ec.message());
  }
#endif
}

/**
 * @brief Invoke shader code generation for one corpus directory.
 * @param input_root Shader source corpus root.
 * @param output_root Output directory for generated files.
 * @param fake_slangc Path to fake slangc wrapper script.
 * @param ns Namespace used for generated output.
 * @return Exit status code from shader_codegen.
 */
int run_shader_codegen(const fs::path &input_root, const fs::path &output_root, const fs::path &fake_slangc, const std::string &ns) {
  const fs::path emit_hpp = output_root / "shaders_generated.hpp";
  const fs::path emit_cpp = output_root / "shaders_generated.cpp";
  std::error_code ec;
  fs::create_directories(output_root, ec);
  if (ec) {
    throw std::runtime_error("failed to create codegen output directory: " + ec.message());
  }

  return run_command({
    VARRE_SHADER_CODEGEN_EXE,
    "--input-root",
    input_root.string(),
    "--output-dir",
    output_root.string(),
    "--emit-hpp",
    emit_hpp.string(),
    "--emit-cpp",
    emit_cpp.string(),
    "--namespace",
    ns,
    "--slangc",
    fake_slangc.string(),
  });
}

/**
 * @brief Read an entire text file into a string.
 * @param path Input file path.
 * @return Full text file content.
 */
std::string read_text_file(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open text file: " + path.string());
  }
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
} // namespace

TEST_CASE("shader codegen reflects descriptor bindings from SPIR-V", "[shaders][codegen][reflection]") {
  const fs::path valid_dir = fs::path(VARRE_SHADERS_CORPUS_VALID_DIR);
  REQUIRE(fs::exists(valid_dir));

  const fs::path temp_root = make_temp_dir("reflection");
  const fs::path output_dir = temp_root / "out";
  const fs::path fake_slangc = temp_root / "fake-slangc";
  write_fake_slangc_script(fake_slangc);

  const int exit_code = run_shader_codegen(valid_dir, output_dir, fake_slangc, "varre::assets::tests::shaders_valid");
  REQUIRE(exit_code == 0);

  const fs::path generated_cpp_path = output_dir / "shaders_generated.cpp";
  REQUIRE(fs::exists(generated_cpp_path));
  const std::string generated_cpp = read_text_file(generated_cpp_path);

  REQUIRE(generated_cpp.find("kShaderBindings_WITH_BINDINGS_COMPUTE_MAIN") != std::string::npos);
  REQUIRE(generated_cpp.find("kShaderBindings_NO_BINDINGS_COMPUTE_MAIN") != std::string::npos);
  REQUIRE(generated_cpp.find("{0U, 1U, 6U, 1U, 32U},") != std::string::npos);
  REQUIRE(generated_cpp.find("{2U, 3U, 1U, 4U, 32U},") != std::string::npos);
  REQUIRE(generated_cpp.find("std::array<DescriptorSetLayoutBinding, 0> kShaderBindings_NO_BINDINGS_COMPUTE_MAIN") != std::string::npos);

  const std::size_t set0_pos = generated_cpp.find("{0U, 1U, 6U, 1U, 32U},");
  const std::size_t set2_pos = generated_cpp.find("{2U, 3U, 1U, 4U, 32U},");
  REQUIRE(set0_pos != std::string::npos);
  REQUIRE(set2_pos != std::string::npos);
  REQUIRE(set0_pos < set2_pos);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("shader codegen fails on invalid SPIR-V payloads", "[shaders][codegen][reflection]") {
  const fs::path invalid_dir = fs::path(VARRE_SHADERS_CORPUS_INVALID_DIR);
  REQUIRE(fs::exists(invalid_dir));

  const fs::path temp_root = make_temp_dir("invalid_spirv");
  const fs::path output_dir = temp_root / "out";
  const fs::path fake_slangc = temp_root / "fake-slangc";
  write_fake_slangc_script(fake_slangc);

  const int exit_code = run_shader_codegen(invalid_dir, output_dir, fake_slangc, "varre::assets::tests::shaders_invalid");
  REQUIRE(exit_code != 0);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

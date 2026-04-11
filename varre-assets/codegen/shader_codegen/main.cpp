/**
 * @file main.cpp
 * @brief Build-time shader asset code generator for Slang sources.
 *
 * This tool discovers `.slang` source files, extracts shader entry points,
 * compiles each entry point to SPIR-V using `slangc`, validates the generated
 * binary with SPIRV-Tools, and emits `shaders_generated.hpp/.cpp`.
 */
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <spirv-tools/libspirv.hpp>
#include <spirv_reflect.h>

#ifndef VARRE_SLANGC_EXECUTABLE
#define VARRE_SLANGC_EXECUTABLE "slangc"
#endif

namespace fs = std::filesystem;

namespace {

/**
 * @brief Vulkan shader-stage bit values used in generated metadata.
 */
constexpr std::uint32_t kVkShaderStageVertexBit = 0x00000001U;
constexpr std::uint32_t kVkShaderStageTessellationControlBit = 0x00000002U;
constexpr std::uint32_t kVkShaderStageTessellationEvaluationBit = 0x00000004U;
constexpr std::uint32_t kVkShaderStageGeometryBit = 0x00000008U;
constexpr std::uint32_t kVkShaderStageFragmentBit = 0x00000010U;
constexpr std::uint32_t kVkShaderStageComputeBit = 0x00000020U;
constexpr std::uint32_t kVkShaderStageTaskBitExt = 0x00000040U;
constexpr std::uint32_t kVkShaderStageMeshBitExt = 0x00000080U;
constexpr std::uint32_t kVkShaderStageRaygenBitKhr = 0x00000100U;

/**
 * @brief Command-line options for shader code generation.
 */
struct CliOptions {
  fs::path input_root;
  fs::path output_dir;
  fs::path emit_hpp;
  fs::path emit_cpp;
  std::string ns = "varre::assets";
  std::string slangc_path = VARRE_SLANGC_EXECUTABLE;
};

/**
 * @brief One discovered shader entry-point declaration from source.
 */
struct EntryPointDecl {
  std::string stage_name;
  std::string function_name;
};

/**
 * @brief Static stage metadata derived from Slang stage names.
 */
struct StageInfo {
  std::string_view enum_variant;
  std::uint32_t vk_stage_flags = 0U;
};

/**
 * @brief One reflected descriptor-set layout binding.
 */
struct DescriptorBinding {
  std::uint32_t set = 0U;
  std::uint32_t binding = 0U;
  std::uint32_t descriptor_type = 0U;
  std::uint32_t descriptor_count = 0U;
  std::uint32_t stage_flags = 0U;
};

/**
 * @brief Output record for one generated shader asset.
 */
struct ShaderBlob {
  std::string enum_name;
  std::string display_name;
  std::string source_path;
  std::string entry_point;
  std::string stage_variant;
  std::uint32_t stage_flags = 0U;
  std::vector<DescriptorBinding> descriptor_bindings;
  std::vector<std::uint8_t> bytes;
};

/**
 * @brief Throw a fatal error with a single message.
 * @param message Human-readable diagnostic.
 */
[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }

/**
 * @brief Quote a single command argument for POSIX shell execution.
 * @param value Raw argument text.
 * @return Escaped argument suitable for shell command construction.
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
 * @brief Execute an external command via `std::system`.
 * @param args Argument vector where `args[0]` is the executable.
 * @return Process exit status code from `std::system`.
 */
int run_command(const std::vector<std::string> &args) {
  if (args.empty()) {
    fail("run_command requires at least one argument");
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
 * @brief Normalize text into a deterministic C++ identifier fragment.
 * @param value Raw string value.
 * @return Identifier-safe uppercase string.
 */
std::string sanitize_identifier(const std::string &value) {
  std::string out;
  out.reserve(value.size() + 8U);
  for (const char ch : value) {
    if (std::isalnum(static_cast<unsigned char>(ch)) != 0) {
      out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
    } else {
      out.push_back('_');
    }
  }

  std::string compact;
  compact.reserve(out.size());
  bool prev_is_underscore = false;
  for (const char ch : out) {
    if (ch == '_') {
      if (!prev_is_underscore) {
        compact.push_back(ch);
      }
      prev_is_underscore = true;
    } else {
      compact.push_back(ch);
      prev_is_underscore = false;
    }
  }

  while (!compact.empty() && compact.front() == '_') {
    compact.erase(compact.begin());
  }
  while (!compact.empty() && compact.back() == '_') {
    compact.pop_back();
  }
  if (compact.empty()) {
    compact = "SHADER";
  }
  if (std::isdigit(static_cast<unsigned char>(compact.front())) != 0) {
    compact.insert(compact.begin(), '_');
  }
  return compact;
}

/**
 * @brief Discover `.slang` files under the input root.
 * @param input_root Root directory to scan recursively.
 * @return Deterministically sorted source file list.
 */
std::vector<fs::path> discover_slang_files(const fs::path &input_root) {
  std::vector<fs::path> files;
  if (!fs::exists(input_root)) {
    return files;
  }

  for (const fs::directory_entry &entry : fs::recursive_directory_iterator(input_root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path path = entry.path();
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](const unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (ext == ".slang") {
      files.push_back(path);
    }
  }

  std::sort(files.begin(), files.end(), [](const fs::path &a, const fs::path &b) { return a.generic_string() < b.generic_string(); });
  return files;
}

/**
 * @brief Read an entire text file.
 * @param path Source file path.
 * @return UTF-8/ASCII text bytes as a string.
 */
std::string read_text_file(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    fail("failed to open text file: " + path.generic_string());
  }
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/**
 * @brief Read an entire binary file.
 * @param path Source file path.
 * @return Raw byte payload.
 */
std::vector<std::uint8_t> read_binary_file(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    fail("failed to open binary file: " + path.generic_string());
  }
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/**
 * @brief Parse Slang entry-point annotations from a shader source file.
 * @param source_path Source file path (for diagnostics).
 * @param source_text Source file text.
 * @return Ordered list of discovered stage/function pairs.
 */
std::vector<EntryPointDecl> parse_entry_points(const fs::path &source_path, const std::string &source_text) {
  const std::regex regex(R"REGEX(\[shader\("([^"]+)"\)\][^\(]*\s([A-Za-z_][A-Za-z0-9_]*)\s*\()REGEX");
  std::vector<EntryPointDecl> entry_points;

  for (std::sregex_iterator it(source_text.begin(), source_text.end(), regex), end; it != end; ++it) {
    EntryPointDecl decl{
      .stage_name = (*it)[1].str(),
      .function_name = (*it)[2].str(),
    };
    entry_points.push_back(std::move(decl));
  }

  if (entry_points.empty()) {
    std::cerr << "shader_codegen: warning: no [shader(\"stage\")] entry points found in " << source_path.generic_string() << '\n';
  }
  return entry_points;
}

/**
 * @brief Map Slang stage names to generated API enum variants and Vulkan flags.
 * @param stage_name Slang stage token from `[shader("...")]`.
 * @return Stage mapping metadata.
 */
StageInfo stage_info_from_slang(const std::string &stage_name) {
  if (stage_name == "vertex") {
    return StageInfo{.enum_variant = "kVertex", .vk_stage_flags = kVkShaderStageVertexBit};
  }
  if (stage_name == "fragment") {
    return StageInfo{.enum_variant = "kFragment", .vk_stage_flags = kVkShaderStageFragmentBit};
  }
  if (stage_name == "compute") {
    return StageInfo{.enum_variant = "kCompute", .vk_stage_flags = kVkShaderStageComputeBit};
  }
  if (stage_name == "geometry") {
    return StageInfo{.enum_variant = "kGeometry", .vk_stage_flags = kVkShaderStageGeometryBit};
  }
  if (stage_name == "tesscontrol") {
    return StageInfo{.enum_variant = "kTessellationControl", .vk_stage_flags = kVkShaderStageTessellationControlBit};
  }
  if (stage_name == "tesseval") {
    return StageInfo{.enum_variant = "kTessellationEvaluation", .vk_stage_flags = kVkShaderStageTessellationEvaluationBit};
  }
  if (stage_name == "task") {
    return StageInfo{.enum_variant = "kTask", .vk_stage_flags = kVkShaderStageTaskBitExt};
  }
  if (stage_name == "mesh") {
    return StageInfo{.enum_variant = "kMesh", .vk_stage_flags = kVkShaderStageMeshBitExt};
  }
  if (stage_name == "raygen") {
    return StageInfo{.enum_variant = "kRaygen", .vk_stage_flags = kVkShaderStageRaygenBitKhr};
  }

  fail("unsupported slang shader stage: " + stage_name);
}

/**
 * @brief Compile one Slang source entry point to SPIR-V.
 * @param slangc_executable Path to slangc executable.
 * @param source_path Slang source file path.
 * @param entry_point Function name used as compilation entry.
 * @param output_spv_path Destination SPIR-V file path.
 */
void compile_shader_entry(const std::string &slangc_executable, const fs::path &source_path, const std::string &entry_point, const fs::path &output_spv_path) {
  std::error_code ec;
  fs::create_directories(output_spv_path.parent_path(), ec);
  if (ec) {
    fail("failed to create shader output directory: " + output_spv_path.parent_path().generic_string() + " (" + ec.message() + ")");
  }

  const int exit_code = run_command({
    slangc_executable,
    source_path.generic_string(),
    "-target",
    "spirv",
    "-fvk-use-entrypoint-name",
    "-entry",
    entry_point,
    "-o",
    output_spv_path.generic_string(),
  });

  if (exit_code != 0) {
    std::ostringstream oss;
    oss << "slangc failed for source '" << source_path.generic_string() << "', entry '" << entry_point << "' (exit " << exit_code << ")";
    fail(oss.str());
  }
}

/**
 * @brief Validate a SPIR-V binary payload using SPIRV-Tools.
 * @param bytes Raw SPIR-V bytes.
 * @param context Human-readable context for diagnostics.
 */
void validate_spirv_binary(const std::vector<std::uint8_t> &bytes, const std::string &context) {
  if (bytes.empty()) {
    fail(context + ": produced empty SPIR-V payload");
  }
  if (bytes.size() % sizeof(std::uint32_t) != 0U) {
    fail(context + ": SPIR-V payload byte size is not word aligned");
  }

  std::vector<std::uint32_t> words(bytes.size() / sizeof(std::uint32_t));
  for (std::size_t i = 0; i < words.size(); ++i) {
    const std::size_t p = i * 4U;
    words[i] = static_cast<std::uint32_t>(bytes[p + 0U]) | (static_cast<std::uint32_t>(bytes[p + 1U]) << 8U) |
               (static_cast<std::uint32_t>(bytes[p + 2U]) << 16U) | (static_cast<std::uint32_t>(bytes[p + 3U]) << 24U);
  }

  std::string diagnostics;
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
  tools.SetMessageConsumer([&](spv_message_level_t /*level*/, const char * /*source*/, const spv_position_t &position, const char *message) {
    diagnostics += "line ";
    diagnostics += std::to_string(position.index);
    diagnostics += ": ";
    diagnostics += (message == nullptr) ? "<no message>" : message;
    diagnostics += '\n';
  });

  spvtools::ValidatorOptions options;
  const bool ok = tools.Validate(words.data(), words.size(), options);
  if (!ok) {
    if (diagnostics.empty()) {
      diagnostics = "SPIRV-Tools validation failed with no diagnostic message.";
    }
    fail(context + ": " + diagnostics);
  }
}

/**
 * @brief Convert SPIRV-Reflect descriptor types to Vulkan descriptor type values.
 * @param descriptor_type Descriptor type from SPIRV-Reflect.
 * @return Numeric Vulkan descriptor-type value.
 */
std::uint32_t reflect_descriptor_type_to_vk(const SpvReflectDescriptorType descriptor_type) { return static_cast<std::uint32_t>(descriptor_type); }

/**
 * @brief Reflect descriptor-set bindings from validated SPIR-V.
 * @param bytes SPIR-V bytes.
 * @param stage_flags Vulkan shader stage flags for this module.
 * @param context Human-readable context for diagnostics.
 * @return Sorted reflected descriptor-binding metadata.
 */
std::vector<DescriptorBinding> reflect_descriptor_bindings(const std::vector<std::uint8_t> &bytes, const std::uint32_t stage_flags,
                                                           const std::string &context) {
  struct ReflectModuleGuard {
    SpvReflectShaderModule module{};
    bool initialized = false;
    ~ReflectModuleGuard() {
      if (initialized) {
        spvReflectDestroyShaderModule(&module);
      }
    }
  } module_guard;

  const SpvReflectResult create_result = spvReflectCreateShaderModule(bytes.size(), bytes.data(), &module_guard.module);
  if (create_result != SPV_REFLECT_RESULT_SUCCESS) {
    fail(context + ": SPIRV-Reflect failed to create shader module");
  }
  module_guard.initialized = true;

  std::uint32_t binding_count = 0U;
  const SpvReflectResult count_result = spvReflectEnumerateDescriptorBindings(&module_guard.module, &binding_count, nullptr);
  if (count_result != SPV_REFLECT_RESULT_SUCCESS) {
    fail(context + ": SPIRV-Reflect failed to enumerate descriptor binding count");
  }

  std::vector<SpvReflectDescriptorBinding *> reflected_bindings(binding_count, nullptr);
  if (binding_count > 0U) {
    const SpvReflectResult bindings_result = spvReflectEnumerateDescriptorBindings(&module_guard.module, &binding_count, reflected_bindings.data());
    if (bindings_result != SPV_REFLECT_RESULT_SUCCESS) {
      fail(context + ": SPIRV-Reflect failed while enumerating descriptor bindings");
    }
  }

  std::vector<DescriptorBinding> bindings;
  bindings.reserve(binding_count);
  for (const SpvReflectDescriptorBinding *binding_ptr : reflected_bindings) {
    if (binding_ptr == nullptr) {
      fail(context + ": SPIRV-Reflect returned a null descriptor binding");
    }

    std::uint32_t descriptor_count = binding_ptr->count;
    if (descriptor_count == 0U) {
      descriptor_count = 1U;
    }

    bindings.push_back(DescriptorBinding{
      .set = binding_ptr->set,
      .binding = binding_ptr->binding,
      .descriptor_type = reflect_descriptor_type_to_vk(binding_ptr->descriptor_type),
      .descriptor_count = descriptor_count,
      .stage_flags = stage_flags,
    });
  }

  std::sort(bindings.begin(), bindings.end(), [](const DescriptorBinding &a, const DescriptorBinding &b) {
    if (a.set != b.set) {
      return a.set < b.set;
    }
    return a.binding < b.binding;
  });

  for (std::size_t i = 1; i < bindings.size(); ++i) {
    if (bindings[i - 1].set == bindings[i].set && bindings[i - 1].binding == bindings[i].binding) {
      std::ostringstream oss;
      oss << context << ": duplicate descriptor binding reflected for set=" << bindings[i].set << ", binding=" << bindings[i].binding;
      fail(oss.str());
    }
  }

  return bindings;
}

/**
 * @brief Build a deterministic enum identifier base for one shader entry.
 * @param input_root Root source directory.
 * @param source_path Slang source path.
 * @param stage_name Slang stage token.
 * @param entry_point Slang entry point function name.
 * @return Identifier base before collision suffix handling.
 */
std::string make_shader_enum_name_base(const fs::path &input_root, const fs::path &source_path, const std::string &stage_name, const std::string &entry_point) {
  fs::path rel = source_path.lexically_relative(input_root);
  if (rel.empty()) {
    rel = source_path.filename();
  }
  rel.replace_extension();

  std::ostringstream oss;
  oss << rel.generic_string() << "_" << stage_name << "_" << entry_point;
  return sanitize_identifier(oss.str());
}

/**
 * @brief Format binary bytes as a C++ hexadecimal initializer body.
 * @param bytes Binary payload.
 * @return Multi-line initializer content.
 */
std::string make_byte_initializer(const std::vector<std::uint8_t> &bytes) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (i % 16U == 0U) {
      oss << "\n    ";
    }
    oss << "0x" << std::setw(2) << static_cast<unsigned>(bytes[i]) << ",";
    if (i % 16U != 15U) {
      oss << ' ';
    }
  }
  oss << '\n';
  return oss.str();
}

/**
 * @brief Escape a string for C++ literal emission.
 * @param value Raw string.
 * @return Escaped string content without quotes.
 */
std::string escape_cpp_string(const std::string &value) {
  std::string out;
  out.reserve(value.size() + 16U);
  for (const unsigned char ch : value) {
    switch (ch) {
    case '\\':
      out += "\\\\";
      break;
    case '\"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out.push_back(static_cast<char>(ch));
      break;
    }
  }
  return out;
}

/**
 * @brief Write output text only when file content differs.
 * @param output_path Destination path.
 * @param content Full file text.
 */
void write_if_changed(const fs::path &output_path, const std::string &content) {
  std::error_code ec;
  if (fs::exists(output_path, ec)) {
    std::ifstream in(output_path, std::ios::binary);
    const std::string existing((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (existing == content) {
      return;
    }
  }

  fs::create_directories(output_path.parent_path(), ec);
  if (ec) {
    fail("failed to create output directory: " + output_path.parent_path().generic_string() + " (" + ec.message() + ")");
  }

  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    fail("failed to open output file for write: " + output_path.generic_string());
  }
  out << content;
}

/**
 * @brief Emit generated public header for shader assets.
 * @param ns Target C++ namespace.
 * @param shaders Ordered generated shader records.
 * @return Header file text.
 */
std::string emit_header(const std::string &ns, const std::vector<ShaderBlob> &shaders) {
  std::ostringstream hpp;
  hpp << "/**\n";
  hpp << " * @file shaders_generated.hpp\n";
  hpp << " * @brief Generated shader asset API.\n";
  hpp << " *\n";
  hpp << " * This file is auto-generated by shader_codegen. Do not edit manually.\n";
  hpp << " */\n";
  hpp << "#pragma once\n\n";
  hpp << "#include <cstddef>\n";
  hpp << "#include <cstdint>\n\n";
  hpp << "namespace " << ns << " {\n\n";
  hpp << "/** @brief Logical shader stage for compiled shader modules. */\n";
  hpp << "enum class ShaderStage : std::uint8_t {\n";
  hpp << "  kUnspecified = 0,\n";
  hpp << "  kVertex,\n";
  hpp << "  kTessellationControl,\n";
  hpp << "  kTessellationEvaluation,\n";
  hpp << "  kGeometry,\n";
  hpp << "  kFragment,\n";
  hpp << "  kCompute,\n";
  hpp << "  kTask,\n";
  hpp << "  kMesh,\n";
  hpp << "  kRaygen,\n";
  hpp << "};\n\n";
  hpp << "/** @brief Stable shader identifiers generated from source and entry points. */\n";
  hpp << "enum class ShaderId : std::uint32_t {\n";
  if (shaders.empty()) {
    hpp << "  kUnimplemented = 0,\n";
  } else {
    for (std::size_t i = 0; i < shaders.size(); ++i) {
      hpp << "  " << shaders[i].enum_name << " = " << i << ",\n";
    }
  }
  hpp << "};\n\n";
  hpp << "/** @brief Lightweight descriptor-layout binding metadata for one shader. */\n";
  hpp << "struct DescriptorSetLayoutBinding {\n";
  hpp << "  std::uint32_t set = 0;\n";
  hpp << "  std::uint32_t binding = 0;\n";
  hpp << "  std::uint32_t descriptor_type = 0;\n";
  hpp << "  std::uint32_t descriptor_count = 0;\n";
  hpp << "  std::uint32_t stage_flags = 0;\n";
  hpp << "};\n\n";
  hpp << "/** @brief Immutable runtime view over one embedded shader module. */\n";
  hpp << "struct ShaderAssetView {\n";
  if (shaders.empty()) {
    hpp << "  ShaderId id = ShaderId::kUnimplemented;\n";
  } else {
    hpp << "  ShaderId id = ShaderId::" << shaders.front().enum_name << ";\n";
  }
  hpp << "  const std::byte* data = nullptr;\n";
  hpp << "  std::size_t size = 0;\n";
  hpp << "  ShaderStage stage = ShaderStage::kUnspecified;\n";
  hpp << "  const char* entry_point = nullptr;\n";
  hpp << "  const DescriptorSetLayoutBinding* descriptor_set_layout_bindings = nullptr;\n";
  hpp << "  std::size_t descriptor_set_layout_binding_count = 0;\n";
  hpp << "};\n\n";
  hpp << "/**\n";
  hpp << " * @brief Resolve generated shader metadata by identifier.\n";
  hpp << " * @param id Shader identifier.\n";
  hpp << " * @return Pointer to immutable metadata, or `nullptr` if unknown.\n";
  hpp << " */\n";
  hpp << "[[nodiscard]] const ShaderAssetView* get_shader(ShaderId id);\n";
  hpp << "/**\n";
  hpp << " * @brief Resolve display name for a shader identifier.\n";
  hpp << " * @param id Shader identifier.\n";
  hpp << " * @return Null-terminated display name, or `<unknown>` when missing.\n";
  hpp << " */\n";
  hpp << "[[nodiscard]] const char* shader_name(ShaderId id);\n";
  hpp << "/**\n";
  hpp << " * @brief Access all generated shader IDs in deterministic order.\n";
  hpp << " * @param out_count Optional output count.\n";
  hpp << " * @return Pointer to immutable contiguous array of IDs.\n";
  hpp << " */\n";
  hpp << "[[nodiscard]] const ShaderId* all_shader_ids(std::size_t* out_count);\n\n";
  hpp << "}  // namespace " << ns << "\n";
  return hpp.str();
}

/**
 * @brief Emit generated shader asset implementation text.
 * @param ns Target C++ namespace.
 * @param shaders Ordered generated shader records.
 * @return Source file text.
 */
std::string emit_cpp(const std::string &ns, const std::vector<ShaderBlob> &shaders) {
  std::ostringstream cpp;
  cpp << "/**\n";
  cpp << " * @file shaders_generated.cpp\n";
  cpp << " * @brief Generated shader asset implementation.\n";
  cpp << " *\n";
  cpp << " * This file is auto-generated by shader_codegen. Do not edit manually.\n";
  cpp << " */\n";
  cpp << "#include \"shaders_generated.hpp\"\n\n";
  cpp << "#include <array>\n";
  cpp << "#include <cstddef>\n";
  cpp << "#include <cstdint>\n\n";
  cpp << "namespace " << ns << " {\n\n";
  cpp << "namespace {\n\n";
  cpp << "struct ShaderRecord {\n";
  cpp << "  ShaderId id;\n";
  cpp << "  const char* name;\n";
  cpp << "  const char* source_path;\n";
  cpp << "  ShaderStage stage;\n";
  cpp << "  std::uint32_t stage_flags;\n";
  cpp << "  const char* entry_point;\n";
  cpp << "  const std::uint8_t* data;\n";
  cpp << "  std::size_t size;\n";
  cpp << "  const DescriptorSetLayoutBinding* bindings;\n";
  cpp << "  std::size_t binding_count;\n";
  cpp << "};\n\n";

  for (const ShaderBlob &shader : shaders) {
    cpp << "alignas(4) static constexpr std::uint8_t kShaderData_" << shader.enum_name << "[] = {";
    cpp << make_byte_initializer(shader.bytes);
    cpp << "};\n";
    cpp << "static constexpr std::array<DescriptorSetLayoutBinding, " << shader.descriptor_bindings.size() << "> kShaderBindings_" << shader.enum_name
        << " = {{";
    if (shader.descriptor_bindings.empty()) {
      cpp << "}};\n\n";
    } else {
      cpp << '\n';
      for (const DescriptorBinding &binding : shader.descriptor_bindings) {
        cpp << "    {" << binding.set << "U, " << binding.binding << "U, " << binding.descriptor_type << "U, " << binding.descriptor_count << "U, "
            << binding.stage_flags << "U},\n";
      }
      cpp << "}};\n\n";
    }
  }

  cpp << "static constexpr std::array<ShaderRecord, " << shaders.size() << "> kShaders = {{\n";
  for (const ShaderBlob &shader : shaders) {
    const std::string display_name_escaped = escape_cpp_string(shader.display_name);
    const std::string source_path_escaped = escape_cpp_string(shader.source_path);
    const std::string entry_point_escaped = escape_cpp_string(shader.entry_point);
    cpp << "    {ShaderId::" << shader.enum_name << ", \"" << display_name_escaped << "\", \"" << source_path_escaped << "\", "
        << "ShaderStage::" << shader.stage_variant << ", " << shader.stage_flags << "U, "
        << "\"" << entry_point_escaped << "\", kShaderData_" << shader.enum_name << ", sizeof(kShaderData_" << shader.enum_name << "), "
        << "kShaderBindings_" << shader.enum_name << ".data(), kShaderBindings_" << shader.enum_name << ".size()},\n";
  }
  cpp << "}};\n\n";

  cpp << "static constexpr std::array<ShaderId, " << shaders.size() << "> kShaderIds = {{\n";
  for (const ShaderBlob &shader : shaders) {
    cpp << "    ShaderId::" << shader.enum_name << ",\n";
  }
  cpp << "}};\n\n";

  cpp << "const ShaderRecord* find_record(const ShaderId id) {\n";
  cpp << "  for (const ShaderRecord& record : kShaders) {\n";
  cpp << "    if (record.id == id) {\n";
  cpp << "      return &record;\n";
  cpp << "    }\n";
  cpp << "  }\n";
  cpp << "  return nullptr;\n";
  cpp << "}\n\n";
  cpp << "}  // namespace\n\n";

  cpp << "const ShaderAssetView* get_shader(const ShaderId id) {\n";
  cpp << "  const ShaderRecord* record = find_record(id);\n";
  cpp << "  if (record == nullptr) {\n";
  cpp << "    return nullptr;\n";
  cpp << "  }\n";
  cpp << "  static thread_local ShaderAssetView view;\n";
  cpp << "  view.id = record->id;\n";
  cpp << "  view.data = reinterpret_cast<const std::byte*>(record->data);\n";
  cpp << "  view.size = record->size;\n";
  cpp << "  view.stage = record->stage;\n";
  cpp << "  view.entry_point = record->entry_point;\n";
  cpp << "  view.descriptor_set_layout_bindings = record->bindings;\n";
  cpp << "  view.descriptor_set_layout_binding_count = record->binding_count;\n";
  cpp << "  return &view;\n";
  cpp << "}\n\n";

  cpp << "const char* shader_name(const ShaderId id) {\n";
  cpp << "  const ShaderRecord* record = find_record(id);\n";
  cpp << "  return record == nullptr ? \"<unknown>\" : record->name;\n";
  cpp << "}\n\n";

  cpp << "const ShaderId* all_shader_ids(std::size_t* out_count) {\n";
  cpp << "  if (out_count != nullptr) {\n";
  cpp << "    *out_count = kShaderIds.size();\n";
  cpp << "  }\n";
  cpp << "  return kShaderIds.data();\n";
  cpp << "}\n\n";

  cpp << "}  // namespace " << ns << "\n";
  return cpp.str();
}

/**
 * @brief Parse CLI arguments into generator options.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Parsed generator options.
 */
CliOptions parse_cli(const int argc, char **argv) {
  CliOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto read_value = [&](const char *flag) -> std::string {
      if (i + 1 >= argc) {
        fail(std::string("missing value for ") + flag);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--input-root") {
      opts.input_root = read_value("--input-root");
    } else if (arg == "--output-dir") {
      opts.output_dir = read_value("--output-dir");
    } else if (arg == "--emit-hpp") {
      opts.emit_hpp = read_value("--emit-hpp");
    } else if (arg == "--emit-cpp") {
      opts.emit_cpp = read_value("--emit-cpp");
    } else if (arg == "--namespace") {
      opts.ns = read_value("--namespace");
    } else if (arg == "--slangc") {
      opts.slangc_path = read_value("--slangc");
    } else {
      fail("unknown argument: " + std::string(arg));
    }
  }

  if (opts.input_root.empty() || opts.output_dir.empty() || opts.emit_hpp.empty() || opts.emit_cpp.empty()) {
    fail("usage: shader_codegen --input-root <dir> --output-dir <dir> --emit-hpp <file> --emit-cpp <file> "
         "[--namespace <ns>] [--slangc <path>]");
  }
  return opts;
}

} // namespace

/**
 * @brief Program entry point.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return `0` on success and non-zero on failure.
 */
int main(const int argc, char **argv) {
  try {
    const CliOptions opts = parse_cli(argc, argv);
    std::error_code ec;
    fs::create_directories(opts.output_dir, ec);
    if (ec) {
      fail("failed to create output directory: " + opts.output_dir.generic_string() + " (" + ec.message() + ")");
    }

    const std::vector<fs::path> source_files = discover_slang_files(opts.input_root);
    if (source_files.empty()) {
      fail("no .slang files found under input root: " + opts.input_root.generic_string());
    }

    std::vector<ShaderBlob> shaders;
    std::unordered_map<std::string, std::size_t> enum_collisions;
    const fs::path compiled_dir = opts.output_dir / "compiled";
    fs::create_directories(compiled_dir, ec);
    if (ec) {
      fail("failed to create compiled output directory: " + compiled_dir.generic_string() + " (" + ec.message() + ")");
    }

    for (const fs::path &source_path : source_files) {
      const std::string source_text = read_text_file(source_path);
      const std::vector<EntryPointDecl> entry_points = parse_entry_points(source_path, source_text);
      for (const EntryPointDecl &entry : entry_points) {
        const StageInfo stage_info = stage_info_from_slang(entry.stage_name);

        std::string enum_name = make_shader_enum_name_base(opts.input_root, source_path, entry.stage_name, entry.function_name);
        std::size_t &collision_count = enum_collisions[enum_name];
        ++collision_count;
        if (collision_count > 1U) {
          enum_name += "_" + std::to_string(collision_count);
        }

        const fs::path output_spv = compiled_dir / (enum_name + ".spv");
        compile_shader_entry(opts.slangc_path, source_path, entry.function_name, output_spv);
        const std::vector<std::uint8_t> spv_bytes = read_binary_file(output_spv);

        std::ostringstream context;
        context << source_path.generic_string() << ":" << entry.function_name << " (" << entry.stage_name << ")";
        validate_spirv_binary(spv_bytes, context.str());
        const std::vector<DescriptorBinding> reflected_bindings = reflect_descriptor_bindings(spv_bytes, stage_info.vk_stage_flags, context.str());

        fs::path rel = source_path.lexically_relative(opts.input_root);
        if (rel.empty()) {
          rel = source_path.filename();
        }

        std::ostringstream display_name;
        display_name << rel.generic_string() << ":" << entry.function_name << "[" << entry.stage_name << "]";

        shaders.push_back(ShaderBlob{
          .enum_name = std::move(enum_name),
          .display_name = display_name.str(),
          .source_path = rel.generic_string(),
          .entry_point = entry.function_name,
          .stage_variant = std::string(stage_info.enum_variant),
          .stage_flags = stage_info.vk_stage_flags,
          .descriptor_bindings = std::move(reflected_bindings),
          .bytes = spv_bytes,
        });
      }
    }

    if (shaders.empty()) {
      fail("no shader entries were generated; ensure source files contain [shader(\"stage\")] entry points");
    }

    std::sort(shaders.begin(), shaders.end(), [](const ShaderBlob &a, const ShaderBlob &b) { return a.enum_name < b.enum_name; });

    write_if_changed(opts.emit_hpp, emit_header(opts.ns, shaders));
    write_if_changed(opts.emit_cpp, emit_cpp(opts.ns, shaders));

    std::cerr << "shader_codegen: generated " << shaders.size() << " shader asset(s) into " << opts.output_dir.generic_string() << '\n';
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "shader_codegen: error: " << ex.what() << '\n';
    return 1;
  }
}

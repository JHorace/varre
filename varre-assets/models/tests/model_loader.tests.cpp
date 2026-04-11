#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "varre/assets/models.hpp"

namespace {
namespace fs = std::filesystem;

/**
 * @brief Read a 32-bit little-endian integer from an arbitrary byte offset.
 * @param data Source byte buffer.
 * @param offset Byte offset inside @p data.
 * @return Decoded unsigned integer value.
 */
std::uint32_t read_u32_le(const std::byte *data, std::size_t offset) {
  const auto b0 = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[offset + 0]));
  const auto b1 = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[offset + 1]));
  const auto b2 = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[offset + 2]));
  const auto b3 = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[offset + 3]));
  return b0 | (b1 << 8U) | (b2 << 16U) | (b3 << 24U);
}

/**
 * @brief Quote a shell argument using single-quote escaping for POSIX shells.
 * @param value Raw argument text.
 * @return Escaped argument safe to concatenate into a shell command line.
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
 * @brief Execute a command by shelling out with escaped arguments.
 * @param args Command arguments where `args[0]` is the executable path.
 * @return Exit code returned by `std::system`.
 */
int run_command(const std::vector<std::string> &args) {
  if (args.empty()) {
    throw std::runtime_error("run_command requires at least one argument");
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
 * @brief Create a unique temporary directory for an individual test run.
 * @param label Human-readable name used as part of the directory prefix.
 * @return Filesystem path to the created directory.
 */
fs::path make_temp_dir(const std::string_view label) {
  const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 rng(static_cast<std::uint64_t>(now));
  std::error_code ec;

  for (std::size_t attempt = 0; attempt < 32U; ++attempt) {
    const std::string name = std::string("varre_") + std::string(label) + "_" + std::to_string(rng());
    const fs::path path = fs::temp_directory_path() / name;
    if (fs::create_directories(path, ec)) {
      return path;
    }
    if (ec) {
      throw std::runtime_error("failed creating temp test directory: " + ec.message());
    }
  }

  throw std::runtime_error("failed to allocate unique temp directory for tests");
}

/**
 * @brief Invoke `model_codegen` for a given corpus directory.
 * @param input_root Root directory containing OBJ inputs.
 * @param output_root Output directory where generated files are written.
 * @param ns Target C++ namespace passed to code generation.
 * @return Process exit code from the generator invocation.
 */
int run_model_codegen(const fs::path &input_root, const fs::path &output_root, const std::string &ns) {
  const fs::path emit_hpp = output_root / "models_generated.hpp";
  const fs::path emit_cpp = output_root / "models_generated.cpp";
  std::error_code ec;
  fs::create_directories(output_root, ec);
  if (ec) {
    throw std::runtime_error("failed to create codegen output directory: " + ec.message());
  }

  return run_command({
    VARRE_MODEL_CODEGEN_EXE,
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
  });
}
} // namespace

TEST_CASE("model loader enumerates embedded assets", "[models]") {
  std::size_t count = 0;
  const varre::assets::ModelId *ids = varre::assets::all_model_ids(&count);

  REQUIRE(ids != nullptr);
  REQUIRE(count >= 1U);
  REQUIRE(std::string(varre::assets::model_name(ids[0])) == "cube");
}

TEST_CASE("model loader returns non-empty data and decodes each model", "[models]") {
  std::size_t count = 0;
  const varre::assets::ModelId *ids = varre::assets::all_model_ids(&count);

  REQUIRE(ids != nullptr);
  REQUIRE(count >= 1U);

  for (std::size_t i = 0; i < count; ++i) {
    const varre::assets::ModelId id = ids[i];

    std::size_t byte_size = 0;
    const std::byte *data = varre::assets::get_model_data(id, &byte_size);
    REQUIRE(data != nullptr);
    REQUIRE(byte_size > 0U);

    const varre::assets::ModelAsset model = varre::assets::load_model(id);
    REQUIRE_FALSE(model.vertices.empty());
    REQUIRE_FALSE(model.indices.empty());
    REQUIRE(model.indices.size() % 3U == 0U);

    for (const std::uint32_t idx : model.indices) {
      REQUIRE(static_cast<std::size_t>(idx) < model.vertices.size());
    }
  }
}

TEST_CASE("model loader rejects unknown IDs", "[models]") {
  const auto invalid_id = static_cast<varre::assets::ModelId>(std::numeric_limits<std::uint32_t>::max());

  REQUIRE_THROWS_AS(varre::assets::get_model_data(invalid_id, nullptr), std::runtime_error);
  REQUIRE_THROWS_AS(varre::assets::load_model(invalid_id), std::runtime_error);
}

TEST_CASE("model binary header has magic/version/flags", "[models]") {
  constexpr std::uint32_t kExpectedMagic = 0x444D5256U; // "VRMD"
  constexpr std::uint32_t kExpectedVersion = 1U;

  std::size_t count = 0;
  const varre::assets::ModelId *ids = varre::assets::all_model_ids(&count);
  REQUIRE(ids != nullptr);
  REQUIRE(count >= 1U);

  for (std::size_t i = 0; i < count; ++i) {
    std::size_t byte_size = 0;
    const std::byte *data = varre::assets::get_model_data(ids[i], &byte_size);
    REQUIRE(data != nullptr);
    REQUIRE(byte_size >= 16U);

    const std::uint32_t magic = read_u32_le(data, 0U);
    const std::uint32_t version = read_u32_le(data, 4U);
    const std::uint32_t flags = read_u32_le(data, 8U);
    const std::uint32_t vertex_count = read_u32_le(data, 12U);

    REQUIRE(magic == kExpectedMagic);
    REQUIRE(version == kExpectedVersion);
    REQUIRE((flags & ~0x7U) == 0U);
    REQUIRE(vertex_count > 0U);
  }
}

TEST_CASE("model loader generates normals when OBJ lacks them", "[models]") {
  std::size_t count = 0;
  const varre::assets::ModelId *ids = varre::assets::all_model_ids(&count);
  REQUIRE(ids != nullptr);
  REQUIRE(count >= 1U);

  bool found_triangle = false;
  for (std::size_t i = 0; i < count; ++i) {
    const varre::assets::ModelId id = ids[i];
    if (std::string(varre::assets::model_name(id)) != "triangle_no_normals.obj") {
      continue;
    }

    found_triangle = true;
    const varre::assets::ModelAsset model = varre::assets::load_model(id);
    REQUIRE(model.vertices.size() == 3U);
    REQUIRE(model.indices.size() == 3U);

    for (const varre::assets::Vertex &v : model.vertices) {
      const float len = std::sqrt((v.nx * v.nx) + (v.ny * v.ny) + (v.nz * v.nz));
      REQUIRE(len == Catch::Approx(1.0F).margin(1e-4F));
      REQUIRE(v.nz == Catch::Approx(1.0F).margin(1e-4F));
    }
  }

  REQUIRE(found_triangle);
}

TEST_CASE("model decoder rejects truncated blobs", "[models]") {
  std::size_t count = 0;
  const varre::assets::ModelId *ids = varre::assets::all_model_ids(&count);
  REQUIRE(ids != nullptr);
  REQUIRE(count >= 1U);

  std::size_t byte_size = 0;
  const std::byte *data = varre::assets::get_model_data(ids[0], &byte_size);
  REQUIRE(data != nullptr);
  REQUIRE(byte_size > 24U);

  std::vector<std::size_t> truncation_sizes = {
    0U, 1U, 2U, 3U, 4U, 8U, 12U, 16U, byte_size - 1U,
  };
  std::sort(truncation_sizes.begin(), truncation_sizes.end());
  truncation_sizes.erase(std::unique(truncation_sizes.begin(), truncation_sizes.end()), truncation_sizes.end());

  for (const std::size_t truncated_size : truncation_sizes) {
    REQUIRE(truncated_size < byte_size);
    REQUIRE_THROWS_AS(varre::assets::decode_model_data(ids[0], data, truncated_size), std::runtime_error);
  }
}

TEST_CASE("model decoder rejects randomized malformed blobs", "[models]") {
  std::size_t count = 0;
  const varre::assets::ModelId *ids = varre::assets::all_model_ids(&count);
  REQUIRE(ids != nullptr);
  REQUIRE(count >= 1U);

  std::size_t byte_size = 0;
  const std::byte *data = varre::assets::get_model_data(ids[0], &byte_size);
  REQUIRE(data != nullptr);
  REQUIRE(byte_size > 0U);

  const std::vector<std::byte> original(data, data + byte_size);
  std::mt19937 rng(0x7A17C0DEU);
  std::uniform_int_distribution<std::size_t> index_dist(0U, byte_size - 1U);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  for (std::size_t iteration = 0; iteration < 64U; ++iteration) {
    std::vector<std::byte> mutated = original;
    const std::size_t mutation_count = 1U + (iteration % 8U);
    for (std::size_t i = 0; i < mutation_count; ++i) {
      const std::size_t index = index_dist(rng);
      mutated[index] = std::byte{static_cast<std::uint8_t>(byte_dist(rng))};
    }

    mutated[0] = std::byte{static_cast<std::uint8_t>(std::to_integer<std::uint8_t>(mutated[0]) ^ 0xFFU)};
    REQUIRE_THROWS_AS(varre::assets::decode_model_data(ids[0], mutated.data(), mutated.size()), std::runtime_error);
  }
}

TEST_CASE("model codegen corpus verifies valid and invalid OBJ cases", "[models][codegen]") {
  const fs::path valid_dir = fs::path(VARRE_MODELS_CORPUS_VALID_DIR);
  const fs::path invalid_dir = fs::path(VARRE_MODELS_CORPUS_INVALID_DIR);
  REQUIRE(fs::exists(valid_dir));
  REQUIRE(fs::exists(invalid_dir));

  const fs::path temp_root = make_temp_dir("models_codegen_corpus");
  const fs::path valid_output = temp_root / "valid";
  const fs::path invalid_output = temp_root / "invalid";

  const int valid_exit = run_model_codegen(valid_dir, valid_output, "varre::assets::tests::valid");
  REQUIRE(valid_exit == 0);
  REQUIRE(fs::exists(valid_output / "models_generated.hpp"));
  REQUIRE(fs::exists(valid_output / "models_generated.cpp"));

  std::ifstream generated_cpp(valid_output / "models_generated.cpp", std::ios::binary);
  REQUIRE(generated_cpp.is_open());
  const std::string generated_text((std::istreambuf_iterator<char>(generated_cpp)), std::istreambuf_iterator<char>());
  REQUIRE(generated_text.find("negative_indices.obj") != std::string::npos);

  const int invalid_exit = run_model_codegen(invalid_dir, invalid_output, "varre::assets::tests::invalid");
  REQUIRE(invalid_exit != 0);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

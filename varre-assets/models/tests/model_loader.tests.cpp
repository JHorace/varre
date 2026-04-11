#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "varre/assets/models.hpp"

namespace {
std::uint32_t read_u32_le(const std::byte *data, std::size_t offset) {
  const auto b0 = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[offset + 0]));
  const auto b1 = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[offset + 1]));
  const auto b2 = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[offset + 2]));
  const auto b3 = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[offset + 3]));
  return b0 | (b1 << 8U) | (b2 << 16U) | (b3 << 24U);
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

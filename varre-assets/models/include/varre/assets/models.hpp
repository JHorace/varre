#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace varre::assets {

enum class ModelId : std::uint32_t {
  kUnimplemented = 0,
};

struct Vertex {
  float px = 0.0F;
  float py = 0.0F;
  float pz = 0.0F;

  float cx = 1.0F;
  float cy = 1.0F;
  float cz = 1.0F;

  float nx = 0.0F;
  float ny = 1.0F;
  float nz = 0.0F;

  float u = 0.0F;
  float v = 0.0F;
};

struct ModelAsset {
  ModelId id = ModelId::kUnimplemented;
  std::vector<Vertex> vertices;
  std::vector<std::uint32_t> indices;
};

[[nodiscard]] const std::byte* get_model_data(ModelId id, std::size_t* out_size);
[[nodiscard]] ModelAsset load_model(ModelId id);

}  // namespace varre::assets


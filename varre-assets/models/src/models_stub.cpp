#include "varre/assets/models.hpp"

namespace varre::assets {

const std::byte* get_model_data(const ModelId /*id*/, std::size_t* const out_size) {
  if (out_size != nullptr) {
    *out_size = 0;
  }
  return nullptr;
}

ModelAsset load_model(const ModelId id) {
  ModelAsset model;
  model.id = id;
  return model;
}

}  // namespace varre::assets


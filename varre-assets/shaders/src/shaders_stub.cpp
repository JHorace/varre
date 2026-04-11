/**
 * @file shaders_stub.cpp
 * @brief Temporary shader asset API stub implementation.
 */
#include "varre/assets/shaders.hpp"

namespace varre::assets {

/**
 * @brief Placeholder shader lookup implementation.
 * @param id Requested shader identifier.
 * @return Always `nullptr` until shader code generation is implemented.
 */
const ShaderAssetView *get_shader(const ShaderId /*id*/) { return nullptr; }

} // namespace varre::assets

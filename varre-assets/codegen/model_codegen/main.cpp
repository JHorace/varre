/**
 * @file main.cpp
 * @brief Build-time model asset code generator.
 *
 * The executable scans OBJ files, converts them to the engine's compact binary
 * representation, and emits a generated C++ API that exposes embedded assets.
 */
#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tiny_obj_loader.h>

namespace fs = std::filesystem;

namespace {

/**
 * @brief Vertex layout used during conversion and serialization.
 */
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

/**
 * @brief Parsed geometry before binary packing.
 */
struct ParsedMesh {
  std::vector<Vertex> vertices;
  std::vector<std::uint32_t> indices;
};

/**
 * @brief Full parsed model payload and source metadata.
 */
struct ParsedModel {
  ParsedMesh mesh;
  std::string shape_names;
};

/**
 * @brief One generated model entry containing enum name and embedded bytes.
 */
struct ModelBlob {
  std::string enum_name;
  std::string display_name;
  std::string source_path;
  std::string shape_names;
  std::uint32_t format_flags = 0;
  std::vector<std::uint8_t> bytes;
};

/**
 * @brief Coordinate and UV conversion settings applied during import.
 */
struct MeshConventions {
  bool flip_handedness = false;
  bool flip_winding = false;
  bool flip_uv_v = false;
};

constexpr std::uint32_t kModelBinaryMagic = 0x444D5256U; // "VRMD"
constexpr std::uint32_t kModelBinaryVersion = 1U;

constexpr std::uint32_t kModelFormatFlagFlipHandedness = 1U << 0U;
constexpr std::uint32_t kModelFormatFlagFlipWinding = 1U << 1U;
constexpr std::uint32_t kModelFormatFlagFlipUvV = 1U << 2U;

/**
 * @brief Command-line options for the code generator.
 */
struct CliOptions {
  fs::path input_root;
  fs::path output_dir;
  fs::path emit_hpp;
  fs::path emit_cpp;
  std::string ns = "varre::assets";
  MeshConventions conventions;
};

/**
 * @brief Throw a formatted fatal error.
 * @param message Human-readable error message.
 */
[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }

/**
 * @brief Throw an OBJ-related fatal error with shape/face context.
 * @param file Source OBJ file.
 * @param shape_index Shape index from tinyobj.
 * @param face_index Face index inside shape.
 * @param message Detailed error message.
 */
[[noreturn]] void fail_obj_shape_face(const fs::path &file, std::size_t shape_index, std::size_t face_index, const std::string &message) {
  std::ostringstream oss;
  oss << file.generic_string() << ": shape " << shape_index << ", face " << face_index << ": " << message;
  fail(oss.str());
}

/**
 * @brief Normalize text into a deterministic C++ enum identifier fragment.
 * @param value Raw string.
 * @return Uppercase identifier-safe string.
 */
std::string sanitize_identifier(const std::string &value) {
  std::string out;
  out.reserve(value.size() + 8);

  for (char ch : value) {
    if (std::isalnum(static_cast<unsigned char>(ch)) != 0) {
      out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
    } else {
      out.push_back('_');
    }
  }

  // Collapse repeated underscores for cleaner IDs.
  std::string compact;
  compact.reserve(out.size());
  bool prev_is_underscore = false;
  for (char ch : out) {
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
    compact = "MODEL";
  }
  if (std::isdigit(static_cast<unsigned char>(compact.front())) != 0) {
    compact.insert(compact.begin(), '_');
  }

  return compact;
}

/**
 * @brief Build a model enum name from an OBJ path.
 * @param input_root Root path used for relative naming.
 * @param obj_path OBJ path.
 * @return Deterministic enum identifier.
 */
std::string make_obj_enum_name(const fs::path &input_root, const fs::path &obj_path) {
  fs::path rel = obj_path.lexically_relative(input_root);
  if (rel.empty()) {
    rel = obj_path.filename();
  }
  rel.replace_extension();
  return sanitize_identifier(rel.generic_string());
}

/**
 * @brief Discover OBJ files under the input root.
 * @param input_root Root directory to scan recursively.
 * @return Deterministically sorted OBJ path list.
 */
std::vector<fs::path> discover_obj_files(const fs::path &input_root) {
  std::vector<fs::path> obj_files;
  if (!fs::exists(input_root)) {
    return obj_files;
  }

  for (const fs::directory_entry &entry : fs::recursive_directory_iterator(input_root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path path = entry.path();
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (ext == ".obj") {
      obj_files.push_back(path);
    }
  }

  std::sort(obj_files.begin(), obj_files.end(), [](const fs::path &a, const fs::path &b) { return a.generic_string() < b.generic_string(); });
  return obj_files;
}

/**
 * @brief Encode mesh-convention settings into model-format flags.
 * @param conventions Active conversion settings.
 * @return Bit-mask stored in generated binary header.
 */
std::uint32_t to_format_flags(const MeshConventions &conventions) {
  std::uint32_t flags = 0U;
  if (conventions.flip_handedness) {
    flags |= kModelFormatFlagFlipHandedness;
  }
  if (conventions.flip_winding) {
    flags |= kModelFormatFlagFlipWinding;
  }
  if (conventions.flip_uv_v) {
    flags |= kModelFormatFlagFlipUvV;
  }
  return flags;
}

/**
 * @brief Parse an OBJ file into a triangulated mesh.
 * @param obj_path Path to OBJ file.
 * @return Parsed mesh data.
 */
ParsedModel parse_obj(const fs::path &obj_path, const MeshConventions &conventions) {
  tinyobj::ObjReaderConfig config;
  config.triangulate = true;
  config.vertex_color = false;
  if (obj_path.has_parent_path()) {
    config.mtl_search_path = obj_path.parent_path().string();
  }

  tinyobj::ObjReader reader;
  if (!reader.ParseFromFile(obj_path.string(), config)) {
    std::ostringstream oss;
    oss << "tinyobj failed to parse OBJ: " << obj_path.generic_string();
    if (!reader.Error().empty()) {
      oss << " (" << reader.Error() << ")";
    }
    fail(oss.str());
  }

  if (!reader.Warning().empty()) {
    std::cerr << "model_codegen: warning: " << obj_path.generic_string() << ": " << reader.Warning() << '\n';
  }

  const tinyobj::attrib_t &attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t> &shapes = reader.GetShapes();

  if (shapes.empty()) {
    fail("OBJ file has no shapes: " + obj_path.generic_string());
  }

  struct VertexKey {
    int v = -1;
    int vt = -1;
    int vn = -1;

    bool operator==(const VertexKey &other) const { return v == other.v && vt == other.vt && vn == other.vn; }
  };

  struct VertexKeyHash {
    std::size_t operator()(const VertexKey &key) const {
      // Small deterministic hash for OBJ index triplets.
      std::size_t seed = static_cast<std::size_t>(key.v);
      seed ^= static_cast<std::size_t>(key.vt) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
      seed ^= static_cast<std::size_t>(key.vn) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
      return seed;
    }
  };

  ParsedMesh mesh;
  std::unordered_map<VertexKey, std::uint32_t, VertexKeyHash> vertex_map;
  std::vector<bool> vertex_has_explicit_normal;
  std::vector<std::string> shape_names;
  shape_names.reserve(shapes.size());
  for (std::size_t shape_index = 0; shape_index < shapes.size(); ++shape_index) {
    const tinyobj::shape_t &shape = shapes[shape_index];
    if (shape.name.empty()) {
      shape_names.push_back(std::string("<unnamed#") + std::to_string(shape_index) + ">");
    } else {
      shape_names.push_back(shape.name);
    }
    std::size_t offset = 0;

    for (std::size_t face_index = 0; face_index < shape.mesh.num_face_vertices.size(); ++face_index) {
      const int face_vertex_count = shape.mesh.num_face_vertices[face_index];
      if (face_vertex_count < 3) {
        fail_obj_shape_face(obj_path, shape_index, face_index, "face has fewer than 3 vertices");
      }

      std::vector<std::uint32_t> face_indices;
      face_indices.reserve(static_cast<std::size_t>(face_vertex_count));

      for (int i = 0; i < face_vertex_count; ++i) {
        const std::size_t idx_offset = offset + static_cast<std::size_t>(i);
        if (idx_offset >= shape.mesh.indices.size()) {
          fail_obj_shape_face(obj_path, shape_index, face_index, "index buffer is shorter than expected");
        }

        const tinyobj::index_t &idx = shape.mesh.indices[idx_offset];
        if (idx.vertex_index < 0) {
          fail_obj_shape_face(obj_path, shape_index, face_index, "vertex index is negative");
        }

        const VertexKey key{
          .v = idx.vertex_index,
          .vt = idx.texcoord_index,
          .vn = idx.normal_index,
        };

        const auto found = vertex_map.find(key);
        if (found != vertex_map.end()) {
          face_indices.push_back(found->second);
          continue;
        }

        const std::size_t vp = static_cast<std::size_t>(key.v) * 3U;
        if (vp + 2U >= attrib.vertices.size()) {
          fail_obj_shape_face(obj_path, shape_index, face_index, "vertex index is out of range");
        }

        Vertex vertex{};
        vertex.px = attrib.vertices[vp + 0U];
        vertex.py = attrib.vertices[vp + 1U];
        vertex.pz = attrib.vertices[vp + 2U];
        if (conventions.flip_handedness) {
          vertex.pz = -vertex.pz;
        }

        if (key.vt >= 0) {
          const std::size_t tp = static_cast<std::size_t>(key.vt) * 2U;
          if (tp + 1U >= attrib.texcoords.size()) {
            fail_obj_shape_face(obj_path, shape_index, face_index, "uv index is out of range");
          }
          vertex.u = attrib.texcoords[tp + 0U];
          vertex.v = attrib.texcoords[tp + 1U];
          if (conventions.flip_uv_v) {
            vertex.v = 1.0F - vertex.v;
          }
        }

        if (key.vn >= 0) {
          const std::size_t np = static_cast<std::size_t>(key.vn) * 3U;
          if (np + 2U >= attrib.normals.size()) {
            fail_obj_shape_face(obj_path, shape_index, face_index, "normal index is out of range");
          }
          vertex.nx = attrib.normals[np + 0U];
          vertex.ny = attrib.normals[np + 1U];
          vertex.nz = attrib.normals[np + 2U];
          if (conventions.flip_handedness) {
            vertex.nz = -vertex.nz;
          }
        }

        if (mesh.vertices.size() >= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
          fail("vertex count exceeds 32-bit index limit for: " + obj_path.generic_string());
        }
        mesh.vertices.push_back(vertex);
        const std::uint32_t new_index = static_cast<std::uint32_t>(mesh.vertices.size() - 1U);
        vertex_map.emplace(key, new_index);
        vertex_has_explicit_normal.push_back(key.vn >= 0);
        face_indices.push_back(new_index);
      }

      for (std::size_t i = 1; i + 1 < face_indices.size(); ++i) {
        if (conventions.flip_winding) {
          mesh.indices.push_back(face_indices[0]);
          mesh.indices.push_back(face_indices[i + 1]);
          mesh.indices.push_back(face_indices[i]);
        } else {
          mesh.indices.push_back(face_indices[0]);
          mesh.indices.push_back(face_indices[i]);
          mesh.indices.push_back(face_indices[i + 1]);
        }
      }

      offset += static_cast<std::size_t>(face_vertex_count);
    }

    if (offset != shape.mesh.indices.size()) {
      std::ostringstream oss;
      oss << "shape index count mismatch (consumed=" << offset << ", total=" << shape.mesh.indices.size() << ")";
      fail_obj_shape_face(obj_path, shape_index, 0, oss.str());
    }
  }

  if (mesh.vertices.empty() || mesh.indices.empty()) {
    fail("OBJ file has no usable geometry: " + obj_path.generic_string());
  }

  // Generate vertex normals only where OBJ data did not provide them.
  bool has_missing_normals = false;
  for (const bool has_normal : vertex_has_explicit_normal) {
    if (!has_normal) {
      has_missing_normals = true;
      break;
    }
  }

  if (has_missing_normals) {
    std::vector<std::array<float, 3>> normal_sums(mesh.vertices.size(), {0.0F, 0.0F, 0.0F});

    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
      const std::uint32_t i0 = mesh.indices[i + 0];
      const std::uint32_t i1 = mesh.indices[i + 1];
      const std::uint32_t i2 = mesh.indices[i + 2];

      if (static_cast<std::size_t>(i0) >= mesh.vertices.size() || static_cast<std::size_t>(i1) >= mesh.vertices.size() ||
          static_cast<std::size_t>(i2) >= mesh.vertices.size()) {
        fail("generated index out of range while computing normals for: " + obj_path.generic_string());
      }

      const Vertex &v0 = mesh.vertices[i0];
      const Vertex &v1 = mesh.vertices[i1];
      const Vertex &v2 = mesh.vertices[i2];

      const float e1x = v1.px - v0.px;
      const float e1y = v1.py - v0.py;
      const float e1z = v1.pz - v0.pz;
      const float e2x = v2.px - v0.px;
      const float e2y = v2.py - v0.py;
      const float e2z = v2.pz - v0.pz;

      const float nx = (e1y * e2z) - (e1z * e2y);
      const float ny = (e1z * e2x) - (e1x * e2z);
      const float nz = (e1x * e2y) - (e1y * e2x);
      const float normal_sign = conventions.flip_winding ? -1.0F : 1.0F;

      const auto add_face_normal = [&](const std::uint32_t idx) {
        if (!vertex_has_explicit_normal[idx]) {
          normal_sums[idx][0] += nx * normal_sign;
          normal_sums[idx][1] += ny * normal_sign;
          normal_sums[idx][2] += nz * normal_sign;
        }
      };

      add_face_normal(i0);
      add_face_normal(i1);
      add_face_normal(i2);
    }

    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
      if (vertex_has_explicit_normal[i]) {
        continue;
      }

      const float nx = normal_sums[i][0];
      const float ny = normal_sums[i][1];
      const float nz = normal_sums[i][2];
      const float len = std::sqrt((nx * nx) + (ny * ny) + (nz * nz));

      if (len > 1e-8F) {
        mesh.vertices[i].nx = nx / len;
        mesh.vertices[i].ny = ny / len;
        mesh.vertices[i].nz = nz / len;
      } else {
        mesh.vertices[i].nx = 0.0F;
        mesh.vertices[i].ny = 1.0F;
        mesh.vertices[i].nz = 0.0F;
      }
    }
  }

  std::ostringstream shape_summary;
  for (std::size_t i = 0; i < shape_names.size(); ++i) {
    if (i > 0) {
      shape_summary << ",";
    }
    shape_summary << shape_names[i];
  }

  return ParsedModel{
    .mesh = std::move(mesh),
    .shape_names = shape_summary.str(),
  };
}

/**
 * @brief Append a 32-bit unsigned integer in little-endian encoding.
 * @param bytes Destination byte vector.
 * @param value Value to encode.
 */
void append_u32_le(std::vector<std::uint8_t> *bytes, const std::uint32_t value) {
  bytes->push_back(static_cast<std::uint8_t>(value & 0xFFU));
  bytes->push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
  bytes->push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
  bytes->push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

/**
 * @brief Append a 32-bit float in little-endian encoding.
 * @param bytes Destination byte vector.
 * @param value Value to encode.
 */
void append_f32_le(std::vector<std::uint8_t> *bytes, const float value) { append_u32_le(bytes, std::bit_cast<std::uint32_t>(value)); }

/**
 * @brief Encode parsed mesh data into the engine binary layout.
 * @param mesh Parsed mesh.
 * @param format_flags Format flags describing applied import conventions.
 * @return Serialized binary payload.
 */
std::vector<std::uint8_t> encode_binary(const ParsedMesh &mesh, const std::uint32_t format_flags) {
  if (mesh.vertices.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    fail("vertex count exceeds 32-bit range during binary encode");
  }
  if (mesh.indices.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    fail("index count exceeds 32-bit range during binary encode");
  }

  std::vector<std::uint8_t> bytes;
  bytes.reserve(16 + mesh.vertices.size() * 44 + mesh.indices.size() * 4);

  append_u32_le(&bytes, kModelBinaryMagic);
  append_u32_le(&bytes, kModelBinaryVersion);
  append_u32_le(&bytes, format_flags);
  append_u32_le(&bytes, static_cast<std::uint32_t>(mesh.vertices.size()));
  for (const Vertex &v : mesh.vertices) {
    append_f32_le(&bytes, v.px);
    append_f32_le(&bytes, v.py);
    append_f32_le(&bytes, v.pz);
    append_f32_le(&bytes, v.cx);
    append_f32_le(&bytes, v.cy);
    append_f32_le(&bytes, v.cz);
    append_f32_le(&bytes, v.nx);
    append_f32_le(&bytes, v.ny);
    append_f32_le(&bytes, v.nz);
    append_f32_le(&bytes, v.u);
    append_f32_le(&bytes, v.v);
  }

  append_u32_le(&bytes, static_cast<std::uint32_t>(mesh.indices.size()));
  for (const std::uint32_t idx : mesh.indices) {
    append_u32_le(&bytes, idx);
  }
  return bytes;
}

/**
 * @brief Apply coordinate and UV conventions to a mesh in-place.
 * @param mesh Mesh to modify.
 * @param conventions Conversion settings.
 */
void apply_mesh_conventions(ParsedMesh *mesh, const MeshConventions &conventions) {
  for (Vertex &vertex : mesh->vertices) {
    if (conventions.flip_handedness) {
      vertex.pz = -vertex.pz;
      vertex.nz = -vertex.nz;
    }
    if (conventions.flip_uv_v) {
      vertex.v = 1.0F - vertex.v;
    }
  }

  if (conventions.flip_winding) {
    for (std::size_t i = 0; i + 2 < mesh->indices.size(); i += 3) {
      std::swap(mesh->indices[i + 1], mesh->indices[i + 2]);
    }
  }
}

/**
 * @brief Build a synthetic cube mesh for a guaranteed built-in asset.
 * @param conventions Conversion settings.
 * @return Parsed mesh representing a cube.
 */
ParsedMesh make_cube_mesh(const MeshConventions &conventions) {
  ParsedMesh mesh;

  struct FaceDef {
    std::array<std::array<float, 3>, 4> positions;
    std::array<float, 3> normal;
  };

  constexpr float h = 0.5F;
  const std::array<FaceDef, 6> faces = {{
    {{{{-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}}}, {0.0F, 0.0F, 1.0F}},
    {{{{h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h}}}, {1.0F, 0.0F, 0.0F}},
    {{{{h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h}}}, {0.0F, 0.0F, -1.0F}},
    {{{{-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h}}}, {-1.0F, 0.0F, 0.0F}},
    {{{{-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h}}}, {0.0F, 1.0F, 0.0F}},
    {{{{-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h}}}, {0.0F, -1.0F, 0.0F}},
  }};

  const std::array<std::array<float, 2>, 4> uv = {{{0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F}}};

  for (std::size_t face = 0; face < faces.size(); ++face) {
    const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
    for (std::size_t i = 0; i < 4; ++i) {
      Vertex v{};
      v.px = faces[face].positions[i][0];
      v.py = faces[face].positions[i][1];
      v.pz = faces[face].positions[i][2];
      v.nx = faces[face].normal[0];
      v.ny = faces[face].normal[1];
      v.nz = faces[face].normal[2];
      v.u = uv[i][0];
      v.v = uv[i][1];
      mesh.vertices.push_back(v);
    }

    mesh.indices.push_back(base + 0U);
    mesh.indices.push_back(base + 1U);
    mesh.indices.push_back(base + 2U);
    mesh.indices.push_back(base + 2U);
    mesh.indices.push_back(base + 3U);
    mesh.indices.push_back(base + 0U);
  }

  apply_mesh_conventions(&mesh, conventions);
  return mesh;
}

/**
 * @brief Format bytes as a C++ hexadecimal initializer body.
 * @param bytes Binary payload.
 * @return Multi-line initializer text.
 */
std::string make_byte_initializer(const std::vector<std::uint8_t> &bytes) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (i % 16 == 0) {
      oss << "\n    ";
    }
    oss << "0x" << std::setw(2) << static_cast<unsigned>(bytes[i]) << ",";
    if (i % 16 != 15) {
      oss << ' ';
    }
  }
  oss << '\n';
  return oss.str();
}

/**
 * @brief Escape a string for safe emission as a C++ string literal.
 * @param value Raw string.
 * @return Escaped literal content (without surrounding quotes).
 */
std::string escape_cpp_string(const std::string &value) {
  std::string out;
  out.reserve(value.size() + 16);
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
 * @brief Write output file only when content changed.
 * @param output_path Destination file path.
 * @param content New content.
 */
void write_if_changed(const fs::path &output_path, const std::string &content) {
  std::error_code ec;
  if (fs::exists(output_path, ec)) {
    std::ifstream in(output_path, std::ios::binary);
    std::string existing((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (existing == content) {
      return;
    }
  }

  fs::create_directories(output_path.parent_path());
  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    fail("could not open output for write: " + output_path.generic_string());
  }
  out << content;
  out.close();
}

/**
 * @brief Emit generated public header text for model assets.
 * @param ns Target C++ namespace.
 * @param models Ordered model records.
 * @return Header file contents.
 */
std::string emit_header(const std::string &ns, const std::vector<ModelBlob> &models) {
  std::ostringstream hpp;
  hpp << "/**\n";
  hpp << " * @file models_generated.hpp\n";
  hpp << " * @brief Generated model asset API.\n";
  hpp << " *\n";
  hpp << " * This file is auto-generated by model_codegen. Do not edit manually.\n";
  hpp << " */\n";
  hpp << "#pragma once\n\n";
  hpp << "#include <cstddef>\n";
  hpp << "#include <cstdint>\n";
  hpp << "#include <vector>\n\n";
  hpp << "namespace " << ns << " {\n\n";
  hpp << "/** @brief Stable identifiers for embedded model assets. */\n";
  hpp << "enum class ModelId : std::uint32_t {\n";
  for (std::size_t i = 0; i < models.size(); ++i) {
    hpp << "  " << models[i].enum_name << " = " << i << ",\n";
  }
  hpp << "};\n\n";
  hpp << "/** @brief Runtime vertex layout decoded from generated model blobs. */\n";
  hpp << "struct Vertex {\n";
  hpp << "  float px = 0.0F;\n";
  hpp << "  float py = 0.0F;\n";
  hpp << "  float pz = 0.0F;\n";
  hpp << "  float cx = 1.0F;\n";
  hpp << "  float cy = 1.0F;\n";
  hpp << "  float cz = 1.0F;\n";
  hpp << "  float nx = 0.0F;\n";
  hpp << "  float ny = 1.0F;\n";
  hpp << "  float nz = 0.0F;\n";
  hpp << "  float u = 0.0F;\n";
  hpp << "  float v = 0.0F;\n";
  hpp << "};\n\n";
  hpp << "/** @brief Decoded model data for engine-side consumption. */\n";
  hpp << "struct ModelAsset {\n";
  hpp << "  ModelId id = ModelId::" << models.front().enum_name << ";\n";
  hpp << "  std::vector<Vertex> vertices;\n";
  hpp << "  std::vector<std::uint32_t> indices;\n";
  hpp << "};\n\n";
  hpp << "/**\n";
  hpp << " * @brief Access raw binary data for a model.\n";
  hpp << " * @param id Model identifier.\n";
  hpp << " * @param out_size Optional output byte size.\n";
  hpp << " * @return Pointer to immutable model bytes.\n";
  hpp << " */\n";
  hpp << "[[nodiscard]] const std::byte* get_model_data(ModelId id, std::size_t* out_size);\n";
  hpp << "/**\n";
  hpp << " * @brief Decode an arbitrary model blob using the generated format parser.\n";
  hpp << " * @param id Model identifier used for diagnostics.\n";
  hpp << " * @param data Pointer to encoded model bytes.\n";
  hpp << " * @param size Size of encoded model data in bytes.\n";
  hpp << " * @return Decoded model object.\n";
  hpp << " */\n";
  hpp << "[[nodiscard]] ModelAsset decode_model_data(ModelId id, const std::byte* data, std::size_t size);\n";
  hpp << "/**\n";
  hpp << " * @brief Decode a model blob into runtime vectors.\n";
  hpp << " * @param id Model identifier.\n";
  hpp << " * @return Decoded model object.\n";
  hpp << " */\n";
  hpp << "[[nodiscard]] ModelAsset load_model(ModelId id);\n";
  hpp << "/**\n";
  hpp << " * @brief Resolve the display name for a model.\n";
  hpp << " * @param id Model identifier.\n";
  hpp << " * @return Null-terminated model display name.\n";
  hpp << " */\n";
  hpp << "[[nodiscard]] const char* model_name(ModelId id);\n";
  hpp << "/**\n";
  hpp << " * @brief Access the full set of generated model identifiers.\n";
  hpp << " * @param out_count Optional output count.\n";
  hpp << " * @return Pointer to immutable contiguous model-id array.\n";
  hpp << " */\n";
  hpp << "[[nodiscard]] const ModelId* all_model_ids(std::size_t* out_count);\n\n";
  hpp << "}  // namespace " << ns << "\n";
  return hpp.str();
}

/**
 * @brief Emit generated implementation text for model assets.
 * @param ns Target C++ namespace.
 * @param models Ordered model records.
 * @return Source file contents.
 */
std::string emit_cpp(const std::string &ns, const std::vector<ModelBlob> &models) {
  std::ostringstream cpp;
  cpp << "/**\n";
  cpp << " * @file models_generated.cpp\n";
  cpp << " * @brief Generated model asset implementation.\n";
  cpp << " *\n";
  cpp << " * This file is auto-generated by model_codegen. Do not edit manually.\n";
  cpp << " */\n";
  cpp << "#include \"models_generated.hpp\"\n\n";
  cpp << "#include <array>\n";
  cpp << "#include <cstring>\n";
  cpp << "#include <limits>\n";
  cpp << "#include <stdexcept>\n";
  cpp << "#include <string>\n\n";
  cpp << "namespace " << ns << " {\n\n";
  cpp << "namespace {\n\n";
  cpp << "struct ModelRecord {\n";
  cpp << "  ModelId id;\n";
  cpp << "  const char* name;\n";
  cpp << "  const char* source_path;\n";
  cpp << "  const char* shape_names;\n";
  cpp << "  std::uint32_t format_flags;\n";
  cpp << "  const std::uint8_t* data;\n";
  cpp << "  std::size_t size;\n";
  cpp << "};\n\n";
  cpp << "constexpr std::uint32_t kModelBinaryMagic = 0x" << std::hex << std::setw(8) << std::setfill('0') << kModelBinaryMagic << std::dec << "U;\n";
  cpp << "constexpr std::uint32_t kModelBinaryVersion = " << kModelBinaryVersion << "U;\n";
  cpp << "constexpr std::size_t kVertexStrideBytes = 44U;\n";
  cpp << "constexpr std::uint32_t kMaxVertexCount = 50'000'000U;\n";
  cpp << "constexpr std::uint32_t kMaxIndexCount = 150'000'000U;\n\n";

  for (const ModelBlob &model : models) {
    cpp << "alignas(4) static constexpr std::uint8_t kModelData_" << model.enum_name << "[] = {";
    cpp << make_byte_initializer(model.bytes);
    cpp << "};\n\n";
  }

  cpp << "static constexpr std::array<ModelRecord, " << models.size() << "> kModels = {{\n";
  for (const ModelBlob &model : models) {
    const std::string display_name_escaped = escape_cpp_string(model.display_name);
    const std::string source_path_escaped = escape_cpp_string(model.source_path);
    const std::string shape_names_escaped = escape_cpp_string(model.shape_names);
    cpp << "    {ModelId::" << model.enum_name << ", \"" << display_name_escaped << "\", "
        << "\"" << source_path_escaped << "\", "
        << "\"" << shape_names_escaped << "\", " << model.format_flags << "U, "
        << "kModelData_" << model.enum_name << ", sizeof(kModelData_" << model.enum_name << ")},\n";
  }
  cpp << "}};\n\n";

  cpp << "static constexpr std::array<ModelId, " << models.size() << "> kModelIds = {{\n";
  for (const ModelBlob &model : models) {
    cpp << "    ModelId::" << model.enum_name << ",\n";
  }
  cpp << "}};\n\n";

  cpp << "const ModelRecord* find_record(ModelId id) {\n";
  cpp << "  for (const ModelRecord& record : kModels) {\n";
  cpp << "    if (record.id == id) {\n";
  cpp << "      return &record;\n";
  cpp << "    }\n";
  cpp << "  }\n";
  cpp << "  return nullptr;\n";
  cpp << "}\n\n";

  cpp << "std::string model_context(const ModelRecord* record) {\n";
  cpp << "  return std::string(\"model \") + record->name + \" (source: \" + record->source_path + \", shapes: \" + record->shape_names + \")\";\n";
  cpp << "}\n\n";

  cpp << "[[noreturn]] void fail_decode(const ModelRecord* record, const std::string& reason) {\n";
  cpp << "  throw std::runtime_error(model_context(record) + \": \" + reason);\n";
  cpp << "}\n\n";

  cpp << "bool checked_add(std::size_t a, std::size_t b, std::size_t* out) {\n";
  cpp << "  if (a > std::numeric_limits<std::size_t>::max() - b) {\n";
  cpp << "    return false;\n";
  cpp << "  }\n";
  cpp << "  *out = a + b;\n";
  cpp << "  return true;\n";
  cpp << "}\n\n";

  cpp << "bool checked_mul(std::size_t a, std::size_t b, std::size_t* out) {\n";
  cpp << "  if (a != 0U && b > std::numeric_limits<std::size_t>::max() / a) {\n";
  cpp << "    return false;\n";
  cpp << "  }\n";
  cpp << "  *out = a * b;\n";
  cpp << "  return true;\n";
  cpp << "}\n\n";

  cpp << "std::uint32_t read_u32(const std::uint8_t* data, std::size_t size, std::size_t* offset, const ModelRecord* record, const char* field) {\n";
  cpp << "  if (*offset > size || size - *offset < 4U) {\n";
  cpp << "    fail_decode(record, std::string(\"truncated while reading \") + field);\n";
  cpp << "  }\n";
  cpp << "  const std::uint32_t value =\n";
  cpp << "      static_cast<std::uint32_t>(data[*offset]) |\n";
  cpp << "      (static_cast<std::uint32_t>(data[*offset + 1]) << 8U) |\n";
  cpp << "      (static_cast<std::uint32_t>(data[*offset + 2]) << 16U) |\n";
  cpp << "      (static_cast<std::uint32_t>(data[*offset + 3]) << 24U);\n";
  cpp << "  *offset += 4U;\n";
  cpp << "  return value;\n";
  cpp << "}\n\n";

  cpp << "float read_f32(const std::uint8_t* data, std::size_t size, std::size_t* offset, const ModelRecord* record, const char* field) {\n";
  cpp << "  const std::uint32_t bits = read_u32(data, size, offset, record, field);\n";
  cpp << "  float out = 0.0F;\n";
  cpp << "  std::memcpy(&out, &bits, sizeof(float));\n";
  cpp << "  return out;\n";
  cpp << "}\n\n";
  cpp << "}  // namespace\n\n";

  cpp << "const ModelId* all_model_ids(std::size_t* out_count) {\n";
  cpp << "  if (out_count != nullptr) {\n";
  cpp << "    *out_count = kModelIds.size();\n";
  cpp << "  }\n";
  cpp << "  return kModelIds.data();\n";
  cpp << "}\n\n";

  cpp << "const char* model_name(ModelId id) {\n";
  cpp << "  const ModelRecord* record = find_record(id);\n";
  cpp << "  return record == nullptr ? \"<unknown>\" : record->name;\n";
  cpp << "}\n\n";

  cpp << "const std::byte* get_model_data(ModelId id, std::size_t* out_size) {\n";
  cpp << "  const ModelRecord* record = find_record(id);\n";
  cpp << "  if (record == nullptr) {\n";
  cpp << "    throw std::runtime_error(std::string(\"unknown ModelId: \") + std::to_string(static_cast<std::uint32_t>(id)));\n";
  cpp << "  }\n";
  cpp << "  if (out_size != nullptr) {\n";
  cpp << "    *out_size = record->size;\n";
  cpp << "  }\n";
  cpp << "  return reinterpret_cast<const std::byte*>(record->data);\n";
  cpp << "}\n\n";

  cpp << "ModelAsset decode_model_data(ModelId id, const std::byte* blob_data, std::size_t blob_size) {\n";
  cpp << "  const ModelRecord* record = find_record(id);\n";
  cpp << "  const ModelRecord unknown_record{id, \"<unknown>\", \"<external>\", \"<external>\", 0U, nullptr, 0U};\n";
  cpp << "  if (record == nullptr) {\n";
  cpp << "    record = &unknown_record;\n";
  cpp << "  }\n";
  cpp << "  if (blob_data == nullptr) {\n";
  cpp << "    fail_decode(record, \"blob_data is null\");\n";
  cpp << "  }\n\n";
  cpp << "  const std::uint8_t* data = reinterpret_cast<const std::uint8_t*>(blob_data);\n";
  cpp << "  const std::size_t size = blob_size;\n";
  cpp << "  std::size_t offset = 0;\n\n";
  cpp << "  const std::uint32_t magic = read_u32(data, size, &offset, record, \"magic\");\n";
  cpp << "  if (magic != kModelBinaryMagic) {\n";
  cpp << "    fail_decode(record, \"invalid model binary magic\");\n";
  cpp << "  }\n";
  cpp << "  const std::uint32_t version = read_u32(data, size, &offset, record, \"version\");\n";
  cpp << "  if (version != kModelBinaryVersion) {\n";
  cpp << "    fail_decode(record, \"unsupported model binary version\");\n";
  cpp << "  }\n";
  cpp << "  const std::uint32_t _flags = read_u32(data, size, &offset, record, \"flags\");\n";
  cpp << "  (void)_flags;\n";

  cpp << "  ModelAsset model;\n";
  cpp << "  model.id = id;\n\n";
  cpp << "  const std::uint32_t vertex_count = read_u32(data, size, &offset, record, \"vertex_count\");\n";
  cpp << "  if (vertex_count > kMaxVertexCount) {\n";
  cpp << "    fail_decode(record, \"vertex_count exceeds safety limit\");\n";
  cpp << "  }\n";
  cpp << "  std::size_t vertex_bytes = 0;\n";
  cpp << "  if (!checked_mul(static_cast<std::size_t>(vertex_count), kVertexStrideBytes, &vertex_bytes)) {\n";
  cpp << "    fail_decode(record, \"vertex payload size overflow\");\n";
  cpp << "  }\n";
  cpp << "  std::size_t vertex_end = 0;\n";
  cpp << "  if (!checked_add(offset, vertex_bytes, &vertex_end) || vertex_end > size) {\n";
  cpp << "    fail_decode(record, \"truncated vertex payload\");\n";
  cpp << "  }\n";
  cpp << "  model.vertices.reserve(static_cast<std::size_t>(vertex_count));\n";
  cpp << "  for (std::uint32_t i = 0; i < vertex_count; ++i) {\n";
  cpp << "    Vertex v;\n";
  cpp << "    v.px = read_f32(data, size, &offset, record, \"px\");\n";
  cpp << "    v.py = read_f32(data, size, &offset, record, \"py\");\n";
  cpp << "    v.pz = read_f32(data, size, &offset, record, \"pz\");\n";
  cpp << "    v.cx = read_f32(data, size, &offset, record, \"cx\");\n";
  cpp << "    v.cy = read_f32(data, size, &offset, record, \"cy\");\n";
  cpp << "    v.cz = read_f32(data, size, &offset, record, \"cz\");\n";
  cpp << "    v.nx = read_f32(data, size, &offset, record, \"nx\");\n";
  cpp << "    v.ny = read_f32(data, size, &offset, record, \"ny\");\n";
  cpp << "    v.nz = read_f32(data, size, &offset, record, \"nz\");\n";
  cpp << "    v.u = read_f32(data, size, &offset, record, \"u\");\n";
  cpp << "    v.v = read_f32(data, size, &offset, record, \"v\");\n";
  cpp << "    model.vertices.push_back(v);\n";
  cpp << "  }\n\n";
  cpp << "  const std::uint32_t index_count = read_u32(data, size, &offset, record, \"index_count\");\n";
  cpp << "  if (index_count > kMaxIndexCount) {\n";
  cpp << "    fail_decode(record, \"index_count exceeds safety limit\");\n";
  cpp << "  }\n";
  cpp << "  std::size_t index_bytes = 0;\n";
  cpp << "  if (!checked_mul(static_cast<std::size_t>(index_count), sizeof(std::uint32_t), &index_bytes)) {\n";
  cpp << "    fail_decode(record, \"index payload size overflow\");\n";
  cpp << "  }\n";
  cpp << "  std::size_t index_end = 0;\n";
  cpp << "  if (!checked_add(offset, index_bytes, &index_end) || index_end > size) {\n";
  cpp << "    fail_decode(record, \"truncated index payload\");\n";
  cpp << "  }\n";
  cpp << "  model.indices.reserve(static_cast<std::size_t>(index_count));\n";
  cpp << "  for (std::uint32_t i = 0; i < index_count; ++i) {\n";
  cpp << "    const std::uint32_t index = read_u32(data, size, &offset, record, \"index\");\n";
  cpp << "    if (index >= vertex_count) {\n";
  cpp << "      fail_decode(record, \"index references vertex out of range\");\n";
  cpp << "    }\n";
  cpp << "    model.indices.push_back(index);\n";
  cpp << "  }\n\n";
  cpp << "  if (offset != size) {\n";
  cpp << "    fail_decode(record, \"trailing bytes in asset blob\");\n";
  cpp << "  }\n";
  cpp << "  return model;\n";
  cpp << "}\n\n";

  cpp << "ModelAsset load_model(ModelId id) {\n";
  cpp << "  std::size_t size = 0;\n";
  cpp << "  const std::byte* data = get_model_data(id, &size);\n";
  cpp << "  return decode_model_data(id, data, size);\n";
  cpp << "}\n\n";

  cpp << "}  // namespace " << ns << "\n";
  return cpp.str();
}

/**
 * @brief Parse command-line arguments.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Parsed CLI options.
 */
CliOptions parse_cli(int argc, char **argv) {
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
    } else if (arg == "--flip-handedness") {
      opts.conventions.flip_handedness = true;
    } else if (arg == "--flip-winding") {
      opts.conventions.flip_winding = true;
    } else if (arg == "--flip-uv-v") {
      opts.conventions.flip_uv_v = true;
    } else {
      fail("unknown argument: " + std::string(arg));
    }
  }

  if (opts.input_root.empty() || opts.output_dir.empty() || opts.emit_hpp.empty() || opts.emit_cpp.empty()) {
    fail("usage: model_codegen --input-root <dir> --output-dir <dir> --emit-hpp <file> --emit-cpp <file> "
         "[--namespace <ns>] [--flip-handedness] [--flip-winding] [--flip-uv-v]");
  }

  return opts;
}

} // namespace

/**
 * @brief Program entry point.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return `0` on success, non-zero on failure.
 */
int main(int argc, char **argv) {
  try {
    const CliOptions opts = parse_cli(argc, argv);
    const std::uint32_t format_flags = to_format_flags(opts.conventions);

    std::error_code ec;
    fs::create_directories(opts.output_dir, ec);
    if (ec) {
      fail("failed to create output dir: " + opts.output_dir.generic_string() + " (" + ec.message() + ")");
    }

    std::vector<ModelBlob> models;
    models.push_back(ModelBlob{
      .enum_name = "CUBE",
      .display_name = "cube",
      .source_path = "<builtin>",
      .shape_names = "cube",
      .format_flags = format_flags,
      .bytes = encode_binary(make_cube_mesh(opts.conventions), format_flags),
    });

    const std::vector<fs::path> obj_files = discover_obj_files(opts.input_root);
    std::unordered_map<std::string, std::size_t> collision_count;

    for (const fs::path &obj_path : obj_files) {
      ParsedModel parsed_model;
      try {
        parsed_model = parse_obj(obj_path, opts.conventions);
      } catch (const std::exception &ex) {
        std::ostringstream oss;
        oss << "while processing " << obj_path.generic_string() << ": " << ex.what();
        throw std::runtime_error(oss.str());
      }

      std::string enum_name = make_obj_enum_name(opts.input_root, obj_path);
      std::size_t &count = collision_count[enum_name];
      ++count;
      if (count > 1) {
        enum_name += "_" + std::to_string(count);
      }

      fs::path rel = obj_path.lexically_relative(opts.input_root);
      if (rel.empty()) {
        rel = obj_path.filename();
      }

      models.push_back(ModelBlob{
        .enum_name = std::move(enum_name),
        .display_name = rel.generic_string(),
        .source_path = rel.generic_string(),
        .shape_names = std::move(parsed_model.shape_names),
        .format_flags = format_flags,
        .bytes = encode_binary(parsed_model.mesh, format_flags),
      });
    }

    const std::string hpp = emit_header(opts.ns, models);
    const std::string cpp = emit_cpp(opts.ns, models);
    write_if_changed(opts.emit_hpp, hpp);
    write_if_changed(opts.emit_cpp, cpp);

    std::cerr << "model_codegen: generated " << models.size() << " model asset(s) into " << opts.output_dir.generic_string() << '\n';
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "model_codegen: error: " << ex.what() << '\n';
    return 1;
  }
}

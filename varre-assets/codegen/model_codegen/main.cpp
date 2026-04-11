#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

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

struct ParsedMesh {
  std::vector<Vertex> vertices;
  std::vector<std::uint32_t> indices;
};

struct ModelBlob {
  std::string enum_name;
  std::string display_name;
  std::vector<std::uint8_t> bytes;
};

struct CliOptions {
  fs::path input_root;
  fs::path output_dir;
  fs::path emit_hpp;
  fs::path emit_cpp;
  std::string ns = "varre::assets";
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

[[noreturn]] void fail_parse(const fs::path& file, std::size_t line_number, const std::string& message) {
  std::ostringstream oss;
  oss << file.generic_string() << ":" << line_number << ": " << message;
  fail(oss.str());
}

std::string trim(const std::string_view in) {
  std::size_t first = 0;
  while (first < in.size() && std::isspace(static_cast<unsigned char>(in[first])) != 0) {
    ++first;
  }

  std::size_t last = in.size();
  while (last > first && std::isspace(static_cast<unsigned char>(in[last - 1])) != 0) {
    --last;
  }

  return std::string(in.substr(first, last - first));
}

int parse_int(const std::string& token, const fs::path& file, std::size_t line_number, const char* field_name) {
  try {
    std::size_t consumed = 0;
    const int value = std::stoi(token, &consumed, 10);
    if (consumed != token.size()) {
      fail_parse(file, line_number, std::string("invalid integer in ") + field_name + ": \"" + token + "\"");
    }
    return value;
  } catch (const std::exception&) {
    fail_parse(file, line_number, std::string("invalid integer in ") + field_name + ": \"" + token + "\"");
  }
}

float parse_float(const std::string& token, const fs::path& file, std::size_t line_number, const char* field_name) {
  try {
    std::size_t consumed = 0;
    const float value = std::stof(token, &consumed);
    if (consumed != token.size()) {
      fail_parse(file, line_number, std::string("invalid float in ") + field_name + ": \"" + token + "\"");
    }
    return value;
  } catch (const std::exception&) {
    fail_parse(file, line_number, std::string("invalid float in ") + field_name + ": \"" + token + "\"");
  }
}

struct ObjIndex {
  int v = 0;
  int vt = 0;
  int vn = 0;
};

ObjIndex parse_obj_index(const std::string& token, const fs::path& file, std::size_t line_number) {
  ObjIndex idx{};

  const std::size_t first_slash = token.find('/');
  if (first_slash == std::string::npos) {
    idx.v = parse_int(token, file, line_number, "face vertex index");
    return idx;
  }

  const std::string v_token = token.substr(0, first_slash);
  if (v_token.empty()) {
    fail_parse(file, line_number, "face token is missing position index: \"" + token + "\"");
  }
  idx.v = parse_int(v_token, file, line_number, "face vertex index");

  const std::size_t second_slash = token.find('/', first_slash + 1);
  if (second_slash == std::string::npos) {
    const std::string vt_token = token.substr(first_slash + 1);
    if (!vt_token.empty()) {
      idx.vt = parse_int(vt_token, file, line_number, "face uv index");
    }
    return idx;
  }

  const std::string vt_token = token.substr(first_slash + 1, second_slash - first_slash - 1);
  const std::string vn_token = token.substr(second_slash + 1);
  if (!vt_token.empty()) {
    idx.vt = parse_int(vt_token, file, line_number, "face uv index");
  }
  if (!vn_token.empty()) {
    idx.vn = parse_int(vn_token, file, line_number, "face normal index");
  }
  return idx;
}

int resolve_obj_index(
    int idx,
    std::size_t count,
    const fs::path& file,
    std::size_t line_number,
    const char* field_name) {
  if (idx == 0) {
    fail_parse(file, line_number, std::string(field_name) + " cannot be 0");
  }

  const long long resolved = idx > 0 ? static_cast<long long>(idx - 1) : static_cast<long long>(count) + idx;
  if (resolved < 0 || resolved >= static_cast<long long>(count)) {
    std::ostringstream oss;
    oss << field_name << " out of range: " << idx << " (count=" << count << ")";
    fail_parse(file, line_number, oss.str());
  }
  return static_cast<int>(resolved);
}

std::string sanitize_identifier(const std::string& value) {
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

std::string make_obj_enum_name(const fs::path& input_root, const fs::path& obj_path) {
  fs::path rel = obj_path.lexically_relative(input_root);
  if (rel.empty()) {
    rel = obj_path.filename();
  }
  rel.replace_extension();
  return sanitize_identifier(rel.generic_string());
}

std::vector<fs::path> discover_obj_files(const fs::path& input_root) {
  std::vector<fs::path> obj_files;
  if (!fs::exists(input_root)) {
    return obj_files;
  }

  for (const fs::directory_entry& entry : fs::recursive_directory_iterator(input_root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path path = entry.path();
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](const unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });

    if (ext == ".obj") {
      obj_files.push_back(path);
    }
  }

  std::sort(obj_files.begin(), obj_files.end(), [](const fs::path& a, const fs::path& b) {
    return a.generic_string() < b.generic_string();
  });
  return obj_files;
}

ParsedMesh parse_obj(const fs::path& obj_path) {
  std::ifstream input(obj_path);
  if (!input.is_open()) {
    fail("could not open OBJ file: " + obj_path.generic_string());
  }

  std::vector<std::array<float, 3>> positions;
  std::vector<std::array<float, 2>> uvs;
  std::vector<std::array<float, 3>> normals;
  ParsedMesh mesh;

  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const std::string trimmed = trim(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }

    std::istringstream iss(trimmed);
    std::string op;
    iss >> op;

    if (op == "v") {
      std::string sx;
      std::string sy;
      std::string sz;
      if (!(iss >> sx >> sy >> sz)) {
        fail_parse(obj_path, line_number, "vertex position requires 3 floats");
      }
      positions.push_back({
          parse_float(sx, obj_path, line_number, "vertex x"),
          parse_float(sy, obj_path, line_number, "vertex y"),
          parse_float(sz, obj_path, line_number, "vertex z"),
      });
      continue;
    }

    if (op == "vt") {
      std::string su;
      std::string sv;
      if (!(iss >> su >> sv)) {
        fail_parse(obj_path, line_number, "vertex UV requires at least 2 floats");
      }
      uvs.push_back({
          parse_float(su, obj_path, line_number, "uv u"),
          parse_float(sv, obj_path, line_number, "uv v"),
      });
      continue;
    }

    if (op == "vn") {
      std::string sx;
      std::string sy;
      std::string sz;
      if (!(iss >> sx >> sy >> sz)) {
        fail_parse(obj_path, line_number, "vertex normal requires 3 floats");
      }
      normals.push_back({
          parse_float(sx, obj_path, line_number, "normal x"),
          parse_float(sy, obj_path, line_number, "normal y"),
          parse_float(sz, obj_path, line_number, "normal z"),
      });
      continue;
    }

    if (op == "f") {
      std::vector<std::string> tokens;
      for (std::string token; iss >> token;) {
        tokens.push_back(token);
      }

      if (tokens.size() < 3) {
        fail_parse(obj_path, line_number, "face requires at least 3 vertices");
      }

      std::vector<std::uint32_t> face_indices;
      face_indices.reserve(tokens.size());

      for (const std::string& token : tokens) {
        const ObjIndex obj_index = parse_obj_index(token, obj_path, line_number);

        const int pos_idx =
            resolve_obj_index(obj_index.v, positions.size(), obj_path, line_number, "position index");
        Vertex vertex{};
        vertex.px = positions[static_cast<std::size_t>(pos_idx)][0];
        vertex.py = positions[static_cast<std::size_t>(pos_idx)][1];
        vertex.pz = positions[static_cast<std::size_t>(pos_idx)][2];

        if (obj_index.vt != 0) {
          const int uv_idx = resolve_obj_index(obj_index.vt, uvs.size(), obj_path, line_number, "uv index");
          vertex.u = uvs[static_cast<std::size_t>(uv_idx)][0];
          vertex.v = uvs[static_cast<std::size_t>(uv_idx)][1];
        }

        if (obj_index.vn != 0) {
          const int normal_idx =
              resolve_obj_index(obj_index.vn, normals.size(), obj_path, line_number, "normal index");
          vertex.nx = normals[static_cast<std::size_t>(normal_idx)][0];
          vertex.ny = normals[static_cast<std::size_t>(normal_idx)][1];
          vertex.nz = normals[static_cast<std::size_t>(normal_idx)][2];
        }

        mesh.vertices.push_back(vertex);
        face_indices.push_back(static_cast<std::uint32_t>(mesh.vertices.size() - 1));
      }

      for (std::size_t i = 1; i + 1 < face_indices.size(); ++i) {
        mesh.indices.push_back(face_indices[0]);
        mesh.indices.push_back(face_indices[i]);
        mesh.indices.push_back(face_indices[i + 1]);
      }
      continue;
    }
  }

  if (mesh.vertices.empty() || mesh.indices.empty()) {
    fail("OBJ file has no usable geometry: " + obj_path.generic_string());
  }

  return mesh;
}

void append_u32_le(std::vector<std::uint8_t>* bytes, const std::uint32_t value) {
  bytes->push_back(static_cast<std::uint8_t>(value & 0xFFU));
  bytes->push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
  bytes->push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
  bytes->push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

void append_f32_le(std::vector<std::uint8_t>* bytes, const float value) {
  append_u32_le(bytes, std::bit_cast<std::uint32_t>(value));
}

std::vector<std::uint8_t> encode_binary(const ParsedMesh& mesh) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(8 + mesh.vertices.size() * 44 + mesh.indices.size() * 4);

  append_u32_le(&bytes, static_cast<std::uint32_t>(mesh.vertices.size()));
  for (const Vertex& v : mesh.vertices) {
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

ParsedMesh make_cube_mesh() {
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

  return mesh;
}

std::string make_byte_initializer(const std::vector<std::uint8_t>& bytes) {
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

void write_if_changed(const fs::path& output_path, const std::string& content) {
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

std::string emit_header(const std::string& ns, const std::vector<ModelBlob>& models) {
  std::ostringstream hpp;
  hpp << "#pragma once\n\n";
  hpp << "#include <cstddef>\n";
  hpp << "#include <cstdint>\n";
  hpp << "#include <vector>\n\n";
  hpp << "namespace " << ns << " {\n\n";
  hpp << "enum class ModelId : std::uint32_t {\n";
  for (std::size_t i = 0; i < models.size(); ++i) {
    hpp << "  " << models[i].enum_name << " = " << i << ",\n";
  }
  hpp << "};\n\n";
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
  hpp << "struct ModelAsset {\n";
  hpp << "  ModelId id = ModelId::" << models.front().enum_name << ";\n";
  hpp << "  std::vector<Vertex> vertices;\n";
  hpp << "  std::vector<std::uint32_t> indices;\n";
  hpp << "};\n\n";
  hpp << "[[nodiscard]] const std::byte* get_model_data(ModelId id, std::size_t* out_size);\n";
  hpp << "[[nodiscard]] ModelAsset load_model(ModelId id);\n";
  hpp << "[[nodiscard]] const char* model_name(ModelId id);\n";
  hpp << "[[nodiscard]] const ModelId* all_model_ids(std::size_t* out_count);\n\n";
  hpp << "}  // namespace " << ns << "\n";
  return hpp.str();
}

std::string emit_cpp(const std::string& ns, const std::vector<ModelBlob>& models) {
  std::ostringstream cpp;
  cpp << "#include \"models_generated.hpp\"\n\n";
  cpp << "#include <array>\n";
  cpp << "#include <cstring>\n";
  cpp << "#include <stdexcept>\n";
  cpp << "#include <string>\n\n";
  cpp << "namespace " << ns << " {\n\n";
  cpp << "namespace {\n\n";
  cpp << "struct ModelRecord {\n";
  cpp << "  ModelId id;\n";
  cpp << "  const char* name;\n";
  cpp << "  const std::uint8_t* data;\n";
  cpp << "  std::size_t size;\n";
  cpp << "};\n\n";

  for (const ModelBlob& model : models) {
    cpp << "alignas(4) static constexpr std::uint8_t kModelData_" << model.enum_name << "[] = {";
    cpp << make_byte_initializer(model.bytes);
    cpp << "};\n\n";
  }

  cpp << "static constexpr std::array<ModelRecord, " << models.size() << "> kModels = {{\n";
  for (const ModelBlob& model : models) {
    cpp << "    {ModelId::" << model.enum_name << ", \"" << model.display_name << "\", "
        << "kModelData_" << model.enum_name << ", sizeof(kModelData_" << model.enum_name << ")},\n";
  }
  cpp << "}};\n\n";

  cpp << "static constexpr std::array<ModelId, " << models.size() << "> kModelIds = {{\n";
  for (const ModelBlob& model : models) {
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

  cpp << "std::uint32_t read_u32(const std::uint8_t* data, std::size_t size, std::size_t* offset, "
         "ModelId id, const char* field) {\n";
  cpp << "  if (*offset + 4 > size) {\n";
  cpp << "    throw std::runtime_error(std::string(\"model \") + model_name(id) + \": truncated while reading \" + field);\n";
  cpp << "  }\n";
  cpp << "  const std::uint32_t value =\n";
  cpp << "      static_cast<std::uint32_t>(data[*offset]) |\n";
  cpp << "      (static_cast<std::uint32_t>(data[*offset + 1]) << 8U) |\n";
  cpp << "      (static_cast<std::uint32_t>(data[*offset + 2]) << 16U) |\n";
  cpp << "      (static_cast<std::uint32_t>(data[*offset + 3]) << 24U);\n";
  cpp << "  *offset += 4;\n";
  cpp << "  return value;\n";
  cpp << "}\n\n";

  cpp << "float read_f32(const std::uint8_t* data, std::size_t size, std::size_t* offset, ModelId id, const char* field) {\n";
  cpp << "  const std::uint32_t bits = read_u32(data, size, offset, id, field);\n";
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
  cpp << "    throw std::runtime_error(\"unknown ModelId\");\n";
  cpp << "  }\n";
  cpp << "  if (out_size != nullptr) {\n";
  cpp << "    *out_size = record->size;\n";
  cpp << "  }\n";
  cpp << "  return reinterpret_cast<const std::byte*>(record->data);\n";
  cpp << "}\n\n";

  cpp << "ModelAsset load_model(ModelId id) {\n";
  cpp << "  const ModelRecord* record = find_record(id);\n";
  cpp << "  if (record == nullptr) {\n";
  cpp << "    throw std::runtime_error(\"unknown ModelId\");\n";
  cpp << "  }\n\n";
  cpp << "  const std::uint8_t* data = record->data;\n";
  cpp << "  const std::size_t size = record->size;\n";
  cpp << "  std::size_t offset = 0;\n\n";
  cpp << "  ModelAsset model;\n";
  cpp << "  model.id = id;\n\n";
  cpp << "  const std::uint32_t vertex_count = read_u32(data, size, &offset, id, \"vertex_count\");\n";
  cpp << "  model.vertices.reserve(vertex_count);\n";
  cpp << "  for (std::uint32_t i = 0; i < vertex_count; ++i) {\n";
  cpp << "    Vertex v;\n";
  cpp << "    v.px = read_f32(data, size, &offset, id, \"px\");\n";
  cpp << "    v.py = read_f32(data, size, &offset, id, \"py\");\n";
  cpp << "    v.pz = read_f32(data, size, &offset, id, \"pz\");\n";
  cpp << "    v.cx = read_f32(data, size, &offset, id, \"cx\");\n";
  cpp << "    v.cy = read_f32(data, size, &offset, id, \"cy\");\n";
  cpp << "    v.cz = read_f32(data, size, &offset, id, \"cz\");\n";
  cpp << "    v.nx = read_f32(data, size, &offset, id, \"nx\");\n";
  cpp << "    v.ny = read_f32(data, size, &offset, id, \"ny\");\n";
  cpp << "    v.nz = read_f32(data, size, &offset, id, \"nz\");\n";
  cpp << "    v.u = read_f32(data, size, &offset, id, \"u\");\n";
  cpp << "    v.v = read_f32(data, size, &offset, id, \"v\");\n";
  cpp << "    model.vertices.push_back(v);\n";
  cpp << "  }\n\n";
  cpp << "  const std::uint32_t index_count = read_u32(data, size, &offset, id, \"index_count\");\n";
  cpp << "  model.indices.reserve(index_count);\n";
  cpp << "  for (std::uint32_t i = 0; i < index_count; ++i) {\n";
  cpp << "    model.indices.push_back(read_u32(data, size, &offset, id, \"index\"));\n";
  cpp << "  }\n\n";
  cpp << "  if (offset != size) {\n";
  cpp << "    throw std::runtime_error(std::string(\"model \") + model_name(id) + \": trailing bytes in asset blob\");\n";
  cpp << "  }\n";
  cpp << "  return model;\n";
  cpp << "}\n\n";

  cpp << "}  // namespace " << ns << "\n";
  return cpp.str();
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto read_value = [&](const char* flag) -> std::string {
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
    } else {
      fail("unknown argument: " + std::string(arg));
    }
  }

  if (opts.input_root.empty() || opts.output_dir.empty() || opts.emit_hpp.empty() || opts.emit_cpp.empty()) {
    fail("usage: model_codegen --input-root <dir> --output-dir <dir> --emit-hpp <file> --emit-cpp <file> "
         "[--namespace <ns>]");
  }

  return opts;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions opts = parse_cli(argc, argv);

    std::error_code ec;
    fs::create_directories(opts.output_dir, ec);
    if (ec) {
      fail("failed to create output dir: " + opts.output_dir.generic_string() + " (" + ec.message() + ")");
    }

    std::vector<ModelBlob> models;
    models.push_back(ModelBlob{
        .enum_name = "CUBE",
        .display_name = "cube",
        .bytes = encode_binary(make_cube_mesh()),
    });

    const std::vector<fs::path> obj_files = discover_obj_files(opts.input_root);
    std::unordered_map<std::string, std::size_t> collision_count;

    for (const fs::path& obj_path : obj_files) {
      ParsedMesh mesh;
      try {
        mesh = parse_obj(obj_path);
      } catch (const std::exception& ex) {
        std::ostringstream oss;
        oss << "while processing " << obj_path.generic_string() << ": " << ex.what();
        throw std::runtime_error(oss.str());
      }

      std::string enum_name = make_obj_enum_name(opts.input_root, obj_path);
      std::size_t& count = collision_count[enum_name];
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
          .bytes = encode_binary(mesh),
      });
    }

    const std::string hpp = emit_header(opts.ns, models);
    const std::string cpp = emit_cpp(opts.ns, models);
    write_if_changed(opts.emit_hpp, hpp);
    write_if_changed(opts.emit_cpp, cpp);

    std::cerr << "model_codegen: generated " << models.size() << " model asset(s) into "
              << opts.output_dir.generic_string() << '\n';
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "model_codegen: error: " << ex.what() << '\n';
    return 1;
  }
}

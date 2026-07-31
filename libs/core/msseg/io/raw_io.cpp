#include "msseg/io/raw_io.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace msseg {
namespace {

std::filesystem::path sidecar_path(const std::filesystem::path& path) {
  std::filesystem::path p = path;
  p += ".dat";
  return p;
}

Dims3 read_sidecar(const std::filesystem::path& path) {
  std::ifstream in(sidecar_path(path));
  if (!in.good()) {
    throw std::runtime_error("Missing dimension sidecar: " + sidecar_path(path).string());
  }
  Dims3 dims;
  in >> dims.width >> dims.height >> dims.depth;
  if (!in) {
    throw std::runtime_error("Malformed dimension sidecar: " + sidecar_path(path).string());
  }
  if (dims.depth == 0) dims.depth = 1;
  return dims;
}

void write_sidecar(const std::filesystem::path& path, const diffg::Dimensions& dims) {
  std::ofstream out(sidecar_path(path));
  if (!out.good()) {
    throw std::runtime_error("Failed to write dimension sidecar: " + sidecar_path(path).string());
  }
  out << dims.width << " " << dims.height << " " << dims.depth << "\n";
}

}  // namespace

Volume read_raw_volume(const std::filesystem::path& path, const Dims3& dims) {
  const std::size_t count = dims.width * dims.height * dims.depth;
  Volume volume(diffg::Dimensions{dims.width, dims.height, dims.depth});
  if (volume.size() != count) {
    throw std::runtime_error("Volume allocation size mismatch for: " + path.string());
  }

  std::ifstream in(path, std::ios::binary);
  if (!in.good()) {
    throw std::runtime_error("Failed to open raw volume: " + path.string());
  }
  in.read(reinterpret_cast<char*>(volume.data()), static_cast<std::streamsize>(count * sizeof(float)));
  if (static_cast<std::size_t>(in.gcount()) != count * sizeof(float)) {
    throw std::runtime_error("Raw volume too short (expected " + std::to_string(count) + " floats): " + path.string());
  }
  return volume;
}

Volume read_raw_volume(const std::filesystem::path& path) {
  return read_raw_volume(path, read_sidecar(path));
}

void write_raw_volume(const std::filesystem::path& path, const Volume& volume) {
  std::ofstream out(path, std::ios::binary);
  if (!out.good()) {
    throw std::runtime_error("Failed to write raw volume: " + path.string());
  }
  out.write(reinterpret_cast<const char*>(volume.data()),
            static_cast<std::streamsize>(volume.size() * sizeof(float)));
  write_sidecar(path, volume.dims());
}

void write_raw_labels(const std::filesystem::path& path, const LabelVolume& labels) {
  std::ofstream out(path, std::ios::binary);
  if (!out.good()) {
    throw std::runtime_error("Failed to write raw labels: " + path.string());
  }
  out.write(reinterpret_cast<const char*>(labels.data()),
            static_cast<std::streamsize>(labels.size() * sizeof(std::int32_t)));
  write_sidecar(path, labels.dims());
}

}  // namespace msseg

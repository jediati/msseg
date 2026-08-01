#include "msseg/io/tiff_io.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include("tinytiffreader.h")
#include "tinytiffreader.h"
#include "tinytiffwriter.h"
#elif __has_include("TinyTIFF/tinytiffreader.h")
#include "TinyTIFF/tinytiffreader.h"
#include "TinyTIFF/tinytiffwriter.h"
#else
#error "TinyTIFF headers not found."
#endif

namespace msseg {
namespace {

// TinyTIFF / TIFF SampleFormat tag values.
constexpr std::uint16_t kTiffSampleFormatUInt = 1;
constexpr std::uint16_t kTiffSampleFormatInt = 2;
constexpr std::uint16_t kTiffSampleFormatFloat = 3;

// Read the single sample plane as its native integer type T and convert to
// float. Returns false if TinyTIFF failed to decode the plane.
template <typename T>
bool read_plane_as_float(TinyTIFFReaderFile* reader, float* out, std::size_t count) {
  std::vector<T> buffer(count);
  if (!TinyTIFFReader_getSampleData(reader, buffer.data(), 0)) return false;
  for (std::size_t i = 0; i < count; ++i) out[i] = static_cast<float>(buffer[i]);
  return true;
}

}  // namespace

diffg::Image<float> read_tiff_float32(const std::filesystem::path& path) {
  TinyTIFFReaderFile* reader = TinyTIFFReader_open(path.string().c_str());
  if (reader == nullptr) {
    throw std::runtime_error("Failed to open TIFF: " + path.string());
  }

  const auto close_reader = [&]() { TinyTIFFReader_close(reader); };

  if (TinyTIFFReader_wasError(reader)) {
    const std::string err = TinyTIFFReader_getLastError(reader);
    close_reader();
    throw std::runtime_error("TinyTIFFReader error: " + err);
  }

  const std::uint32_t width = TinyTIFFReader_getWidth(reader);
  const std::uint32_t height = TinyTIFFReader_getHeight(reader);
  const std::uint16_t samples = TinyTIFFReader_getSamplesPerPixel(reader);
  const std::uint16_t bits = TinyTIFFReader_getBitsPerSample(reader, 0);
  const std::uint16_t format = TinyTIFFReader_getSampleFormat(reader);

  if (samples != 1) {
    close_reader();
    throw std::runtime_error("Input TIFF must be single-sample (grayscale).");
  }

  diffg::Image<float> image(diffg::Dimensions{static_cast<std::size_t>(width), static_cast<std::size_t>(height), 1});
  const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

  // A missing SampleFormat tag defaults to unsigned integer per the TIFF spec;
  // TinyTIFF reports 0 in that case. Everything downstream runs on float32, so
  // integer samples are widened here.
  const std::uint16_t fmt = (format == 0) ? kTiffSampleFormatUInt : format;

  bool ok = false;
  if (fmt == kTiffSampleFormatFloat && bits == 32) {
    ok = TinyTIFFReader_getSampleData(reader, image.data(), 0);
  } else if (fmt == kTiffSampleFormatUInt && bits == 8) {
    ok = read_plane_as_float<std::uint8_t>(reader, image.data(), count);
  } else if (fmt == kTiffSampleFormatUInt && bits == 16) {
    ok = read_plane_as_float<std::uint16_t>(reader, image.data(), count);
  } else if (fmt == kTiffSampleFormatUInt && bits == 32) {
    ok = read_plane_as_float<std::uint32_t>(reader, image.data(), count);
  } else if (fmt == kTiffSampleFormatInt && bits == 8) {
    ok = read_plane_as_float<std::int8_t>(reader, image.data(), count);
  } else if (fmt == kTiffSampleFormatInt && bits == 16) {
    ok = read_plane_as_float<std::int16_t>(reader, image.data(), count);
  } else if (fmt == kTiffSampleFormatInt && bits == 32) {
    ok = read_plane_as_float<std::int32_t>(reader, image.data(), count);
  } else {
    close_reader();
    throw std::runtime_error("Unsupported TIFF sample layout (bits=" + std::to_string(bits) + ", format=" +
                             std::to_string(format) +
                             "); expected single-sample 8/16/32-bit integer or 32-bit float.");
  }

  if (!ok) {
    const std::string err = TinyTIFFReader_getLastError(reader);
    close_reader();
    throw std::runtime_error("TinyTIFFReader_getSampleData failed: " + err);
  }

  close_reader();
  return image;
}

void write_tiff_float32(const std::filesystem::path& path, int width, int height, const float* data) {
  TinyTIFFWriterFile* writer =
      TinyTIFFWriter_open(path.string().c_str(), 32, TinyTIFFWriter_Float, 1, static_cast<std::uint32_t>(width),
                          static_cast<std::uint32_t>(height), TinyTIFFWriter_Greyscale);
  if (writer == nullptr) {
    throw std::runtime_error("Failed to open output float32 TIFF writer: " + path.string());
  }
  if (!TinyTIFFWriter_writeImage(writer, const_cast<float*>(data))) {
    const std::string err = TinyTIFFWriter_getLastError(writer);
    TinyTIFFWriter_close(writer);
    throw std::runtime_error("TinyTIFFWriter_writeImage float32 failed: " + err);
  }
  TinyTIFFWriter_close(writer);
}

void write_tiff_mask_u8(const std::filesystem::path& path, int width, int height, const std::uint8_t* data) {
  TinyTIFFWriterFile* writer =
      TinyTIFFWriter_open(path.string().c_str(), 8, TinyTIFFWriter_UInt, 1, static_cast<std::uint32_t>(width),
                          static_cast<std::uint32_t>(height), TinyTIFFWriter_Greyscale);
  if (writer == nullptr) {
    throw std::runtime_error("Failed to open output TIFF writer: " + path.string());
  }
  if (!TinyTIFFWriter_writeImage(writer, const_cast<std::uint8_t*>(data))) {
    const std::string err = TinyTIFFWriter_getLastError(writer);
    TinyTIFFWriter_close(writer);
    throw std::runtime_error("TinyTIFFWriter_writeImage failed: " + err);
  }
  TinyTIFFWriter_close(writer);
}

void write_tiff_int32(const std::filesystem::path& path, int width, int height, const std::int32_t* data) {
  TinyTIFFWriterFile* writer =
      TinyTIFFWriter_open(path.string().c_str(), 32, TinyTIFFWriter_Int, 1, static_cast<std::uint32_t>(width),
                          static_cast<std::uint32_t>(height), TinyTIFFWriter_Greyscale);
  if (writer == nullptr) {
    throw std::runtime_error("Failed to open output int32 TIFF writer: " + path.string());
  }
  if (!TinyTIFFWriter_writeImage(writer, const_cast<std::int32_t*>(data))) {
    const std::string err = TinyTIFFWriter_getLastError(writer);
    TinyTIFFWriter_close(writer);
    throw std::runtime_error("TinyTIFFWriter_writeImage int32 failed: " + err);
  }
  TinyTIFFWriter_close(writer);
}

}  // namespace msseg

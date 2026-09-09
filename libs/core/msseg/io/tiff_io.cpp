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

// Read sample plane `sample` as its native integer type T and convert to
// float. Returns false if TinyTIFF failed to decode the plane.
template <typename T>
bool read_plane_as_float(TinyTIFFReaderFile* reader, std::uint16_t sample, float* out, std::size_t count) {
  std::vector<T> buffer(count);
  if (!TinyTIFFReader_getSampleData(reader, buffer.data(), sample)) return false;
  for (std::size_t i = 0; i < count; ++i) out[i] = static_cast<float>(buffer[i]);
  return true;
}

// Decode one sample plane into `out` as float32. Returns false when TinyTIFF
// failed; throws for a sample layout this reader does not handle.
bool read_sample_plane(TinyTIFFReaderFile* reader, std::uint16_t fmt, std::uint16_t bits, std::uint16_t sample,
                       float* out, std::size_t count) {
  if (fmt == kTiffSampleFormatFloat && bits == 32) return TinyTIFFReader_getSampleData(reader, out, sample) != 0;
  if (fmt == kTiffSampleFormatUInt && bits == 8) return read_plane_as_float<std::uint8_t>(reader, sample, out, count);
  if (fmt == kTiffSampleFormatUInt && bits == 16) return read_plane_as_float<std::uint16_t>(reader, sample, out, count);
  if (fmt == kTiffSampleFormatUInt && bits == 32) return read_plane_as_float<std::uint32_t>(reader, sample, out, count);
  if (fmt == kTiffSampleFormatInt && bits == 8) return read_plane_as_float<std::int8_t>(reader, sample, out, count);
  if (fmt == kTiffSampleFormatInt && bits == 16) return read_plane_as_float<std::int16_t>(reader, sample, out, count);
  if (fmt == kTiffSampleFormatInt && bits == 32) return read_plane_as_float<std::int32_t>(reader, sample, out, count);
  throw std::runtime_error("Unsupported TIFF sample layout (bits=" + std::to_string(bits) + ", format=" +
                           std::to_string(fmt) + "); expected 8/16/32-bit integer or 32-bit float samples.");
}

struct ReaderGuard {
  TinyTIFFReaderFile* reader = nullptr;
  ~ReaderGuard() {
    if (reader != nullptr) TinyTIFFReader_close(reader);
  }
};

// The shared reader. `require_single` reproduces read_tiff_float32's historical
// contract (and error text) exactly; otherwise every sample becomes a plane.
InputSlice read_planes_impl(const std::filesystem::path& path, const ColorInputPolicy& policy,
                            bool require_single) {
  ReaderGuard guard;
  guard.reader = TinyTIFFReader_open(path.string().c_str());
  TinyTIFFReaderFile* reader = guard.reader;
  if (reader == nullptr) {
    throw std::runtime_error("Failed to open TIFF: " + path.string());
  }
  if (TinyTIFFReader_wasError(reader)) {
    const std::string err = TinyTIFFReader_getLastError(reader);
    throw std::runtime_error("TinyTIFFReader error: " + err);
  }

  const std::uint32_t width = TinyTIFFReader_getWidth(reader);
  const std::uint32_t height = TinyTIFFReader_getHeight(reader);
  const std::uint16_t samples = TinyTIFFReader_getSamplesPerPixel(reader);
  const std::uint16_t bits = TinyTIFFReader_getBitsPerSample(reader, 0);
  const std::uint16_t format = TinyTIFFReader_getSampleFormat(reader);

  if (require_single && samples != 1) {
    throw std::runtime_error("Input TIFF must be single-sample (grayscale).");
  }
  if (samples == 0) {
    throw std::runtime_error("Input TIFF reports zero samples per pixel: " + path.string());
  }

  // Alpha is inferred from the count: 4 samples read as RGBA, 2 as gray+alpha.
  std::uint16_t keep = samples;
  bool alpha_dropped = false;
  if (policy.alpha == ColorInputPolicy::Alpha::Drop && (samples == 4 || samples == 2)) {
    keep = static_cast<std::uint16_t>(samples - 1);
    alpha_dropped = true;
  }

  InputSlice slice;
  slice.planes = diffg::MultiImage<float>(
      diffg::Dimensions{static_cast<std::size_t>(width), static_cast<std::size_t>(height), 1}, keep);
  slice.file_samples = samples;
  slice.alpha_dropped = alpha_dropped;
  const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

  // A missing SampleFormat tag defaults to unsigned integer per the TIFF spec;
  // TinyTIFF reports 0 in that case. Everything downstream runs on float32, so
  // integer samples are widened here.
  const std::uint16_t fmt = (format == 0) ? kTiffSampleFormatUInt : format;

  for (std::uint16_t s = 0; s < keep; ++s) {
    if (!read_sample_plane(reader, fmt, bits, s, slice.planes.channel_data(s), count)) {
      const std::string err = TinyTIFFReader_getLastError(reader);
      throw std::runtime_error("TinyTIFFReader_getSampleData failed (sample " + std::to_string(s) + "): " + err);
    }
  }
  return slice;
}

}  // namespace

diffg::Image<float> read_tiff_float32(const std::filesystem::path& path) {
  return read_planes_impl(path, ColorInputPolicy{}, /*require_single=*/true).scalar();
}

InputSlice read_tiff_planes(const std::filesystem::path& path, const ColorInputPolicy& policy) {
  return read_planes_impl(path, policy, /*require_single=*/false);
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

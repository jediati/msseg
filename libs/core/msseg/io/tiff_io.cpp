#include "msseg/io/tiff_io.hpp"

#include <stdexcept>
#include <string>

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

constexpr std::uint16_t kTiffSampleFormatFloat = 3;

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

  if (samples != 1 || bits != 32 || format != kTiffSampleFormatFloat) {
    close_reader();
    throw std::runtime_error("Input TIFF must be single-sample float32.");
  }

  diffg::Image<float> image(diffg::Dimensions{static_cast<std::size_t>(width), static_cast<std::size_t>(height), 1});

  if (!TinyTIFFReader_getSampleData(reader, image.data(), 0)) {
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

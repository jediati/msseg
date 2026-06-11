#include "mscoupon/io.hpp"

#include <fstream>
#include <stdexcept>

#if __has_include("tinytiffreader.h")
#include "tinytiffreader.h"
#include "tinytiffwriter.h"
#elif __has_include("TinyTIFF/tinytiffreader.h")
#include "TinyTIFF/tinytiffreader.h"
#include "TinyTIFF/tinytiffwriter.h"
#else
#error "TinyTIFF headers not found."
#endif

namespace mscoupon {
namespace {

constexpr uint16_t kTiffSampleFormatFloat = 3;

}  // namespace

Image2D read_tiff_float32(const std::filesystem::path& path) {
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

  const uint32_t width = TinyTIFFReader_getWidth(reader);
  const uint32_t height = TinyTIFFReader_getHeight(reader);
  const uint16_t samples = TinyTIFFReader_getSamplesPerPixel(reader);
  const uint16_t bits = TinyTIFFReader_getBitsPerSample(reader, 0);
  const uint16_t format = TinyTIFFReader_getSampleFormat(reader);

  if (samples != 1 || bits != 32 || format != kTiffSampleFormatFloat) {
    close_reader();
    throw std::runtime_error("Input TIFF must be single-sample float32.");
  }

  Image2D image;
  image.width = static_cast<int>(width);
  image.height = static_cast<int>(height);
  image.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

  if (!TinyTIFFReader_getSampleData(reader, image.pixels.data(), 0)) {
    const std::string err = TinyTIFFReader_getLastError(reader);
    close_reader();
    throw std::runtime_error("TinyTIFFReader_getSampleData failed: " + err);
  }

  close_reader();
  return image;
}

void write_tiff_mask_u8(const std::filesystem::path& path, const Mask2D& mask) {
  TinyTIFFWriterFile* writer = TinyTIFFWriter_open(path.string().c_str(), 8, TinyTIFFWriter_UInt, 1,
                                                   static_cast<uint32_t>(mask.width), static_cast<uint32_t>(mask.height),
                                                   TinyTIFFWriter_Greyscale);
  if (writer == nullptr) {
    throw std::runtime_error("Failed to open output TIFF writer: " + path.string());
  }

  if (!TinyTIFFWriter_writeImage(writer, mask.pixels.data())) {
    const std::string err = TinyTIFFWriter_getLastError(writer);
    TinyTIFFWriter_close(writer);
    throw std::runtime_error("TinyTIFFWriter_writeImage failed: " + err);
  }

  TinyTIFFWriter_close(writer);
}

void write_tiff_float32(const std::filesystem::path& path, const Image2D& image) {
  TinyTIFFWriterFile* writer = TinyTIFFWriter_open(path.string().c_str(), 32, TinyTIFFWriter_Float, 1,
                                                   static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height),
                                                   TinyTIFFWriter_Greyscale);
  if (writer == nullptr) {
    throw std::runtime_error("Failed to open output float32 TIFF writer: " + path.string());
  }
  if (!TinyTIFFWriter_writeImage(writer, image.pixels.data())) {
    const std::string err = TinyTIFFWriter_getLastError(writer);
    TinyTIFFWriter_close(writer);
    throw std::runtime_error("TinyTIFFWriter_writeImage float32 failed: " + err);
  }
  TinyTIFFWriter_close(writer);
}

void write_tiff_int32(const std::filesystem::path& path, int width, int height, const std::vector<int>& labels) {
  TinyTIFFWriterFile* writer = TinyTIFFWriter_open(path.string().c_str(), 32, TinyTIFFWriter_Int, 1,
                                                   static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                                                   TinyTIFFWriter_Greyscale);
  if (writer == nullptr) {
    throw std::runtime_error("Failed to open output int32 TIFF writer: " + path.string());
  }
  if (!TinyTIFFWriter_writeImage(writer, labels.data())) {
    const std::string err = TinyTIFFWriter_getLastError(writer);
    TinyTIFFWriter_close(writer);
    throw std::runtime_error("TinyTIFFWriter_writeImage int32 failed: " + err);
  }
  TinyTIFFWriter_close(writer);
}

void write_segment_table_csv(const std::filesystem::path& path, const std::vector<SegmentStat>& stats) {
  std::ofstream out(path);
  if (!out.good()) {
    throw std::runtime_error("Failed to write table: " + path.string());
  }
  out << "segment_id,slice_index,area,min_value,max_value,mean_value,min_x,min_y,max_x,max_y\n";
  for (const auto& row : stats) {
    out << row.segment_id << "," << row.slice_index << "," << row.area << "," << row.min_value << "," << row.max_value << ","
        << row.mean_value << "," << row.min_x << "," << row.min_y << "," << row.max_x << "," << row.max_y << "\n";
  }
}

}  // namespace mscoupon

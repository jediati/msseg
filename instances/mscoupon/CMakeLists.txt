# The mscoupon instance: the original 2D TIFF-slice batch analysis, now a
# verified instance built on msseg_core + msseg_io.
add_msseg_instance(mscoupon
  PYTHON
  NEEDS_IO
  LIB_SOURCES
    lib/mscoupon/mscoupon_cli.cpp
    lib/mscoupon/config.cpp
    lib/mscoupon/sequence.cpp
    lib/mscoupon/io.cpp
    lib/mscoupon/filter.cpp
    lib/mscoupon/msc_stage.cpp
    lib/mscoupon/stats.cpp
    lib/mscoupon/pipeline.cpp
)

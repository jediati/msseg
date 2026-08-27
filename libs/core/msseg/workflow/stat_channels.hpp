#pragma once

#include <string>
#include <vector>

#include "msseg/workflow/params.hpp"

namespace msseg {

// Expansion of a StatsSpec's channel requests into the ordered, named channel
// list every downstream layer indexes by SLOT.
//
// This is deliberately free of diffg: config validation, the CSV header, the
// query schema and the GUI dropdown all need to know the channel names without
// computing anything, and mscoupon's config parser runs long before any raster
// exists. `build_stat_channels` (filter/filter_stage.hpp) is what turns the same
// list into actual pixels.
//
// Ordering is base, filtered, then the derived requests in config order, each
// expanded sigma-major. That ordering is part of the contract: it is the column
// order of the feature table and of the CSVs.
std::vector<ResolvedStatChannel> resolve_stat_channels(const StatsSpec& spec);

// Canonical sigma rendering for a channel name: "%g", so 0.7 -> "0.7" and
// 3.0 -> "3". Exposed because config error messages need to name the channel a
// user would have to type.
std::string format_sigma(double sigma);

// True iff `kind` names a derived (Gaussian-derivative) channel, i.e. anything
// other than the two rasters the pipeline already has.
bool is_derived_channel_kind(const std::string& kind);

// The derived kinds a workflow may name, for error messages.
const std::vector<std::string>& derived_channel_kinds();

// How many channels one request expands to. 2 for `hessian` in 2D (largest and
// smallest eigenvalue), 1 otherwise. Throws for an unknown kind.
int channels_per_sigma(const std::string& kind);

}  // namespace msseg

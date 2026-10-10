#pragma once

// Public Planar CT aggregation header.  The implementation remains in the
// compatibility SourceOrbit header while callers migrate to this namespace.
#include "NonStandardFpBp/SourceOrbitTomography/YkSourceOrbitTomography.hpp"

namespace YK::NonStandardFpBp::PlanarCt {

using GeometryConfig = SourceOrbitGeometryConfig;
using Sampling = SourceTrajectorySampling;

using ::YK::NonStandardFpBp::buildSourceOrbitGeometry;
using ::YK::NonStandardFpBp::buildSourceTrajectoryGeometry;
using ::YK::NonStandardFpBp::buildUniformTrajectoryAngles;
using ::YK::NonStandardFpBp::makeSafeWindow;

} // namespace YK::NonStandardFpBp::PlanarCt

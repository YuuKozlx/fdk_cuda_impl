# Source-orbit tomography

This module contains the non-standard scan in which the detector and object
remain fixed and only the source moves around the Y axis.  The trajectory
angle is therefore a source-position parameter, not a physical gantry angle.

Each view is processed on a fixed-size local detector window:

```text
raw fixed detector -> safe rotated-window resample -> preweight -> U filter
-> backproject with the current source position -> accumulate
```

The standard circular FDK pipeline is intentionally not used as the public
entry point here.  Its validation and weighting code assumes a circular
source/detector trajectory.  The CUDA preweight, filter, and generic
per-view backprojection stages may be reused after the source-orbit scheduler
supplies the correct geometry and window origin for each view.

Two ray-preserving resampling modes are available. `InPlaneRotated` retains the
physical detector plane and rotates the local U/V basis by
`angle[i] - angle[0]`. `PerspectiveOrthogonal` maps directly from the original
detector to a per-view virtual plane whose normal is the source-to-volume-center
ray. The latter uses one projective interpolation, not a crop followed by a
second interpolation. In both modes the same map is applied to projection
samples and `detS/detU/detV`, and bilinear interpolation uses a zero border.

The reference ray used for analytic weighting is the ray from each source
position to the configured reconstruction-volume center. It is not the ray to
the center of the integer crop window; any rounding displacement is retained
as the mapped detector principal-point offset.

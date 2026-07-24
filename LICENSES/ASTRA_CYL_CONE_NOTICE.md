# ASTRA cylindrical detector geometry attribution

The cylindrical detector geometry conventions under `src/CylFpBp` were designed
with reference to ASTRA Toolbox's `cyl_cone_vec` implementation:

- Project: https://github.com/astra-toolbox/astra-toolbox
- Reference revision: `e29e2ad609c62de07a80510e965ec2675a56a027`
- Relevant geometry/kernel files:
  `CylConeVecProjectionGeometry3D.h`, `GeometryUtil3D.cpp`, `cone_cyl.cu`
- Original license for those files: MIT License

The YKCBCT implementation uses its own geometry structure, grid-stride launch
policy, linear-memory projector, and matched trilinear scatter backprojector.
It does not copy ASTRA's CUDA projector kernel.

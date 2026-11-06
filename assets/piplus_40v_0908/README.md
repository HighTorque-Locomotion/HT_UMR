# PiPlus 40V 0908 / 22 DOF

This is the robot asset used by the BONES-SEED SOMA uniform local-region adapter.
Model: `PiPlus_S_12L8A0G2H0W_LSE_ZedMini_40V_260908`.

- 22 scalar joints: 12 leg, 8 arm and 2 head joints; no waist or gripper DOFs.
- A floating base adds 7 configuration coordinates and 6 velocity coordinates:
  MuJoCo `nq=29`, `nv=28`. The floating base is not counted in the 22 robot DOFs.
- `piplus_40v_260908_no_hyperextension.xml` is the production MJCF, including knee
  limits `[0, 2.18]` (right) and `[0, 2.182]` (left), in radians.
- `meshes/` contains all 27 referenced STL meshes. The compiler uses relative
  `meshdir="meshes"`; there are no workstation-specific asset paths.
- `asset_manifest.json` lists the joint names, source model identifier and mesh
  SHA-256 hashes. Packaging changes the MJCF mesh directory and normalizes XML whitespace;
  meshes and physical model parameters are preserved.

The solver settings intentionally narrow some MJCF limits (for example head pitch)
and are applied by `scripts/local_regions/configure.py`. Use the model together
with `robot_configs/local_regions/piplus_40v_0908_settings.json` and the calibrated
local-region profile. Do not substitute the existing `assets/PiPlusPro` model:
it is a different robot variant and its correspondence is not interchangeable.

The source asset package declares BSD in the preserved `source_package.xml`.
Human model assets and BONES-SEED motions remain external inputs under their own terms.
See the [22-DOF instructions](../../README.md#piplus-22dof) and
[local-region guide](../../docs/local_region_umr_zh.md) for the full workflow.

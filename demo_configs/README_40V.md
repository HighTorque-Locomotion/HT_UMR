# PiPlus 40V defaults

For the root cause, diagnostic evidence, and checks required when adapting a new
XML, see [新 XML 适配避坑与验收清单](../docs/new_xml_surface_binding_checklist_zh.md).
For ready-to-run collision commands and foot-specific checks, see
[自碰撞开启与足部避碰速查](../docs/self_collision_quickstart_zh.md).

All `piplus_s_40v` demo configurations now inherit the corrected surface binding
from `umr_demo_piplus_s_40v.json`. The ordinary and full-length entry points no
longer use the old arm binding. Other robot configurations are unchanged.

The correction restricts the nearest-surface projection by source segment:

| Source segment | Robot body |
| --- | --- |
| leftArm / rightArm | l_upper_arm_link / r_upper_arm_link |
| leftForeArm / rightForeArm | l_elbow_link / r_elbow_link |
| leftHand / rightHand | l_wrist_link / r_wrist_link |

Points are projected onto these bodies' actual visual meshes; local positions
and surface normals are recomputed before constructing the transported normal
targets. This prevents arm observations from being attached upstream of the
joint they need to constrain. It is a local robot-specific adaptation of UMR,
not an unchanged upstream configuration.

Original 40V axes, geometry, and joint limits are retained. The configuration
uses 3 iterations per frame, 60 initialization iterations, and the existing
1024-slot / 20-epoch correspondence demo. No GMR or ccrp trajectory is used as
the retargeting input. The `_semantic_arms` configs are compatibility aliases
with separate output locations; they no longer define a separate fix.

The unsuccessful Pro-XML arm-limit experiment configuration has been removed
from the active configs. Historical local data/video/model artifacts are not
deleted. Normal outputs now use `output/piplus_s_40v_fixed_retarget/` so that
legacy 40V results are not silently replaced.

## Full dance, with or without strong self-collision constraints

Run from the repository root, with the licensed SMPL-X model and 40V MJCF
available at the paths configured in the base demo. To change the local robot
path, edit `robot.xml` or use the retargeter's `--robot-xml` override.

```bash
# Build/reuse the correspondence and run the corrected full dance.
python scripts/humanoid_retarget_pipeline.py --config demo_configs/umr_demo_piplus_s_40v_full.json

# Strong collision profile, reusing the existing correspondence.
python scripts/retarget_smpl_to_humanoid_surface_vector.py \
  --config demo_configs/umr_demo_piplus_s_40v_self_collision_strong.json \
  --slots output/demo_piplus_s_40v/correspondence_slots/correspondence_slots_final.npz

MUJOCO_GL=egl python scripts/visualize_robot_retarget_result.py \
  --result output/dance1_40v_fixed_self_collision/dance1_subject2_40v_fixed_strong_self_collision.npz \
  --record-video output/dance1_40v_fixed_self_collision/dance1_subject2_40v_fixed_strong_self_collision.mp4 \
  --record-width 960 --record-height 540 --camera-mode root --camera-distance 1.2

# Checks do not require SMPL-X models, robot meshes, or previous motion outputs.
python scripts/check_surface_slot_body_bindings.py
```

The strong profile enables robot geometry self-penetration constraints, not just
the human self-contact map. Parameters:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| collision_threshold | 0.1 m | Candidate collision search distance |
| robot_self_penetration_cost | 10000 | Soft separation penalty weight |
| robot_self_penetration_tolerance | 0.003 m | Soft target clearance |
| robot_self_penetration_hard_constraint | true | Add signed-distance inequalities |
| robot_self_penetration_margin | 0.002 m | Desired hard-constraint clearance |
| robot_self_penetration_hard_slack | true | Allow violation when constraints conflict |
| robot_self_penetration_hard_slack_cost | 1000000 | Penalize that violation strongly |
| trajectory_filter_mode | off | Avoid post-filtering violating solved constraints |

The default ordinary profile does not enable strong collision constraints;
select this separate profile explicitly. Collision geometry is approximate and
slack is enabled, so this does not guarantee zero mesh penetration or dynamically
safe hardware motion. Inspect full-sequence collision and continuity results
before any real-robot use.

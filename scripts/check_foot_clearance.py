#!/usr/bin/env python3
"""Focused regression checks for sole alignment, swing planning and smoothing."""
import unittest
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from numpy.testing import assert_allclose

import retarget_foot_clearance as foot


class FootClearanceChecks(unittest.TestCase):
    def test_support_patch_detects_toe_contact_and_preserves_yaw(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="foot" pos="0 0 .05"><freejoint/>
            <geom type="box" size=".1 .04 .05"/>
          </body></worldbody></mujoco>''')
        soles = foot.build_robot_soles(model, dict(left="foot", right="foot"))
        foot.build_support_patches(model, soles)
        poses = np.repeat(model.qpos0[None], 3, axis=0)
        # Rotate only around Z: a flat foot stays flat at any heading.
        poses[1, 3:7] = [np.cos(.4), 0, 0, np.sin(.4)]
        # Pitch by 30 degrees and translate up until only the toe touches.
        angle = np.pi / 6
        poses[2, 3:7] = [np.cos(angle/2), 0, np.sin(angle/2), 0]
        poses[2, 2] = .1*np.sin(angle) + .05*np.cos(angle)
        heights, _ = foot.robot_sole_trajectory(model, poses, soles)
        maximum, span = foot.support_patch_trajectory(model, poses, soles)
        assert_allclose(heights, 0, atol=1e-8)
        assert_allclose(maximum[:2], 0, atol=1e-8)
        assert_allclose(span[:2], 0, atol=1e-8)
        assert_allclose(maximum[2], .1, atol=1e-8)
        assert_allclose(span[2], .1, atol=1e-8)

    def test_support_patch_height_jacobian(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="foot" pos="0 0 .2"><freejoint/>
            <geom type="box" size=".1 .04 .05"/>
          </body></worldbody></mujoco>''')
        soles = foot.build_robot_soles(model, dict(left="foot", right="foot"))
        foot.build_support_patches(model, soles)
        data = mujoco.MjData(model)
        data.qpos[3:7] = [.8, .2, .4, .4]
        data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
        mujoco.mj_forward(model, data)
        pose = data.qpos.copy()
        points, jac = foot.sole_height_kinematics(model, data, soles[0])
        patch = soles[0]["support_indices"]
        for dof in range(model.nv):
            dq = np.zeros(model.nv); dq[dof] = 1e-7
            data.qpos[:] = pose
            mujoco.mj_integratePos(model, data.qpos, dq, 1)
            mujoco.mj_forward(model, data)
            actual = foot.sole_world_points(data, soles[0])
            assert_allclose((actual[patch, 2]-points[patch, 2])/1e-7, jac[patch, dof], atol=2e-8)

    def timing_plan(self):
        h = np.zeros((61, 2)); h[10:41, 0] = .008
        v = np.zeros_like(h); v[10:41, 0] = .3
        centers = np.zeros((61, 2, 3)); centers[:, 0, 0] = np.cumsum(v[:, 0]) / 30
        config = dict(self.motion_config(), transport_timing={"enabled": True})
        return foot.plan_swing_clearance(h, 30, config, v, centers), centers, config

    def test_transport_waits_for_lift_and_stops_before_touchdown(self):
        plan, centers, _ = self.timing_plan()
        times = plan["transport_sample_times"][:, 0]
        event = plan["transport_timing_events"][0]
        a, b = event["start"], event["end"]
        self.assertTrue(np.all(np.diff(times) >= 0))
        self.assertEqual(times[a], a); self.assertEqual(times[b], b)
        self.assertEqual(times[a + 1], a)
        self.assertEqual(times[b - 1], b)
        self.assertGreater(plan["target_heights"][a + 1, 0], 0)
        self.assertGreater(plan["target_heights"][b - 1, 0], 0)
        assert_allclose(plan["target_heights"][plan["stance"]], 0)
        shifted = foot.sample_track(centers[:, 0], times)
        assert_allclose(shifted[a + 1, :2], centers[a, 0, :2])
        assert_allclose(shifted[b - 1, :2], centers[b, 0, :2])
        self.assertGreater(np.linalg.norm(shifted[b, :2] - shifted[a, :2]), .1)

    def test_timing_disabled_preserves_legacy_plan(self):
        # The disabled path must not create a timing map or change legacy XY.
        config = dict(self.motion_config(), transport_timing={"enabled": False})
        h = np.zeros((61, 2)); h[10:41, 0] = .008
        v = np.zeros_like(h); v[10:41, 0] = .3
        centers = np.zeros((61, 2, 3)); centers[:, 0, 0] = np.cumsum(v[:, 0]) / 30
        before = foot.plan_swing_clearance(h, 30, self.motion_config(), v, centers)
        after = foot.plan_swing_clearance(h, 30, config, v, centers)
        for key in before:
            if isinstance(before[key], np.ndarray): assert_allclose(before[key], after[key])
        self.assertNotIn("transport_sample_times", after)

    def test_timing_keeps_partial_flights_and_stationary_pivots(self):
        config = dict(self.motion_config(), transport_timing={"enabled": True})
        h = np.full((20, 2), .02); v = np.full_like(h, .3)
        centers = np.zeros((20, 2, 3)); centers[:, :, 0] = np.arange(20)[:, None] * .01
        plan = foot.plan_swing_clearance(h, 30, config, v, centers)
        assert_allclose(plan["transport_sample_times"], np.repeat(np.arange(20)[:, None], 2, axis=1))
        self.assertFalse(plan["transport_timing_active"].any())
        plan = foot.plan_swing_clearance(h * 0, 30, config, v * 0, centers)
        self.assertFalse(plan["transport_timing_active"].any())

    def test_short_step_finishes_transfer_before_braking_sample(self):
        # A minimal three-sample flight: lift, translate, brake, then contact.
        config = dict(self.motion_config(), transport_timing={"enabled": True})
        plan = {"stance": np.ones((10, 2), dtype=bool), "transport": np.zeros((10, 2), dtype=bool),
                "source_heights": np.zeros((10, 2)), "target_heights": np.zeros((10, 2)),
                "transport_envelope": np.zeros((10, 2)), "events": []}
        plan["stance"][3:6, 0] = False; plan["transport"][3:6, 0] = True
        foot.plan_transport_timing(plan, 30, config)
        times = plan["transport_sample_times"][:, 0]
        self.assertEqual(times[3], 2)
        self.assertEqual(times[4], 6)
        self.assertEqual(times[5], 6)
        self.assertGreater(plan["transport_speed_height"][4, 0], .015)
        self.assertEqual(plan["transport_speed_height"][5, 0], 0)

    def test_retimed_vectors_follow_same_source_time_and_stay_unit(self):
        plan, _, _ = self.timing_plan()
        angles = np.arange(61) * .02
        vectors = np.repeat(np.stack([np.cos(angles), np.sin(angles), np.zeros(61)], axis=1)[:, None], 3, axis=1)
        result = foot.retime_foot_vectors(vectors, plan, [np.array([0]), np.array([1])])
        assert_allclose(np.linalg.norm(result, axis=2), 1)
        assert_allclose(result[:, 2], vectors[:, 2])
        event = plan["transport_timing_events"][0]
        assert_allclose(result[event["start"] + 1, 0], vectors[event["start"], 0])

    def test_retimed_foot_positions_keep_height_target_and_other_body_slots(self):
        plan, centers, _ = self.timing_plan()
        slots = np.ones((61, 3, 3))
        slots[:, :2] = centers
        slots[:, :2, 2] = plan["source_heights"]
        result = foot.shift_foot_targets(slots, plan, [np.array([0]), np.array([1])])
        assert_allclose(result[:, :2, 2], plan["target_heights"], atol=1e-12)
        assert_allclose(result[:, 2], slots[:, 2])
        event = plan["transport_timing_events"][0]
        assert_allclose(result[event["start"] + 1, 0, :2], slots[event["start"], 0, :2])
        assert_allclose(result[event["end"] - 1, 0, :2], slots[event["end"], 0, :2])

    def test_material_velocity_constraints_cover_translation_and_tilt(self):
        import smpl_surface_retarget_common as common
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="left" pos="0 0 .025"><joint type="free"/><geom type="box" size=".1 .04 .02"/></body>
          <body name="right" pos="0 -.3 .025"><geom type="box" size=".1 .04 .02"/></body>
          </worldbody></mujoco>''')
        soles = foot.build_robot_soles(model, {"left": "left", "right": "right"})
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        previous = foot.sole_world_points(data, soles[0]).copy()
        A, b = foot.sole_horizontal_velocity_rows(model, data, soles[0], previous, .03, 30)
        desired = np.array([.02, .01, .001, .1, .1, .1])
        dq = common.solve_clarabel_qp_step(np.eye(6), -desired, 0., np.full(6, -.2), np.full(6, .2), ineq_A=A, ineq_b=b)
        mujoco.mj_integratePos(model, data.qpos, dq, 1); mujoco.mj_forward(model, data)
        points = foot.sole_world_points(data, soles[0])
        self.assertLess(np.linalg.norm(points[:, :2] - previous[:, :2], axis=1).max() * 30, .04)

    def motion_config(self):
        return {"mode": "motion_aware", "contact_height": .01, "contact_speed": .05,
                "contact_confirm_time": .05, "travel_speed_on": .10, "travel_speed_off": .04,
                "min_travel_distance": .01, "slow_travel_distance": .02,
                "min_travel_height": .03, "preserve_low_support": True}

    def test_motion_clearance_covers_low_travel_after_a_high_peak(self):
        h = np.zeros((121, 2)); h[20:101, 0] = .008; h[30, 0] = .08
        v = np.zeros_like(h); v[20:101, 0] = .3
        centers = np.zeros((121, 2, 3)); centers[:, 0, 0] = np.cumsum(v[:, 0]) / 30
        phase = h.copy()
        plan = foot.plan_swing_clearance(h, 30, self.motion_config(), v, centers, phase)
        self.assertGreaterEqual(plan["target_heights"][45:90, 0].min(), .03)
        self.assertAlmostEqual(plan["target_heights"][30, 0], .08)
        assert_allclose(plan["target_heights"][:, 1], 0)
        assert_allclose(plan["target_heights"][plan["stance"]], 0)

    def test_zero_height_floors_preserve_source_swing_and_contacts(self):
        config = dict(self.motion_config(), min_travel_height=0., min_peak_height=0.,
                      transport_timing={"enabled": False})
        for height, speed in ((.008, .3), (.02, 0.), (.15, .3)):
            with self.subTest(height=height, speed=speed):
                h = np.full((61, 2), .006)  # Source sole bias during native support.
                h[10:41, 0] = height
                phase = np.zeros_like(h); phase[10:41, 0] = height
                v = np.zeros_like(h); v[10:41, 0] = speed
                centers = np.zeros((61, 2, 3)); centers[:, 0, 0] = np.cumsum(v[:, 0]) / 30
                before = foot.plan_swing_clearance(h, 30, self.motion_config(), v, centers, phase)
                plan = foot.plan_swing_clearance(h, 30, config, v, centers, phase)
                self.assertTrue(plan["swing"][:, 0].any())
                self.assertTrue(plan["stance"][:, 1].all())
                assert_allclose(plan["stance"], before["stance"])
                assert_allclose(plan["target_heights"][plan["swing"]], h[plan["swing"]])
                assert_allclose(plan["target_heights"][plan["stance"]], 0)
                assert_allclose(plan["boost"], 0)
                self.assertTrue(plan["events"])
                self.assertFalse(any(e["boosted"] for e in plan["events"]))
                self.assertNotIn("transport_sample_times", plan)

    def test_zero_travel_height_keeps_separate_peak_compensation(self):
        h = np.zeros((61, 2)); h[10:41, 0] = .008
        v = np.zeros_like(h); v[10:41, 0] = .3
        centers = np.zeros((61, 2, 3)); centers[:, 0, 0] = np.cumsum(v[:, 0]) / 30
        config = dict(self.motion_config(), min_travel_height=0., min_peak_height=.035)
        plan = foot.plan_swing_clearance(h, 30, config, v, centers)
        self.assertAlmostEqual(plan["target_heights"][:, 0].max(), .035)

    def test_invalid_travel_height_and_zero_height_with_timing_are_rejected(self):
        h = np.zeros((20, 2)); centers = np.zeros((20, 2, 3))
        for value in (-.01, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "min_travel_height"):
                foot.plan_swing_clearance(h, 30, dict(self.motion_config(), min_travel_height=value), h, centers)
        config = dict(self.motion_config(), min_travel_height=0., transport_timing={"enabled": True})
        with self.assertRaisesRegex(ValueError, "min_travel_height"):
            foot.plan_swing_clearance(h, 30, config, h, centers)

    def test_native_contact_overrides_smpl_height_bias(self):
        h = np.full((40, 2), .012); phase = np.zeros_like(h); v = np.zeros_like(h)
        plan = foot.plan_swing_clearance(h, 30, self.motion_config(), v, np.zeros((40, 2, 3)), phase)
        self.assertTrue(plan["stance"].all())
        assert_allclose(plan["target_heights"], 0)

    def test_stationary_pivot_patch_is_not_transport(self):
        h = np.full((80, 2), .003); v = np.full_like(h, .01)
        centers = np.zeros((80, 2, 3)); centers[:, 0, 0] = .04 * np.sin(np.arange(80) / 10)
        plan = foot.plan_swing_clearance(h, 30, self.motion_config(), v, centers)
        self.assertFalse(plan["transport"].any())
        self.assertTrue(plan["stance"].all())

    def test_motion_ramps_do_not_consume_confirmed_contacts(self):
        h = np.zeros((81, 2)); h[20:61, 0] = .008
        v = np.zeros_like(h); v[20:61, 0] = .3
        centers = np.zeros((81, 2, 3)); centers[:, 0, 0] = np.cumsum(v[:, 0]) / 30
        plan = foot.plan_swing_clearance(h, 30, self.motion_config(), v, centers)
        assert_allclose(plan["target_heights"][:20], 0)
        assert_allclose(plan["target_heights"][61:], 0)
        self.assertLess(np.abs(np.diff(plan["target_heights"][:, 0])).max(), .02)

    def test_motion_keeps_a_high_stationary_held_foot(self):
        h = np.zeros((40, 2)); h[:, 0] = .15
        plan = foot.plan_swing_clearance(h, 30, self.motion_config(), np.zeros_like(h), np.zeros((40, 2, 3)))
        assert_allclose(plan["target_heights"][:, 0], .15)
        self.assertFalse(plan["stance"][:, 0].any())

    def test_native_foot_fk_uses_absolute_root_and_original_rate(self):
        text = '''HIERARCHY
ROOT Hips
{
 OFFSET 0 100 0
 CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
 JOINT LeftFoot
 {
  OFFSET 10 -80 0
  CHANNELS 3 Yrotation Xrotation Zrotation
  JOINT LeftToe
  {
   OFFSET 0 0 20
   CHANNELS 3 Yrotation Xrotation Zrotation
  }
 }
 JOINT RightFoot
 {
  OFFSET -10 -80 0
  CHANNELS 3 Yrotation Xrotation Zrotation
  JOINT RightToe
  {
   OFFSET 0 0 20
   CHANNELS 3 Yrotation Xrotation Zrotation
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.1
0 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
10 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "foot.bvh"; path.write_text(text)
            height, speed, center, _ = foot.bvh_foot_phase(path, np.arange(2), .5)
            with self.assertRaises(ValueError):
                foot.bvh_foot_phase(path, np.arange(2), .5, expected_frames=3)
            with self.assertRaises(ValueError):
                foot.bvh_foot_phase(path, np.arange(2), .5, expected_fps=20)
        assert_allclose(height, 0)
        assert_allclose(speed, .5)
        assert_allclose(center[:, :, 2], .1)
        assert_allclose(center[1, :, 0] - center[0, :, 0], .05)

    def test_stance_anchor_is_fixed_until_contact_ends(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="left" pos=".1 .2 .1"><joint type="free"/>
            <geom type="box" size=".1 .04 .02"/></body>
          <body name="right" pos="0 -.2 .1"><geom type="box" size=".1 .04 .02"/></body>
          </worldbody></mujoco>''')
        soles = foot.build_robot_soles(model, {"left": "left", "right": "right"})
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        state = {}; foot.update_stance_anchors(data, soles, [True, False], state)
        original = state[0]["world_xy"].copy()
        data.qpos[0] += .1; mujoco.mj_forward(model, data)
        foot.update_stance_anchors(data, soles, [True, False], state)
        assert_allclose(state[0]["world_xy"], original)
        foot.update_stance_anchors(data, soles, [False, False], state)
        self.assertFalse(state)
        foot.update_stance_anchors(data, soles, [True, False], state)
        assert_allclose(state[0]["world_xy"], original + [.1, 0])
    def test_surface_ground_uses_soles_and_rejects_one_low_outlier(self):
        vertices = np.zeros((20, 8, 3), dtype=np.float32)
        vertices[:, :, 2] = -.03
        vertices[0, :4, 2] = -.5
        joints = np.zeros((20, 2, 3), dtype=np.float32)
        joints[:, :, 2] = .02
        original = vertices.copy()
        aligned, aligned_joints, ground, info = foot.align_source_soles(
            vertices, joints, [np.arange(4), np.arange(4, 8)], 60)
        self.assertAlmostEqual(ground, -.03)
        assert_allclose(aligned[1:, :, 2], 0, atol=1e-8)
        assert_allclose(aligned_joints[:, :, 2], .05, atol=1e-8)
        assert_allclose(vertices, original)
        self.assertFalse(info["low_height_fallback"])

    def test_absolute_peak_and_stance_preservation(self):
        heights = np.zeros((61, 2))
        heights[10:51, 0] = .018 * np.sin(np.linspace(0, np.pi, 41)) ** 2
        plan = foot.plan_swing_clearance(heights, 60)
        self.assertEqual(sum(e["boosted"] for e in plan["events"]), 1)
        self.assertAlmostEqual(plan["target_heights"][:, 0].max(), .035)
        assert_allclose(plan["target_heights"][:, 1], 0)
        assert_allclose(plan["boost"][[0, -1]], 0)
        event = plan["events"][0]
        self.assertEqual(plan["boost"][event["start"], 0], 0)
        self.assertEqual(plan["boost"][event["end"], 0], 0)
        self.assertLess(np.abs(np.diff(plan["boost"][:, 0])).max(), .004)

    def test_high_kick_not_lowered_and_partial_lifts_not_boosted(self):
        heights = np.zeros((61, 2))
        heights[10:51, 0] = .15 * np.sin(np.linspace(0, np.pi, 41)) ** 2
        heights[:, 1] = .015  # airborne throughout this partial clip
        plan = foot.plan_swing_clearance(heights, 60)
        assert_allclose(plan["boost"], 0)
        self.assertAlmostEqual(plan["target_heights"][:, 0].max(), .15)
        assert_allclose(plan["target_heights"][:, 1], .015)

    def test_contact_jitter_and_tip_pivot_do_not_trigger_lift(self):
        heights = np.full((50, 2), .001)
        heights[20, 0] = .015
        plan = foot.plan_swing_clearance(heights, 60)
        self.assertFalse(plan["swing"].any())
        assert_allclose(plan["boost"], 0)

    def test_groups_can_override_zero_defaults_without_affecting_arms(self):
        names = ["l_shoulder_pitch_joint", "r_shoulder_pitch_joint", "l_calf_joint", "head_joint"]
        groups = {"arms": {"joints": ["*_shoulder_*"], "smooth_cost": 4., "temporal_smooth_cost": 8.},
                  "legs": {"joints": ["*_calf_joint"], "smooth_cost": 1., "temporal_smooth_cost": 1.}}
        smooth, temporal, _ = foot.joint_smoothing_weights(names, 0, 0, groups)
        assert_allclose(smooth, [4, 4, 1, 0])
        assert_allclose(temporal, [8, 8, 1, 0])
        with self.assertRaises(ValueError):
            foot.joint_smoothing_weights(names, 1, 1, {"bad": {"joints": ["missing*"]}})
        with self.assertRaises(ValueError):
            foot.joint_smoothing_weights(names, 1, 1, {"a": {"joints": ["*"]}, "b": {"joints": ["*"]}})

    def test_robot_sole_measures_tilted_box_not_joint_center(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="left" pos="0 .1 .2"><joint name="tilt" axis="0 1 0"/>
          <geom type="box" size=".1 .04 .02"/></body>
          <body name="right" pos="0 -.1 .1"><geom type="box" size=".1 .04 .02"/></body>
          </worldbody></mujoco>''')
        soles = foot.build_robot_soles(model, {"left": "left", "right": "right"})
        theta = .4
        heights, _ = foot.robot_sole_trajectory(model, np.array([[theta]]), soles)
        self.assertAlmostEqual(heights[0, 0], .2 - .1 * np.sin(theta) - .02 * np.cos(theta), places=7)
        self.assertAlmostEqual(heights[0, 1], .08, places=7)

    def test_slot_compensation_only_changes_foot_z(self):
        heights = np.zeros((61, 2)); heights[10:51, 0] = .02 * np.sin(np.linspace(0, np.pi, 41)) ** 2
        plan = foot.plan_swing_clearance(heights, 60)
        slots = np.ones((61, 5, 3))
        shifted = foot.shift_foot_targets(slots, plan, [np.array([0, 1]), np.array([2, 3])])
        assert_allclose(shifted[:, :, :2], slots[:, :, :2])
        assert_allclose(shifted[:, 4], slots[:, 4])
        assert_allclose(shifted[:, 2:4], slots[:, 2:4])

    def test_vectorized_sole_jacobian_matches_finite_difference(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="left"><joint type="free"/><geom type="box" size=".1 .04 .02"/></body>
          <body name="right"><geom type="box" size=".1 .04 .02"/></body>
          </worldbody></mujoco>''')
        soles = foot.build_robot_soles(model, {"left": "left", "right": "right"})
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        _, jac = foot.sole_height_kinematics(model, data, soles[0])
        q = data.qpos.copy(); eps = 1e-6
        for dof in range(model.nv):
            delta = np.zeros(model.nv); delta[dof] = eps
            data.qpos[:] = q; mujoco.mj_integratePos(model, data.qpos, delta, 1.)
            mujoco.mj_forward(model, data); plus = foot.sole_world_points(data, soles[0])[:, 2]
            data.qpos[:] = q; mujoco.mj_integratePos(model, data.qpos, -delta, 1.)
            mujoco.mj_forward(model, data); minus = foot.sole_world_points(data, soles[0])[:, 2]
            assert_allclose(jac[:, dof], (plus-minus)/(2*eps), atol=1e-8)

    def test_shared_slack_is_independent_of_hull_sample_count(self):
        import smpl_surface_retarget_common as common
        results = []
        for count in (1, 20):
            result = common.solve_clarabel_qp_step(
                np.eye(1), np.zeros(1), 0, np.array([-2.]), np.array([2.]),
                ineq_A=-np.ones((count, 1)), ineq_b=-np.ones(count),
                ineq_soft_costs=np.full(count, 3.), ineq_soft_groups=np.zeros(count, dtype=int))
            results.append(result)
        assert_allclose(results, [[.75], [.75]], atol=1e-5)

    def test_clearance_covers_both_toe_and_heel_during_tilt(self):
        import smpl_surface_retarget_common as common
        height_jac = np.array([[1., .1], [1., -.1]])
        step = common.solve_clarabel_qp_step(
            np.diag([10., 1.]), np.array([0., -.4]), 0.,
            np.array([-1., -.5]), np.array([1., .5]),
            ineq_A=-height_jac, ineq_b=np.full(2, -.035),
            ineq_soft_costs=np.full(2, 1e6), ineq_soft_groups=np.zeros(2, dtype=int))
        self.assertGreaterEqual((height_jac @ step).min(), .035 - 1e-5)


if __name__ == "__main__":
    unittest.main()

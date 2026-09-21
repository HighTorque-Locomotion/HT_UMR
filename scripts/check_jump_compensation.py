#!/usr/bin/env python3
"""Behavioural checks for jump recognition, fixed-time COM and support roles."""
import unittest
import numpy as np
import mujoco
from numpy.testing import assert_allclose

import retarget_jump_compensation as jump
import retarget_foot_clearance as foot


def synthetic_jump(takeoff=(0, 1), landing=(0, 1)):
    n, a, b, fps = 80, 20, 38, 30
    heights = np.zeros((n, 2, 2)); speed = np.zeros_like(heights)
    t = np.arange(b-a+1)/fps
    flight = .5*9.81*t*(t[-1]-t)
    for side in range(2):
        heights[a:b+1, side] = flight[:, None]
        speed[a+1:b, side] = np.maximum(.4, np.abs(9.81*(t[-1]/2-t[1:-1])))[:, None]
        if side not in takeoff: heights[:a+1, side] = .25
        if side not in landing: heights[b:, side] = .25
    hips = np.zeros((n, 3)); hips[:, 2] = 1; hips[a:b+1, 2] += flight
    hands = np.zeros((n, 3)); hands[:, 2] = 1.1
    return {"marker_height": heights, "marker_speed": speed,
            "joints": {"Hips": hips, "LeftHand": hands.copy(), "RightHand": hands.copy()}}, fps


class JumpChecks(unittest.TestCase):
    def test_support_roles_distinguish_single_and_double_jumps(self):
        for takeoff, landing, label in [((0,1),(0,1),"double_to_double"), ((0,),(0,),"left_to_left"),
                                        ((1,),(1,),"right_to_right"), ((0,),(0,1),"left_to_double"),
                                        ((0,1),(1,),"double_to_right"), ((0,1),(0,),"double_to_left"),
                                        ((1,),(0,1),"right_to_double"), ((0,),(1,),"left_to_right"),
                                        ((1,),(0,),"right_to_left")]:
            tracks, fps = synthetic_jump(takeoff, landing)
            result = jump.detect_jumps(tracks, fps, {})
            self.assertEqual(len(result["events"]), 1)
            e = result["events"][0]
            self.assertEqual(e["label"], label)
            self.assertEqual(e["takeoff_feet"], list(takeoff)); self.assertEqual(e["landing_feet"], list(landing))
            self.assertTrue((result["support_override"][e["start"]+1:e["end"]] == 0).all())

    def test_one_grounded_foot_is_not_a_single_leg_jump(self):
        tracks, fps = synthetic_jump()
        tracks["marker_height"][:, 0] = 0
        tracks["marker_speed"][:, 0] = 0
        self.assertFalse(jump.detect_jumps(tracks, fps, {})["events"])

    def test_fast_ground_sliding_is_not_flight(self):
        tracks, fps = synthetic_jump()
        tracks["marker_height"][:] = 0; tracks["marker_speed"][:] = 2
        self.assertFalse(jump.detect_jumps(tracks, fps, {})["events"])

    def test_overlapping_next_preparation_preserves_previous_landing(self):
        tracks, fps = synthetic_jump()
        a, b = 43, 61
        t = np.arange(b-a+1)/fps
        arc = .5*9.81*t*(t[-1]-t)
        tracks["marker_height"][a:b+1] = arc[:,None,None]
        tracks["marker_speed"][a+1:b] = np.maximum(.4,np.abs(9.81*(t[-1]/2-t[1:-1])))[:,None,None]
        tracks["joints"]["Hips"][a:b+1,2] = 1+arc
        # An impact sample can have high marker speed, followed by confirmed
        # contact. The second jump's preparation begins at this first landing.
        tracks["marker_speed"][38] = 1.0
        r=jump.detect_jumps(tracks,fps,{})
        self.assertEqual(len(r["events"]),2)
        for e in r["events"]:
            for side in e["landing_feet"]:
                self.assertEqual(r["support_override"][e["landing_contact_frames"][str(side)],side],1)

    def test_low_marker_and_slow_marker_must_be_the_same_point(self):
        tracks, fps = synthetic_jump()
        tracks["marker_height"][:20, :, 0] = 0
        tracks["marker_speed"][:20, :, 0] = 2
        tracks["marker_height"][:20, :, 1] = .2
        tracks["marker_speed"][:20, :, 1] = 0
        self.assertFalse(jump.detect_jumps(tracks, fps, {})["events"])

    def test_hand_supported_flight_is_rejected(self):
        tracks, fps = synthetic_jump()
        tracks["joints"]["LeftHand"][:, 2] = .02
        result = jump.detect_jumps(tracks, fps, {})
        self.assertFalse(result["events"])
        self.assertTrue(any(e["reason"] == "possible_hand_or_knee_support" for e in result["rejected"]))

    def test_confirmed_landing_ends_flight_despite_marker_floor_bias(self):
        tracks, fps = synthetic_jump()
        tracks["marker_height"][38:] = .012  # close, stationary contact; above release height
        result = jump.detect_jumps(tracks, fps, {})
        self.assertEqual(len(result["events"]),1)
        e=result["events"][0]
        self.assertEqual(e["end"],38)
        self.assertFalse(result["native_support"][e["start"]+1:e["end"]].any())

    def test_small_alternating_flight_requires_more_evidence(self):
        tracks, fps = synthetic_jump((0,), (1,))
        tracks["joints"]["Hips"][:, 2] = 1+.23*(tracks["joints"]["Hips"][:,2]-1)
        result = jump.detect_jumps(tracks, fps, {})
        self.assertFalse(result["events"])
        self.assertTrue(any(e["reason"] == "alternating_gait_or_uncertain_leap" for e in result["rejected"]))

    def test_unequal_height_ballistic_target_preserves_endpoints_and_time(self):
        n, fps = 60, 30
        com = np.zeros((n, 3)); com[:, 2] = .36
        com[30:, 2] = .34
        e = {"start": 18, "end": 30, "prepare_start": 12, "settle_end": 36}
        plan = jump.build_com_plan(com, {"events": [e]}, fps, {})
        z = plan["target_z"]
        self.assertEqual(z[18], com[18, 2]); self.assertAlmostEqual(z[30], com[30, 2])
        assert_allclose(np.diff(z[18:31], n=2)*fps**2, -9.81, atol=1e-9)
        self.assertAlmostEqual(plan["events"][0]["takeoff_com_vz"], (.34-.36)/.4+9.81*.4/2)
        self.assertEqual(len(z), n)
        assert_allclose(z[:12], com[:12, 2]); assert_allclose(z[37:], com[37:, 2])

    def test_height_limit_skips_instead_of_flattening_ballistic_arc(self):
        com = np.zeros((60,3)); com[:,2] = .36
        e = {"start": 15, "end": 36, "prepare_start": 10, "settle_end": 42}
        plan = jump.build_com_plan(com, {"events": [e]}, 30, {"max_com_rise": .2})
        self.assertFalse(plan["events"]); self.assertFalse(plan["active"].any())
        assert_allclose(plan["target_z"], com[:,2])

    def test_repeated_jumps_have_a_continuous_contact_bridge(self):
        com = np.zeros((80,3)); com[:,2] = .36
        events = [{"start":20,"end":32,"prepare_start":15,"settle_end":38},
                  {"start":37,"end":49,"prepare_start":32,"settle_end":55}]
        plan = jump.build_com_plan(com,{"events":events},30,{})
        self.assertEqual(len(plan["events"]),2)
        z=plan["target_z"]
        self.assertLess(z[34], .36)
        c=np.polyfit(np.arange(6)/30,z[32:38],2)
        self.assertAlmostEqual(c[1],plan["events"][0]["landing_com_vz"],places=8)
        self.assertAlmostEqual(2*c[0]*(5/30)+c[1],plan["events"][1]["takeoff_com_vz"],places=8)

    def test_support_feet_are_not_translated_with_airborne_body(self):
        slots = np.zeros((3, 3, 3)); slots[:,2,2] = .4
        foot_plan = {"target_heights": np.array([[0,.04],[.03,.04],[0,0.]]),
                     "source_heights": np.zeros((3,2)), "stance": np.array([[1,0],[0,0],[1,1]],dtype=bool), "events": []}
        new, feet = jump.compensate_targets(slots, foot_plan, {"delta_z": np.array([-.02,.10,-.03])}, [np.array([0]),np.array([1])])
        assert_allclose(new[:,2,2], [.38,.5,.37])
        assert_allclose(feet["target_heights"], [[0,.02],[.13,.14],[0,0]])
        assert_allclose(new[[0,2],0,2], 0)

    def test_jumps_are_excluded_from_walking_foot_retiming(self):
        h = np.zeros((61,2)); h[10:41] = .04
        v = np.zeros_like(h); v[10:41] = .3
        centers = np.zeros((61,2,3)); centers[:,:,0] = np.cumsum(v,axis=0)/30
        config = {"mode":"motion_aware","transport_timing":{"enabled":True}}
        exclusion = np.zeros(61,dtype=bool); exclusion[8:44]=True
        plan = foot.plan_swing_clearance(h,30,config,v,centers,timing_exclusion=exclusion)
        self.assertFalse(plan["transport_timing_active"].any())

    def test_com_uses_robot_masses_not_root_position(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="root"><freejoint/>
          <geom type="sphere" size=".05" mass="1"/><body pos="0 0 .4"><geom type="sphere" size=".05" mass="3"/></body>
          </body></worldbody></mujoco>''')
        q=np.repeat(m.qpos0[None],2,axis=0);q[1,2]=.2
        com,body=jump.robot_com_trajectory(m,q)
        assert_allclose(com[:,2],[.3,.5]);self.assertEqual(m.body(body).name,"root")


if __name__ == "__main__": unittest.main()

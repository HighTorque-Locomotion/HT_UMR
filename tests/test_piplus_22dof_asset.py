"""Check that the published robot is complete and matches the 22-DOF profile."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/piplus_40v_0908"


class PiPlusAssetTests(unittest.TestCase):
    def test_asset_loads_after_relocation_and_preserves_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "robot"
            shutil.copytree(
                ASSETS, target, ignore=shutil.ignore_patterns("*.floating_mjcf.xml")
            )
            model = mujoco.MjModel.from_xml_path(
                str(target / "piplus_40v_260908_no_hyperextension.xml")
            )
            self.assertEqual((model.nq, model.nv), (29, 28))
            scalar = [
                model.joint(i).name
                for i in range(model.njnt)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
            ]
            expected = json.loads((ASSETS / "asset_manifest.json").read_text())
            self.assertEqual(scalar, expected["scalar_joint_names"])
            self.assertEqual(len(scalar), 22)
            settings = json.loads(
                (
                    ROOT / "robot_configs/local_regions/piplus_40v_0908_settings.json"
                ).read_text()
            )
            self.assertEqual(set(scalar), set(settings["robot"]["tpose_qpos"]))
            for name, upper in [("r_calf_joint", 2.18), ("l_calf_joint", 2.182)]:
                joint = model.joint(name)
                np.testing.assert_allclose(joint.range, [0.0, upper])
                self.assertTrue(joint.limited[0])

    def test_mesh_checksums(self):
        manifest = json.loads((ASSETS / "asset_manifest.json").read_text())
        self.assertEqual(manifest["mesh_count"], 27)
        for item in manifest["meshes"]:
            content = (ASSETS / item["path"]).read_bytes()
            self.assertEqual(len(content), item["size_bytes"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()

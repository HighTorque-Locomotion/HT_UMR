"""Small asset-independent checks for semantic surface binding and 40V defaults."""
import copy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

import retarget_smpl_to_humanoid_surface_vector as rt
from humanoid_retarget_config import load_config


def main():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <asset><mesh name="tetra" vertex="0 0 0 .05 0 0 0 .05 0 0 0 .05"
        face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset>
      <worldbody><body name="base"><freejoint/>
        <geom type="mesh" mesh="tetra" group="1" contype="0" conaffinity="0"/>
        <body name="upper" pos=".2 0 0"><joint name="upper_joint" axis="0 0 1"/>
          <geom type="mesh" mesh="tetra" group="1" contype="0" conaffinity="0"/>
          <body name="forearm" pos=".2 0 0"><joint name="elbow_joint" axis="0 1 0"/>
            <geom type="mesh" mesh="tetra" group="1" contype="0" conaffinity="0"/>
          </body>
        </body>
      </body></worldbody></mujoco>''')
    config = {'robot': {'name': 'toy', 'xml': 'unused.xml', 'point_cloud_center': 'body:base',
                       'sample_pose': 'default', 'frame_transform': {'smpl_to_robot_root_matrix': np.eye(3).tolist()}}}
    points = np.array([[.01,.01,0],[.02,0,.01],[0,.01,.02]],dtype=np.float32)
    labels = np.full(3,rt.SMPLX_PART_IDS['rightForeArm'],dtype=np.int32)
    def bind(cfg, **kwargs):
        return rt.bind_robot_slots(model,SimpleNamespace(config_data=cfg),points,source_slot_part_ids=labels,**kwargs)
    original = bind(config)
    empty = copy.deepcopy(config);empty['robot']['surface_slot_body_bindings']={}
    for key in ('geom_ids','local_pos','local_normals','root_points'):
        np.testing.assert_array_equal(original[key],bind(empty)[key])
    fixed_config = copy.deepcopy(config)
    fixed_config['robot']['surface_slot_body_bindings']={'rightForeArm':['forearm']}
    fixed = bind(fixed_config)
    import retarget_smpl_to_humanoid_surface_vector_batch as batch
    batch_fixed=batch.bind_robot_slots(model,SimpleNamespace(config_data=fixed_config),points,source_slot_part_ids=labels)
    for key in ('geom_ids','local_pos','local_normals','root_points'):
        np.testing.assert_array_equal(fixed[key],batch_fixed[key])
    assert all(model.geom_bodyid[g]==model.body('forearm').id for g in fixed['geom_ids'])
    np.testing.assert_allclose(np.linalg.norm(fixed['local_normals'],axis=1),1,atol=1e-6)
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    vertices,faces,_,_,_,_=rt.collect_root_mesh(model,data,np.unique(fixed['geom_ids']),'body:base')
    projected=rt.common.bind_points_to_mesh(fixed['root_points'],vertices,faces)
    assert projected['errors'].max()<1e-6
    dof=int(model.jnt_dofadr[model.joint('elbow_joint').id])
    old_cache=rt.common.TemplateSlotKinematicsCache(model,data,original)
    new_cache=rt.common.TemplateSlotKinematicsCache(model,data,fixed)
    assert all(np.linalg.norm(old_cache.point_jacobian(i)[:,dof])==0 for i in range(3))
    assert any(np.linalg.norm(new_cache.point_jacobian(i)[:,dof])>1e-6 for i in range(3))
    for mapping in ({'unknown_segment':['forearm']},{'rightForeArm':['missing_body']},{'rightForeArm':[]}):
        invalid=copy.deepcopy(config);invalid['robot']['surface_slot_body_bindings']=mapping
        try:bind(invalid)
        except ValueError:pass
        else:raise AssertionError(f'Invalid mapping was accepted: {mapping}')
    try:bind(fixed_config,project_to_surface=False)
    except ValueError:pass
    else:raise AssertionError('Semantic binding must project onto a real surface')
    root=Path(__file__).resolve().parents[1]
    checked=[]
    for path in sorted((root/'demo_configs').glob('*.json')):
        cfg=load_config(path)
        if cfg.get('robot',{}).get('name')!='piplus_s_40v':continue
        assert len(cfg['robot']['surface_slot_body_bindings'])==6,path
        assert cfg['solver']['iters']==3 and cfg['solver']['pose_init_iters']==60,path
        assert cfg['robot']['joint_limits']['r_elbow_joint']==[-1.745,1.745],path
        checked.append(path.name)
    assert checked
    strong=load_config(root/'demo_configs/umr_demo_piplus_s_40v_self_collision_strong.json')
    assert strong['solver']['robot_self_penetration_hard_constraint']
    assert strong['solver']['robot_self_penetration_cost']==10000
    assert strong['solver']['trajectory_filter_mode']=='off'
    assert not (root/'demo_configs/umr_aiming1_40v_pro_xml_limits.json').exists()
    print('PASS: surface membership, articulated influence, legacy compatibility, invalid inputs, 40V defaults, strong collision')
    print('40V configurations:',', '.join(checked))


if __name__=='__main__':main()

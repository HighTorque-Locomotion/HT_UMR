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

# 对比
基础演示与机器人对照：

| 配置 | 机器人／默认动作 | 主要设置与用途 |
|---|---|---|
| [demo_g1_quick](/data/HT_UMR/demo_configs/umr_demo_g1_quick.json) | G1／`dance1_subject2` | 只处理前 **60 帧**；1024 个对应点、训练 20 轮，用于快速跑通 G1 流程 |
| [demo_piplus_s_40v](/data/HT_UMR/demo_configs/umr_demo_piplus_s_40v.json) | PiPlus S 40V／`dance1_subject2` | 前 **60 帧**；配置 40V 模型、关节限位和手臂语义绑定；每帧求解 3 次、初始化 60 次 |
| [demo_piplus_s_40v_full](/data/HT_UMR/demo_configs/umr_demo_piplus_s_40v_full.json) | 40V／完整 `dance1_subject2` | 在上一个配置上取消帧数限制；**不增加对应点数量或训练轮数** |
| [demo_pipluspro_full](/data/HT_UMR/demo_configs/umr_demo_pipluspro_full.json) | PiPlusPro／完整 `dance1_subject2` | 使用 Pro 模型、T-pose 和关节限位；1024 点、20 轮；继承默认每帧 1 次、初始化 15 次求解 |
| [aiming1_piplus_s_40v](/data/HT_UMR/demo_configs/umr_aiming1_piplus_s_40v.json) | 40V／完整 `aiming1_subject1` | 40V 完整版切换到瞄准动作，并修改输出路径 |
| [aiming1_pipluspro](/data/HT_UMR/demo_configs/umr_aiming1_pipluspro.json) | Pro／完整 `aiming1_subject1` | Pro 完整版切换到瞄准动作，用于机器人之间的对照 |
| [dance1_40v_semantic_arms](/data/HT_UMR/demo_configs/umr_dance1_40v_semantic_arms.json) | 40V／完整 `dance1_subject2` | 兼容旧入口；现在与普通 40V 完整版采用相同的手臂绑定修复，主要区别是输出路径 |
| [aiming1_40v_semantic_arms](/data/HT_UMR/demo_configs/umr_aiming1_40v_semantic_arms.json) | 40V／完整 `aiming1_subject1` | 上述兼容入口的瞄准动作版本 |
| [lafan_dance2_subject4_g1_compare](/data/HT_UMR/demo_configs/umr_lafan_dance2_subject4_g1_compare.json) | G1／完整 `dance2_subject4` | 使用 G1 示例限位；**4096 点、500 轮、CPU**；作为原始 G1 对照，不启用新增足部补偿或机器人自碰撞约束 |

以上配置默认没有开启机器人几何自穿透约束，但仍继承基础地面约束和人体自接触映射。所有 40V 配置都已包含手臂绑定修复，`semantic_arms` 不再代表额外功能。

自碰撞与肩部连续性：

| 配置 | 相对基础配置的变化 | 实现的功能 |
|---|---|---|
| [aiming1_40v_self_collision](/data/HT_UMR/demo_configs/umr_aiming1_40v_self_collision.json) | 开启自碰撞硬约束；间隙目标 **0 mm**；允许松弛，惩罚 **1000**；保留 LQR 后滤波 | 瞄准动作的普通自碰撞版本 |
| [demo_piplus_s_40v_self_collision_strong](/data/HT_UMR/demo_configs/umr_demo_piplus_s_40v_self_collision_strong.json) | 软惩罚 **10000**、软目标间隙 **3 mm**；硬目标间隙 **2 mm**、松弛惩罚 **1000000**；关闭后滤波 | 完整舞蹈的强自碰撞版本，强化全身符合条件的几何对避碰 |
| [aiming1_40v_self_collision_strong](/data/HT_UMR/demo_configs/umr_aiming1_40v_self_collision_strong.json) | 继承强自碰撞，切换到完整 `aiming1_subject1` | 瞄准动作的强自碰撞版本 |
| [zoo3_piplus_s_40v_self_collision_strong](/data/HT_UMR/demo_configs/umr_zoo3_piplus_s_40v_self_collision_strong.json) | 强自碰撞＋`_limit.xml` 模型；双侧肩部及上臂关节每次迭代限制 **0.03 rad**；平滑权重 **4/8**；按脚关节对齐地面；设备 `auto` | 处理转换后的 Noitom `zoo_3`，减少肩部突跳；也是后续足部配置的基础 |

这些自碰撞约束允许松弛，因此“strong”表示更强的约束设置，不代表结果必然零穿透。

足底、抬脚与脚步时序：

以下配置均继承 **Zoo3 的强自碰撞、限位模型和肩部连续性设置**。

| 配置 | 新增／修改设置 | 实现的功能 |
|---|---|---|
| [piplus_s_40v_foot_clearance](/data/HT_UMR/demo_configs/umr_piplus_s_40v_foot_clearance.json) | 用**足底表面**估计地面；`peak` 模式，摆动峰值下限 **3.5 cm**；肩部平滑 **4/8**、腿部 **1/1** | 改善地面对齐，提高可信低抬脚摆动的峰值，并分别控制肩腿平滑 |
| [lafan_dance_foot_clearance](/data/HT_UMR/demo_configs/umr_lafan_dance_foot_clearance.json) | 改为 `motion_aware`；从原 BVH 判断接触；移动阶段高度下限 **3 cm**；支撑点水平锚定；最多 8 次碰撞补求解 | 处理移动途中拖地、接触判断偏差和支撑脚滑动 |
| [lafan_dance2_subject4_clearance_fix](/data/HT_UMR/demo_configs/umr_lafan_dance2_subject4_clearance_fix.json) | 将上述配置的输入固定为完整 `dance2_subject4`，单独输出 | 该舞蹈片段的 3 cm 修复实例 |
| [lafan_dance2_subject4_clearance_5cm](/data/HT_UMR/demo_configs/umr_lafan_dance2_subject4_clearance_5cm.json) | 上一个实例的移动高度下限 **3 → 5 cm** | 比较提高抬脚高度的效果；没有增加脚步时序重排 |
| [lafan_dance_transport_timing](/data/HT_UMR/demo_configs/umr_lafan_dance_transport_timing.json) | 在 `motion_aware` 上开启 `transport_timing`；抬脚阶段 0.08 s、落脚阶段 0.10 s；增加近地水平速度约束 | 实现**先抬后移、落脚前减速**，减少起落脚阶段的近地移动 |
| [lafan_dance2_subject4_transport_timing](/data/HT_UMR/demo_configs/umr_lafan_dance2_subject4_transport_timing.json) | 上述时序配置指定完整 `dance2_subject4`，单独输出 | 该舞蹈片段的时序修复实例，移动高度仍为 3 cm |
| [lafan_transport_timing_3cm](/data/HT_UMR/demo_configs/umr_lafan_transport_timing_3cm.json) | 显式设置移动高度 **3 cm** | 与当前 `lafan_dance_transport_timing` 实际参数相同，提供明确的高度入口 |
| [lafan_transport_timing_5cm](/data/HT_UMR/demo_configs/umr_lafan_transport_timing_5cm.json) | 移动高度改为 **5 cm** | 时序修复＋更高的移脚目标 |
| [lafan_run_foot_contact](/data/HT_UMR/demo_configs/umr_lafan_run_foot_contact.json) | 基于 `lafan_dance_foot_clearance`；开启 BVH 腿脚参考校准；支撑足底放平权重 **1000000**；最多 8 次穿地补求解；关闭模糊低位时强留一脚支撑的规则；时序重排关闭 | 面向跑步的足部接触配置，强化支撑足底贴地；**不启用跳跃 COM 补偿** |

这里有三个不同层次：`peak` 提高一次摆动的**峰值**；`motion_aware` 提高**移动阶段**的脚高；`transport_timing` 进一步调整脚部**什么时候抬、什么时候移、什么时候落**。

跳跃重心与 BVH 参考校准：

以下配置均继承 **3 cm 时序配置**，并开启跳跃重心（COM）补偿：识别可信跳跃，保持起落帧和腾空时间，按 `g=9.81 m/s²` 构造机器人质量重心的竖直轨迹。普通摆步继续使用脚步时序调整；可信跳跃窗口采用跳跃处理。

| 配置 | 移动阶段高度下限 | BVH 腿脚参考校准 | 区别与用途 |
|---|---:|:---:|---|
| [lafan_jump_compensation](/data/HT_UMR/demo_configs/umr_lafan_jump_compensation.json) | 3 cm | 关 | 跳跃补偿基础配置；增加 COM 高度、速度跟踪，以及起落阶段支撑约束 |
| [lafan_jump_compensation_5cm](/data/HT_UMR/demo_configs/umr_lafan_jump_compensation_5cm.json) | 5 cm | 关 | 仅提高普通移脚高度下限 |
| [lafan_jump_compensation_10cm](/data/HT_UMR/demo_configs/umr_lafan_jump_compensation_10cm.json) | 10 cm | 关 | 同上，使用 10 cm |
| [lafan_jump_compensation_20cm](/data/HT_UMR/demo_configs/umr_lafan_jump_compensation_20cm.json) | 20 cm | 关 | 同上，使用 20 cm |
| [lafan_jump_compensation_bvh_reference](/data/HT_UMR/demo_configs/umr_lafan_jump_compensation_bvh_reference.json) | 3 cm | 开 | 校准双侧大腿、小腿、脚的姿态参考和横向位置，改善参考偏差引起的腿内收、脚倾斜 |
| [lafan_jump_compensation_5cm_bvh_reference](/data/HT_UMR/demo_configs/umr_lafan_jump_compensation_5cm_bvh_reference.json) | 5 cm | 开 | 参考校准＋5 cm 移脚 |
| [lafan_jump_compensation_10cm_bvh_reference](/data/HT_UMR/demo_configs/umr_lafan_jump_compensation_10cm_bvh_reference.json) | 10 cm | 开 | 参考校准＋10 cm 移脚 |
| [lafan_jump_compensation_20cm_bvh_reference](/data/HT_UMR/demo_configs/umr_lafan_jump_compensation_20cm_bvh_reference.json) | 20 cm | 开 | 参考校准＋20 cm 移脚 |
| [lafan_jumps1_subject2_com](/data/HT_UMR/demo_configs/umr_lafan_jumps1_subject2_com.json) | 3 cm | 关 | 基础跳跃配置指定完整 `jumps1_subject2`，单独输出 |

**使用时容易混淆的几点：**

- **`5cm/10cm/20cm` 都是移脚高度下限，不是跳跃高度。** 跳跃 COM 目标由腾空时间及起落高度决定；这些变体没有修改跳跃参数。
- **通用 `lafan_*` 配置不一定默认读取 LAFAN 动作。** 没有显式覆盖 `motion` 的配置，包括通用跳跃和跑步配置，实际仍继承 `zoo_3` 输入；需要通过命令行或批处理指定目标动作。
- **多个通用变体也继承同一个输出路径**，比较结果时应分别指定输出位置。
- 除两个 60 帧演示外，其余配置均为 `max_frames=0`，表示不限制帧数；全部关闭自动查看器。BVH 接触识别和参考校准还需要相应转换信息及可读取的原始 BVH。

---

# 批量处理
``` 
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u scripts/retarget_noitom_bvh_batch.py \
    --source-root /data/GMR/assets/lafan \
    --pattern 'run*.bvh' \
    --config demo_configs/umr_lafan_run_foot_contact.json \
    --slots output/demo_piplus_s_40v/correspondence_slots/correspondence_slots_final.npz \
    --output-root output/piplush0w_lafan_umr_runflat_260917 \
    --source-format lafan \
    --output-name-template '{name}_{date}_{robot}_umr' \
    --output-date 260917 \
    --workers 4 \
    --cpu-threads 2

```

# 使用新绑定的点
``` 
python -u scripts/retarget_lafan_betas16_batch.py \
  --source-root /data/GMR/assets/lafan \
  --pattern 'fall*.bvh' \
  --config output/piplus_lafan_walk_betas16_clearance15_260926/debug_leg_angles/leg_binding_no_clearance.json \
  --slots output/piplus_walk1_betas16_legacy/correspondence_slots/correspondence_slots_final.npz \
  --converter-python miniconda3/envs/gmr/bin/python \
  --output-root output/piplus_lafan_fall_betas16_legbind_no_clearance_260926 \
  --output-name-template '{name}_{date}_{robot}_umr_betas16_legbind_no_clearance' \
  --output-date 260926 \
  --workers 4 \
  --cpu-threads 2

  ```

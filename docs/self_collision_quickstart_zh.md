# UMR 自碰撞开关：40V 与足部避碰速查

日期：2026-09-13。以下命令适用于本仓库修复后的 40V 和逐帧重定向入口。

## 1. 通俗原理

可以把机器人各部件的碰撞几何想成“外壳”。每次优化姿态时，程序找出靠得太近的两个外壳，计算它们的间距，再计算“哪些关节往哪个方向转，可以让它们分开”。接着同时考虑人体动作拟合和避碰要求，更新关节角度。

- 间距为正：尚未相碰；为零：接触；为负：发生穿透。
- **软惩罚**：太近就扣分，让优化器倾向于分开。
- **硬约束**：要求下一步预测间距达到指定值。
- **松弛量（slack）**：如果避碰和动作/限位冲突，允许付出较大代价后少量违反约束，避免无解。

底层用 MuJoCo 寻找候选碰撞对（`mj_collision`）、计算有符号距离（`mj_geomDistance`），再用距离对关节的 Jacobian 构造局部优化约束。这是逐帧、局部线性近似的避碰，不是全局路径规划或连续碰撞检测，也不是通过物理仿真把脚弹开。

代码：[碰撞检测与距离 Jacobian](../scripts/smpl_surface_retarget_common.py) 中的 `build_robot_self_penetration_cache`、`compute_robot_self_penetration_rows` 和 `collision_relative_jacobian`；[求解器](../scripts/retarget_smpl_to_humanoid_surface_vector.py) 中的 `robot_self_cost` 和 `robot_self_penetration_hard_constraint` 分支。

## 2. 最方便的开启方法：复用强自碰撞配置

普通 40V 配置默认没有开启机器人自穿透约束。下面的强配置已包含手臂绑定修复，**对全身有效，也包含符合碰撞条件的左右脚/脚踝碰撞，不是只对脚生效**。

本机先进入环境和仓库；其他机器替换仓库路径：

```bash
conda activate umr
cd /home/sunteng/Projects/Retargeting/UMR
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl
```

完整 `dance1_subject2` 重定向并录制。新输出名带 `_rerun`，不覆盖此前交付的视频；再次执行仍会使用同名输出，请按需改名：

```bash
python scripts/retarget_smpl_to_humanoid_surface_vector.py \
  --config demo_configs/umr_demo_piplus_s_40v_self_collision_strong.json \
  --slots output/demo_piplus_s_40v/correspondence_slots/correspondence_slots_final.npz \
  --out output/self_collision_runs/dance1_subject2_40v_strong_rerun.npz

python scripts/visualize_robot_retarget_result.py \
  --result output/self_collision_runs/dance1_subject2_40v_strong_rerun.npz \
  --record-video output/self_collision_runs/dance1_subject2_40v_strong_rerun.mp4 \
  --record-width 960 --record-height 540 --camera-mode root --camera-distance 1.2
```

`aiming1_subject1`：把配置换成 `demo_configs/umr_aiming1_40v_self_collision_strong.json`，同时把上述输出和渲染输入中的动作名称换成 `aiming1_subject1`。两份配置都是完整动作，`motion.max_frames=0`。

前提：UMR 依赖、许可 SMPL-X 模型、40V XML/mesh、源动作和对应点缓存均已准备好。如果没有对应点缓存，可直接运行完整流水线：

```bash
python scripts/humanoid_retarget_pipeline.py \
  --config demo_configs/umr_demo_piplus_s_40v_self_collision_strong.json
```

该流水线使用配置内的默认输出位置，可能覆盖同名结果。换新 XML 必须使用新机器人的配置和对应点，不能复用 40V 的 slots；先执行[新 XML 适配清单](new_xml_surface_binding_checklist_zh.md)。

## 3. 给已有配置加开关

将以下字段合并进配置已有的 `solver` 对象，不要新建第二个同名 `solver`：

```json
{
  "solver": {
    "collision_threshold": 0.1,
    "robot_self_penetration_cost": 10000.0,
    "robot_self_penetration_tolerance": 0.003,
    "robot_self_penetration_hard_constraint": true,
    "robot_self_penetration_margin": 0.002,
    "robot_self_penetration_hard_slack": true,
    "robot_self_penetration_hard_slack_cost": 1000000.0,
    "trajectory_filter_mode": "off"
  }
}
```

| 参数 | 通俗含义 |
| --- | --- |
| `collision_threshold: 0.1` | 候选对搜索距离，单位米；不是要求两脚始终相隔 10 cm。求解器还会与 tolerance、margin 取最大值作为实际搜索阈值。 |
| `robot_self_penetration_cost: 10000` | 软惩罚权重；越大越偏向避碰，但可能牺牲动作还原。 |
| `robot_self_penetration_tolerance: 0.003` | 软惩罚期望保留 3 mm 净空；距离低于它才产生惩罚。 |
| `robot_self_penetration_hard_constraint: true` | 开启有符号距离不等式约束。 |
| `robot_self_penetration_margin: 0.002` | 硬约束目标净空 2 mm；零表示不穿透，负值允许轻微穿透。 |
| `robot_self_penetration_hard_slack: true` | 保留冲突时的退让机制，不承诺绝对零穿透。 |
| `robot_self_penetration_hard_slack_cost: 1000000` | 退让代价；越大越不愿违反约束，并非越大越稳定。 |
| `trajectory_filter_mode: "off"` | 不做后续轨迹滤波，避免滤波重新引入穿透。 |

只开硬约束的最小字段是 `robot_self_penetration_hard_constraint=true`，但其他参数会继承配置值；为便于复现，优先使用上面的完整强配置。

不改 JSON 时，也可给逐帧命令追加等价参数（命令行优先于配置）：

```bash
--collision-threshold 0.1 \
--robot-self-penetration-cost 10000 \
--robot-self-penetration-tolerance 0.003 \
--robot-self-penetration-hard-constraint \
--robot-self-penetration-margin 0.002 \
--robot-self-penetration-hard-slack \
--robot-self-penetration-hard-slack-cost 1000000 \
--trajectory-filter-mode off
```

以上是追加参数片段，不是独立命令。要关闭机器人自碰撞优化，需要同时追加 `--no-robot-self-penetration-hard-constraint --robot-self-penetration-cost 0`；只关硬约束仍可能保留软惩罚。关闭不会连带关闭地面约束。

## 4. 足部碰撞：开启后还要核对什么？

1. 左右脚的碰撞 geom 必须覆盖实际脚部，而不只是有漂亮的 visual mesh。`contype=0` 且 `conaffinity=0` 的 geom 会被本功能跳过。
2. MuJoCo 的碰撞 mask、模型 `<contact><exclude .../>`、父子/同刚体过滤等设置会影响候选对；启用优化不会自动恢复被 XML 排除的脚部碰撞。不要为了解决漏检而盲目打开所有相邻零件之间的碰撞。
3. 当前没有“仅左右脚”专用配置开关。强配置使用全身有效候选对；只优化脚部需要另外实现碰撞对筛选，本文不提供不存在的选项。
4. 脚和另一只脚是**自碰撞**；脚穿地是**地面穿透**，由 `ground_penetration_*` 控制。`self_contact_map_cost` 是人体接触关系迁移，也不是机器人脚部防穿透开关。
5. 开启后必须重新重定向生成 qpos，再录制视频。只重新渲染旧 NPZ 不会修复旧轨迹。

## 5. 如何确认生效，以及仍穿透时怎么办？

运行日志应出现 `[SurfaceRetarget][RobotSelfPenetration]`，其中 `hard_constraint=True`、`cost=10000`、`geoms` 大于零。**有候选 geom 不代表左右脚一定进入了候选对**；还需要回放检查及逐帧碰撞统计。

可对新结果执行下面的只读检查：统计所有有效机器人自碰撞接触中，超过 1 mm 的穿透帧数和部件对（不包含地面）。如果改了输出名，请同步修改 `result`：

```bash
python - <<'PY'
from collections import Counter
import mujoco
import numpy as np

result = "output/self_collision_runs/dance1_subject2_40v_strong_rerun.npz"
with np.load(result, allow_pickle=False) as motion:
    qpos = motion["qpos"]
    xml = str(motion["robot_xml"].item())
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
counts, bad_frames, worst = Counter(), 0, 0.0
for q in qpos:
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    pairs = set()
    for c in data.contact:
        if c.geom1 < 0 or c.geom2 < 0:
            continue
        a, b = model.geom_bodyid[[c.geom1, c.geom2]]
        if a == 0 or b == 0:
            continue
        worst = max(worst, -float(c.dist))
        if c.dist < -0.001:
            pairs.add(tuple(sorted((model.body(a).name, model.body(b).name))))
    bad_frames += bool(pairs)
    counts.update(pairs)
print("frames:", len(qpos), "frames over 1 mm:", bad_frames)
print("worst penetration (mm):", worst * 1000)
print("body pairs / affected frames:", counts.most_common())
PY
```

该统计遵循 XML 的碰撞过滤和碰撞几何，不能发现未建模、被过滤的碰撞，也不检查两帧之间的连续运动。

已验证的完整 dance1_subject2（3945 帧）：修复绑定、未开强碰撞为 654 帧超过 1 mm；开启本文强配置后为 5 帧，残余在脚踝之间，最大约 14.1 mm。**这些参数明显改善避碰，但不是足部零穿透保证。**

仍有脚部穿透时，先核对碰撞几何/过滤和具体帧；再以独立输出做短片段实验，例如追加 `--iters 10` 增加局部迭代，或逐步调整净空/松弛代价，最后重跑全长。增加搜索距离不等于增大脚间距。`--no-robot-self-penetration-hard-slack` 可禁止该类约束退让，但可能导致求解不可行；也不能消除局部线性化、离散采样或几何近似的限制。所有调整均需重新验证动作和连续性，不能直接作为真机安全保证。

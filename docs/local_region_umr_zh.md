# PiPlus 40V 0908：UMR 局部分界重定向

本分支提取了 BONES-SEED 批处理实际使用的局部分界适配器，基于官方 UMR
`d1cdaedaf67a827198da71c5fa831f48b2e51f91`。为保持已验证行为，分支直接从该提交建立，
未混入工作区中其他实验修改。没有修改上游求解器文件。

## 方法与适用范围

先用 **PiPlus 40V 260908 自身模型**采样 4096 点、训练 500 epochs 的对应关系，
再对学习出的机器人覆盖点施加人工定义的解剖分区。这一步是我们加入的推理阶段适配，
**不是 UMR 官方内置约束，也不是重新训练网络**。

1. 保留上游人体分区、slot 身份和采样配额。
2. 对每个受限区域，仅保留允许的机器人 body，并用 T-pose 下的空间边界裁剪三角面。
3. 将该区域学习出的点云相对位置映射到目标区域范围，投影到最近的合法三角面。
4. 保存 body 名、该 body 内的 geom 序号、局部位置与法向；运行时重新绑定，避免全局 geom
   编号因浮动基座 XML 生成而改变。

生产标定共限制 **14 个双侧肢体分区、2411 / 4096 点**。头、髋、躯干和肩帽区域保留原映射。
上臂分成近肩段和远肩段；前臂、手掌可使用同一 wrist 网格的不同空间部分。
脚部覆盖包含脚壳以上约 10 mm 的小腿末端，小腿从这一分界向上覆盖，适配较大的机器人脚。
准确边界见 `robot_configs/local_regions/piplus_40v_0908.json`，单位米、Z 向上。

参数模板保留膝盖防反屈（右膝 `[0, 2.18]`、左膝 `[0, 2.182]` rad）、肘部限位、
肩部关节每步变化限制 0.03、平滑项及自穿透硬约束。XML 本身也应使用匹配的防反屈资产。
重定向输出是运动学参考轨迹，不代表动力学平衡或实机安全已验证。

**仅标定于 SOMA uniform 模板。** 运行时冻结已校准的人体分区标签，最多接受 10 个边界点的
数值性标签变化，并写入审计；更多变化会报错。这个阈值不能证明源模板兼容，操作者仍须确保
模板、slot 顺序及机器人资产一致。不同机器人、SOMA proportional、SMPL-X 或任意 BVH
需要重新标定，不能直接套用本配置。

## 文件与外部输入

- `scripts/local_regions/configure.py`：从外部资产生成当前机器的配置。
- `capture.py`：捕获原始对应关系和分区标签，不求解运动轨迹。
- `prepare.py` / `clipped_surface.py`：生成受限对应关系。
- `retarget.py`：调用原始 UMR 求解器，注入局部分界绑定与源标签稳定处理。
- `robot_configs/local_regions/`：分区边界和生产参数模板。

依赖沿用根目录安装说明；需要可用的 MuJoCo、trimesh proximity 依赖、SOMA、人体模型及 CUDA
环境。本次未修改依赖版本。分支不新增 BONES-SEED 原始数据、SMPL/SOMA 人体资产、机器人网格、
训练权重、派生点云 NPZ 或动作结果。请自行取得相应许可，并提供：

- 匹配的 PiPlus 40V 260908 MJCF 与其 mesh 目录，visual geom 在 group 1；
- SOMA uniform BVH、`soma_assets`、对应的 SOMA USD 骨架；
- 已授权的人体模型目录；
- 可选：当前生产版本训练完成的 `correspondence_slots_final.npz` 与配套 `mapping.npz/json`。

## 从外部资产生成配置

在仓库根目录、已激活 UMR 环境的终端中执行。将下列路径替换为实际位置；输出建议放在已忽略的
`output/` 下。配置引用绝对路径，移动到另一台机器后应重新生成配置。

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl
export UMR_SOMA_ASSETS_PATH=/path/to/soma_assets
export UMR_SOMA_SMPLX_ASSETS_PATH="$UMR_SOMA_ASSETS_PATH"

python scripts/local_regions/configure.py \
  --robot-xml /path/to/piplus_40v_260908_no_hyperextension.xml \
  --bvh /path/to/motions_uniform/bvh/example.bvh \
  --soma-assets "$UMR_SOMA_ASSETS_PATH" \
  --soma-usd /path/to/shapes/soma_base_rig/soma_base_skel_minimal.usd \
  --smpl-models /path/to/smpl \
  --out output/local_regions/example
```

`--max-frames 12` 可用于首次试跑；默认 0 表示整段。输出为 `config.json`，动作结果路径为同目录
`motion.npz`。已有 config 不会覆盖。浮动基座 XML 由上游自动生成，资产引用应保持有效。

## 训练或复用已有对应关系

如已具备生产版本训练产物，直接设置 `SLOTS=/path/to/correspondence_slots_final.npz`，跳过训练。
如需从自身模型重新采样并训练：

```bash
python scripts/humanoid_retarget_pipeline.py \
  --config output/local_regions/example/config.json --stage build
python scripts/humanoid_retarget_pipeline.py \
  --config output/local_regions/example/config.json --stage train
SLOTS=output/local_regions/example/correspondence_slots/correspondence_slots_final.npz
```

训练过程仍由官方 pipeline 执行。重新训练存在数值差异，应重新捕获、生成分区并检查覆盖效果。
本次提取验证复用了现有生产权重，没有重新跑 500 epochs 训练。

## 捕获、施加分区并运行

```bash
python scripts/local_regions/capture.py \
  --config output/local_regions/example/config.json \
  --slots "$SLOTS" --out output/local_regions/captured

python scripts/local_regions/prepare.py \
  --mapping output/local_regions/captured/mapping.npz \
  --mapping-report output/local_regions/captured/mapping.json \
  --slots "$SLOTS" --out output/local_regions/prepared

python scripts/local_regions/retarget.py \
  --config output/local_regions/example/config.json \
  --prepared output/local_regions/prepared \
  --audit-dir output/local_regions/example/audit \
  --soma-assets "$UMR_SOMA_ASSETS_PATH"
```

也可直接将生产版本配套的 `mapping.npz/json` 传给 prepare；其中记录的机器人 XML 路径必须存在。
`prepared` 中的 `regions.npz` 与 `constrained_slots.npz` 必须作为同一套产物使用。
相同资产、模板和 slot 可复用一套 prepared 处理多个动作；每个动作应有独立配置、结果目录和审计目录。
各输出目录拒绝覆盖，失败重试时请使用新目录或自行检查后移走旧目录。

**批量运行必须调用此处 `retarget.py`**。直接调用官方 batch/pipeline 的 retarget 阶段不会注入
局部分界与源标签冻结。此分支不包含现有服务器调度服务、下载器或视频流水线。

可用官方查看器打开结果：

```bash
python scripts/visualize_robot_retarget_result.py \
  --result output/local_regions/example/motion.npz --play
```

## 检查与已做验证

审计文件包含：`region_audit.json` 的各分区点数及投影距离、`source_binding_audit.json` 的源标签
恢复数量，以及运行审计目录的 `mapping.json` / `mapping.npz`。运行时验证投影后点位误差小于
`1e-6` 米。修改资产或分界后应先检查覆盖图和短动作，再进行批量处理。

本分支提取时（2026-11-06）完成：

- 使用生产输入重新生成 regions 和 constrained_slots；所有数组与生产版本逐项完全一致。
- 运行 `reach_jump_R_002__A274_M` 前 12 帧：输出形状 `(12, 29)`，姿态均为有限数；
  右膝范围 `[0.08111, 0.10315]`、左膝 `[0.06887, 0.10192]` rad，无反屈。
- 2411 个受限点最大绑定误差 `4.051e-9` 米，恢复 1 个源分区边界标签。
- 独立捕获脚本成功导出 4096 个 slot 的映射，求解使用其中 456 个采样点。
- 三角面裁剪测试覆盖面积、几何归属、空交集及无限边界：

```bash
python -m unittest discover -s tests -p test_local_region_surface.py
```

适配器读取上游绑定函数调用者的局部变量，所以依赖此固定 UMR 版本。升级上游后必须重新检查接口、
捕获映射并做回归验证。上述短动作检查仅证明提取后的方法可执行，不代表所有高动态动作均已通过质量检查。

# 04 · 新项目与扩展计划（基于 myideal.txt）

本章把 `myideal.txt` 里的想法整理成可执行方案：如何以 autowzry 为核心重构 agent，如何规划训练阶段，以及上线前要注意的事项。

## 1. autowzry-agent 架构草案
| 层级 | 目标 | 可能的文件/模块 |
| --- | --- | --- |
| **设备层** | 统一封装 autowzry / airtest / adb 控制，屏蔽不同模拟器差异。 | `agent/device_controller.py`：对接 `autowzry.wzry_task`、adb 指令、截图接口。|
| **感知层** | 状态识别、对战检测、奖励计算。用 autowzry-lite 或现有 ppocr/onnx。 | `agent/perception.py`：提供 `get_state()`、`detect_phase()`、`calc_reward()`。|
| **策略层** | 接入不同模型（监督/强化学习、ONNX/PyTorch）。 | `agent/policy.py`：定义 `forward(state)`、`select_action()`。|
| **训练脚本** | `train_supervised.py`, `train_rl.py` 分别处理离线标注数据与在线强化训练。 | 读取统一的 `config.yaml`。|
| **配置** | 描述设备连接、分辨率、模型路径、训练超参。 | `config.yaml`。|

通过这种拆分，外部只需 `from agent import Agent; agent.step()`，方便与 autowzry 主程序组合。

## 2. 训练流程：监督 → 强化的两阶段
1. **阶段 1（监督学习）**：
   - 利用 autowzry 记录一批“屏幕截图 + 人类操作方向”的数据，存成 `dataset/{state.png, action.json}`。
   - 训练一个轻量模型（可用线性层或小型 CNN）预测“前进/后退/停留”三类动作，快速获得一个还算合理的初始策略。
   - 输出 ONNX 或 PyTorch 权重，部署到策略层，作为阶段 2 的起点。
2. **阶段 2（强化学习）**：
   - 以阶段 1 的模型为 policy_net 初始化，并把 epsilon 设为 0.2 左右（继续保留少量探索）。
   - 运行在线训练：截图 → 动作 → 回报，采用 DQN 或 PPO 方式迭代。
   - 可加入 `autowzry` 的事件回调以判断对战开始/结束，更精准地裁剪 episode。

这种“先监督、后强化”的做法能迅速得到能动的 agent，减少随机探索带来的无效样本。

## 3. 配置与文件设计示例
`config.yaml` 可以包含：
```yaml
device:
  backend: autowzry  # 或 adb
  serial: 127.0.0.1:5555
  resolution: [2400, 1080]
perception:
  start_model: models/start.onnx
  death_model: models/death.onnx
  use_autowzry_lite: false
policy:
  type: dqn
  model_path: models/move_policy.pt
  action_space: move_only
training:
  mode: rl  # 或 supervised
  batch_size: 64
  gamma: 0.95
  epsilon: 0.8
  epsilon_schedule: [[0,1.0],[10000,0.5],[50000,0.1]]
logging:
  save_dir: logs/
  screenshot_debug: true
```
这样脚本可以做到：`python train_rl.py --config config.yaml`，内部根据配置装配不同模块。

## 4. 依赖与部署建议
- **裁剪大型依赖**：
  - 把 autowzry 核心逻辑用 Nuitka 编译成 `.pyd/.so`，对外只暴露 Python API，减少开源时的代码泄露。
  - 对于 `ppocr-onnx`、`PyQt5` 等大依赖，可通过“用户自装”策略，仅在 README 中列出安装命令。
- **模块化奖励**：考虑把奖励函数拆成插件式：`reward.attack`, `reward.defense`, `reward.objective`，便于快速更换策略。
- **日志与可视化**：统一写入 `logs/episodes.jsonl`，内容包括时间戳、平均 reward、胜负，用于后续分析。
- **CI/CD**：在本地写 `tests/test_agent.py` 验证截图管线、动作映射、奖励函数，不依赖真实设备即可检查大部分问题。

## 5. 风险与待办
| 风险 | 对策 |
| --- | --- |
| 设备适配复杂（不同模拟器/真机分辨率、延迟） | 配置里强制使用相对坐标；在设备层抽象成统一 API，必要时通过模板方法适配。|
| 数据标注成本高 | 阶段 1 可先用少量录屏 + 手动标注，或利用 autowzry 自动生成“向前/向后”标签。|
| 奖励信号噪声 | 结合 autowzry 的事件检测（兵线、塔血量）提供更丰富的 reward，而不仅仅依赖 OCR。|
| 线程/进程安全 | 设计统一的 `ExperienceStore` 接口，可选择线程安全/无锁版本；必要时转为多进程 + 队列。|
| 依赖体积大 | 把可选功能（如 ONNX 推理）改成按需导入；提供 `requirements-lite.txt` 与完整版本。|

按照以上规划推进，你可以从当前仓库平滑过渡到更模块化的 autowzry-agent，同时保留既有经验。

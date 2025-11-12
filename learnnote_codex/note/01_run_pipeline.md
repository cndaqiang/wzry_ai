# 01 · 代码怎么跑

本章按“先能跑 → 再懂结构 → 最后会排错”展开，适合刚从监督学习转到强化学习的工程师。

## 1. 起步：环境与脚本入口
- **主脚本**：`train.py`（见 `train.py:1-119`）会同时启动两个线程：数据采集 (`data_collector`) 和训练 (`train_agent`)。直接在 Conda 环境中运行 `python train.py` 即可。
- **设备必需**：
  1. 保证 adb 能连上模拟器或真机，运行 `scrcpy-win64-v2.0/adb.exe connect 127.0.0.1:5555`，与 `argparses.py:76-80` 中的默认参数一致。
  2. `AndroidTool` 需要窗口标题或 airtest 配置；默认 `args.window_title='MuMu安卓设备'`，如你用的是其它模拟器，请在 `argparses.py` 里改掉。
  3. 如果你启用 `airtest_mobileauto`，把 `airtest_config` 指向 `config.example.yaml` 或自己的 YAML，这样 `AndroidTool` 会走 autowzry 封装。
- **第一次运行的自检**：
  1. 先单独运行 `python - <<'PY' ... tool = AndroidTool(); print(tool.screenshot_window().shape)` 检查截图是否成功。
  2. 把 `tool.show_scrcpy()` 注释取消，确认 scrcpy 窗口已弹出。
  3. 若计划训练，确保 `models/start.onnx` 与 `models/death.onnx` 在 `models/` 目录下。

## 2. 整体流程（train.py）
1. **初始化模块**：创建 `AndroidTool`、`Environment`、`DQNAgent`、奖励工具 `GetRewordUtil`、全局状态 `GlobalInfo`。
2. **数据采集线程** `data_collector()`：
   - 截图 `tool.screenshot_window()` → 检查是否进入对局（`OnnxRunner('models/start.onnx')` 或 `tool.autowzry.判断对战中()`）。
   - 循环执行：
     1. `agent.select_action(state)` 生成动作（epsilon-greedy）。
     2. `env.step(action)` 调 adb 操作，并拿下一帧、奖励、done。
     3. `globalInfo.store_transition_dqn(...)` 把 `(state, action, reward, next_state, done)` 写入经验池。
3. **训练线程** `train_agent()`：
   - 当经验池 ≥ batch size（`globalInfo.is_memory_bigger_batch_size_dqn()`）时，调用 `agent.replay()` 进行一次 DQN 更新。
   - 每 `args.num_episodes` 次训练保存模型到 `src/wzry_ai.pt`。

可参考下图式顺序：
```
截图 -> 判断对局 -> 选择动作 -> ADB 操作 -> 下一帧 + 奖励 -> 存经验 -> (异步) 采样 batch -> Bellman 更新 -> 更新 epsilon
```

## 3. 模块职责速查
| 模块 | 作用 | 常用入口 |
| --- | --- | --- |
| `android_tool.py` | 设备控制、截图、动作执行、airtest/autowzry 适配。 | `AndroidTool.action_move/info/attack`、`screenshot_window`。|
| `wzry_env.py` | 对环境交互做薄封装：调用 `AndroidTool` 执行动作，再拿奖励。 | `Environment.step(action)`。|
| `getReword.py` | 奖励函数：用 OCR (`TextSystem`) + onnx 死亡检测 + HSV/攻击条检测计算 reward/done。 | `GetRewordUtil.get_reword(image,isFrame,action)`。|
| `dqnAgent.py` | 包含策略网络、目标网络、经验回放、epsilon 策略、Bellman 更新。 | `DQNAgent.select_action`、`DQNAgent.replay`。|
| `net_actor.py` | 卷积主体 + 多头线性层，输出八个动作维度的 Q 值。 | `NetDQN.forward(x)`。|
| `memory.py` | 循环缓冲区 + 随机采样。 | `ReplayMemory.push/sample`。|
| `globalInfo.py` | 单例全局状态：对局标记、三种算法的经验池、文件锁写训练数据。 | `GlobalInfo.store_transition_dqn` 等。

要单独调试模块时，可以：
1. 在 Python REPL 中 `from getReword import GetRewordUtil; img = cv2.imread('xxx'); print(GetRewordUtil().get_reword(img, True, action))` 验证 reward。
2. `agent = DQNAgent(); dummy = np.zeros((640,640,3), dtype=np.uint8); print(agent.select_action(dummy))` 检查网络输出。

## 4. 常见问题与排查
| 症状 | 排查步骤 |
| --- | --- |
| **截图返回 None** | 确认 scrcpy 窗口标题与 `args.window_title` 一致；使用 airtest 时，确保 `config.example.yaml` 正确指向设备；在 `AndroidTool.screenshot_window` 打印异常信息。|
| **对局一直判未开始** | `models/start.onnx` 是否存在；若改用 autowzry 的检测函数，先确保它在 airtest 配置里启用。可以临时注释 `if tool.autowzry...`，直接设定 `globalInfo.set_game_start()` 进入循环进行压测。|
| **训练线程不跑** | 在 `train_agent()` 添加 `print(len(globalInfo.dqn_memory))` 确认经验数量；调小 `batch_size` 以验证训练流程是否正常；确保 `globalInfo` 单例初始化的 `buffer_capacity` 足够大。|
| **GPU/CPU 设备冲突** | `argparses.py:81-88` 控制 `device_id`；如果没有 GPU，自动切到 CPU，但要把 batch size 调小以免显存不足。|
| **模型加载失败** | `args.model_path` 默认 `src/wzry_ai.pt`。若首次运行本地没有文件，请把该参数设为空或在 `dqnAgent.py` 中捕获异常。|
| **动作执行卡死** | `AndroidTool` 中有 `self.onlymove` 可在调试时屏蔽 info/attack；adb 命令执行慢可以试着减少 `ThreadPoolExecutor` 的任务或开启 airtest 模式。|

## 5. 进一步的实践建议
1. **先录制一小段数据**：让采集线程跑几分钟，确认 `training_data.json` 或日志输出的 reward 没问题，再开启长时间训练。
2. **分级调试**：
   - 阶段 1：只运行 `data_collector`，把 `training_thread` 注释掉，确保动作执行和奖励 OK。
   - 阶段 2：写一个脚本对 `ReplayMemory` 压测（参考 `note/02_python_patterns.md`）。
   - 阶段 3：将 epsilon 固定在 0（纯网络动作）或 1（纯随机）测试极端情况，观察 reward 变化。|
3. **记录日志**：在 `agent.replay()` 里打印 loss、epsilon、平均 reward，方便判断模型是否学习；也可将 `GlobalInfo.update_data_file` 输出的 JSON 做可视化。
4. **自动重连设备**：`train.py:71-72` 已在 `connect_status()` 为 False 时调用 `tool.移动后连接设备()`，确保该函数在 `AndroidTool` 中实现并可稳定重连。

把这些流程跑通后，再去读 02/03 章的细节会更容易吸收。

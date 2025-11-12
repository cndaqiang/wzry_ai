# 02 · Python 技巧与项目模式

目标：把本项目中出现的关键 Python 语法/模式讲清楚，告诉你它解决什么问题、如何亲手试，以及在别的工程里怎么复用。

## 1. ReplayMemory：经验回放容器
- **是什么**：`memory.py:8-33` 定义的循环缓冲区，保存 `(state, action, reward, next_state, done)`。在强化学习里它用来打乱时间关联，让训练更稳定。
- **最小示例**：
```python
from memory import ReplayMemory, Transition
mem = ReplayMemory(capacity=3)
mem.push('s0','a0',0,'s1',False)
mem.push('s1','a1',1,'s2',True)
mem.push('s2','a2',0,'s3',False)
mem.push('s3','a3',1,'s4',True)  # 覆盖最旧样本
batch = mem.sample(2)
print(batch)
```
- **使用注意**：
  1. 容量设置过小会导致样本重复、过拟合；默认 10000，可按显存/内存调整。
  2. `random.sample` 要求当前样本数 ≥ batch_size，记得在采样前先检查（`globalInfo.is_memory_bigger_batch_size_dqn()`）。
  3. 如果你要并发写入，可像 `globalInfo` 那样加锁。

## 2. `@singleton`：全局状态共享
- **定义位置**：`globalInfo.py:13-33`。
- **为什么要用**：采集线程和训练线程都要访问经验池，如果各自实例化 GlobalInfo，会造成两个不同的 ReplayMemory。单例保证全局只有一份状态。
- **如何扩展**：
  - 若要新增“训练指标缓存”之类的字段，在 `GlobalInfo.__init__` 添加成员，并通过 `set_value/get_value` 读写。
  - 单例也意味着你必须注意线程安全：`store_transition_ppo` 使用 `threading.Lock`，而 DQN 分支为了速度没有锁。如果你发现多线程写入冲突，可仿照 PPO 的写法加锁。

## 3. 图像处理管线：Win32 + PyQt + OpenCV
- **截图**：`android_tool.py:15-70` 通过 win32 句柄或 airtest 获取窗口图像，若启用 PyQt，需要先创建 `QApplication`。
- **处理**：
  - `dqnAgent.py:67-73` 的 `preprocess_image` 用 `cv2.resize` + `torch.from_numpy` + `permute` 把 BGR 图像变成 `[C,H,W]` 浮点张量。
  - `getReword.py:28-92` 中用 HSV 空间分割红色攻击条：`cv2.cvtColor -> cv2.inRange -> bitwise_and`，最后统计面积。
- **新手建议**：
  1. 先用 `cv2.imwrite('debug.jpg', state)` 保存截图，肉眼确认区域坐标是否正确。
  2. 如果分辨率变了，记得同步更新 `argparses.move_actions_detail` 中的相对坐标，否则 ADB 点击位置会偏移。
  3. HSV 阈值依赖色彩，建议在调试阶段开放一个可配置文件，把 `lower_bound/upper_bound` 写进去方便调优。

## 4. 并发：Thread 与 ThreadPoolExecutor
- **线程**：`train.py:115-119` 使用 `threading.Thread` 启动训练线程。由于主要是张量计算（CPU/GPU），GIL 不是瓶颈，且训练和采集分离可以防止阻塞。
- **线程池**：`getReword.py:177-205` 用 `ThreadPoolExecutor` 并行执行 OCR、死亡检测、攻击条识别，缩短奖励计算延迟。
- **经验**：
  1. Python 的 GIL 让纯 CPU 计算线程无法并行，但这里的线程调度主要在 IO/等待 adb 指令期间，所以收益明显。
  2. 如果你的奖励函数改成重型推理，建议把 `ThreadPoolExecutor` 换成 `ProcessPoolExecutor` 或单独的服务，否则 GIL 会拖慢推理。
  3. 线程池的 `as_completed` 可以捕获异常，实际使用时最好加 try/except 打印错误，避免 reward 返回空导致训练崩溃。

## 5. 其他高频技巧
- **argparse 统一配置**：在 `argparses.py` 里集中声明训练/设备参数，其他模块只需要 `from argparses import args`。新手常见问题是“修改了参数却没生效”——记得重新启动脚本，因为解析发生在 import 阶段。
- **分模块导入**：`DQNAgent` 里只在需要时才把图像转换到 `device` 上，避免在 CPU 上占用多份内存。你在别的项目里也可以遵循“处理函数尽量无状态，参数由构造函数注入”的写法。
- **命名元组 Transition**：`Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))` 让 batch 拆包更清晰。如果要加字段（例如 `info`），只要在命名元组里追加即可，但记得更新所有使用位置。
- **文件锁 FileLock**：`globalInfo.update_data_file` 用 `FileLock` 避免多个进程写同一个 JSON。你要输出可视化数据时，也可以用同样套路。

把这些模式掌握后，你就可以更从容地阅读 03 章的强化学习内容，并将代码迁移到自己的工程里。

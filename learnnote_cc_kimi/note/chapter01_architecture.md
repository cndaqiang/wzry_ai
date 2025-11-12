# 第一章：项目架构与文件功能分析

## 1.1 项目整体架构概览

本项目是一个基于深度强化学习的王者荣耀AI训练系统，采用Deep Q-Network (DQN) 算法让AI学会控制游戏角色。项目采用模块化设计，代码结构清晰，便于理解和扩展。

### 核心架构图

```
训练主循环 (train.py)
├── 数据收集线程 (data_collector)
│   ├── 截图获取游戏状态
│   ├── 动作选择 (随机/网络预测)
│   ├── 执行动作并获取反馈
│   └── 存储训练经验
└── 训练线程 (train_agent)
    ├── 从经验池采样数据
    ├── 计算损失函数
    ├── 更新网络参数
    └── 定期保存模型

核心组件层
├── DQN智能体 (dqnAgent.py) - 强化学习算法实现
├── 神经网络 (net_actor.py) - CNN模型定义
├── 游戏环境 (wzry_env.py) - 动作执行与状态获取
└── 奖励系统 (getReword.py) - 奖励计算

基础设施层
├── Android工具 (android_tool.py) - 设备控制
├── 经验回放 (memory.py) - 数据存储管理
├── 参数配置 (argparses.py) - 全局参数
└── ONNX推理 (onnxRunner.py) - 游戏状态检测
```

## 1.2 核心训练文件功能解析

### train.py - 训练主入口

**核心作用**：协调整个训练过程，采用双线程架构实现数据收集与网络训练的并行执行。

**关键代码解析**：

```python
# 双线程架构设计
training_thread = threading.Thread(target=train_agent)  # 训练线程
training_thread.start()
data_collector()  # 数据收集线程（主线程）
```

**数据收集线程工作流程**：

```python
# 1. 检测游戏状态
tool.autowzry.判断对战中()  # 使用airtest库检测是否在战斗中

# 2. 获取当前动作
action = agent.select_action(state)  # 通过DQN网络或随机选择动作

# 3. 执行动作并获取反馈
next_state, reward, done, info = env.step(action)

# 4. 存储训练经验
globalInfo.store_transition_dqn(state, action, reward, next_state, done)
```

**训练线程工作流程**：

```python
# 等待经验池有足够数据
if not globalInfo.is_memory_bigger_batch_size_dqn():
    time.sleep(1)
    continue

# 执行网络训练
agent.replay()  # 从经验池采样并更新网络

# 定期保存模型
if count % args.num_episodes == 0:
    agent.save_model('src/wzry_ai.pt')
```

**作者思考亮点**：
- 使用`autowzry.判断对战中()`替代ONNX模型检测，提高效率和准确率
- 详细注释解释了DQN的核心思想：模型预测Q值，通过Q值与实际奖励的差值进行学习
- 双线程设计避免数据采集与训练相互阻塞

### dqnAgent.py - DQN算法实现

**核心作用**：实现Deep Q-Network算法，负责动作选择、网络训练和参数管理。

**关键设计解析**：

**动作空间定义**：
```python
self.action_sizes = [2, 360, 9, 11, 3, 360, 100, 5]
# 对应含义：
# [是否移动, 移动角度, 信息操作, 攻击对象, 动作类型, 参数1角度, 参数2距离, 参数3时长]
```

**双网络架构**：
```python
self.policy_net = NetDQN().to(self.device)  # 实时更新，用于动作选择
self.target_net = NetDQN().to(self.device)  # 延迟更新，用于计算目标Q值
```

**动作选择策略**（Epsilon-Greedy）：
```python
if rand <= self.epsilon:
    # 随机探索：返回随机动作
    return [np.random.randint(size) for size in self.action_sizes]
else:
    # 利用学习：通过网络预测最优动作
    q_values = self.policy_net(tmp_state_640_640)
    return [np.argmax(q.detach().cpu().numpy()) for q in q_values]
```

**损失函数计算**（核心训练逻辑）：
```python
# 计算当前Q值（实际执行动作的Q值）
state_action_q_values = move_action_q.gather(1, batch_action[:, 0].unsqueeze(1)) + ...

# 计算目标Q值（Bellman方程）
next_state_values[non_final_mask] = torch.max(next_move_action_q, 1)[0] + ...
expected_state_action_values = batch_reward + self.gamma * next_state_values * (1 - batch_done)

# 计算损失（MSE）
loss = self.criterion(state_action_q_values, expected_state_action_values.unsqueeze(1))
```

**作者深度思考**：
- 详细解释了Q值学习的本质：模型学习预测每个动作的"得分"
- 阐明了目标网络的作用：提高训练稳定性，避免目标值频繁变化
- 解释了gamma参数的影响：平衡当前奖励与未来奖励的重要性

## 1.3 神经网络模型架构

### net_actor.py - NetDQN网络结构

**核心作用**：定义CNN网络结构，将游戏截图转换为动作Q值预测。

**网络架构详解**：

```python
class NetDQN(nn.Module):
    def __init__(self):
        # 卷积层：提取视觉特征
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4)  # 640x640 -> 159x159
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)  # 159x159 -> 78x78

        # 全连接层：特征到动作Q值映射
        self.fc = nn.Linear(conv_output_size, 256)  # 特征提取
        self.fc1 = nn.Linear(256, 256)  # 特征转换

        # 8个独立输出分支，每个对应一个动作维度
        self.fc_move = nn.Linear(256, 2)      # 移动决策Q值
        self.fc_angle = nn.Linear(256, 360)   # 移动角度Q值
        self.fc_info = nn.Linear(256, 9)      # 信息操作Q值
        self.fc_attack = nn.Linear(256, 11)   # 攻击操作Q值
        # ... 其他分支
```

**前向传播流程**：
```python
def forward(self, x):
    # 1. CNN特征提取
    x = F.relu(self.conv1(x))  # [batch, 64, 159, 159]
    x = F.relu(self.conv2(x))  # [batch, 128, 78, 78]
    x = x.view(x.size(0), -1)  # 展平 -> [batch, 特征总数]

    # 2. 全连接特征转换
    x = self.fc(x)             # -> [batch, 256]
    x = F.relu(self.fc1(x))    # -> [batch, 256]

    # 3. 多分支Q值输出
    move_action_q = self.fc_move(x)    # -> [batch, 2]
    angle_q = self.fc_angle(x)         # -> [batch, 360]
    # ... 其他分支

    return move_action_q, angle_q, info_action_q, attack_action_q, ...
```

**技术亮点**：
- **动态卷积输出计算**：`_get_conv_output_size`方法自动计算卷积层输出尺寸
- **多分支设计**：针对不同动作类型设计独立输出层，避免相互干扰
- **Xavier初始化**：使用合适的权重初始化策略，保证训练稳定性

## 1.4 游戏交互接口设计

### android_tool.py - 设备控制核心

**核心作用**：封装Android设备控制功能，提供统一的截图和动作执行接口。

**双模式支持**：
```python
def __init__(self, airtest_config=""):
    if len(self.airtest_config) > 0:
        # Airtest模式：更精确的状态检测和控制
        self.airtest_init()
    else:
        # ADB模式：传统的ADB命令控制
        self.scrcpy_dir = scrcpy_dir
```

**坐标系统设计**：
```python
def calculate_endpoint(self, start_point, radius, angle):
    # 基于16:9屏幕比例的相对坐标计算
    # 确保在不同分辨率设备上都能正常工作
    angle_rad = math.radians(angle)
    end_x = int(start_point[0] + radius * math.cos(angle_rad))
    end_y = int(start_point[1] - radius * math.sin(angle_rad))
    return end_x, end_y
```

**异步执行机制**：
```python
self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

def execute_move(self, task_params):
    # 异步执行移动操作，避免阻塞训练流程
    self.executor.submit(self._execute_move_internal, task_params)
```

**动作执行逻辑**：
- **移动操作**：基于角度计算滑动终点，模拟摇杆控制
- **点击操作**：预设功能按钮坐标，执行精确点击
- **滑动操作**：根据参数计算滑动轨迹，实现技能释放

### wzry_env.py - 环境封装

**核心作用**：实现标准强化学习环境接口，封装状态获取和动作执行逻辑。

**环境接口实现**：
```python
def step(self, action):
    # 1. 解析动作参数
    move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3 = action

    # 2. 执行具体动作
    self.tool.execute_action(action)

    # 3. 获取新状态
    next_state = self.tool.screenshot_window()

    # 4. 计算奖励
    reward = self.rewordUtil.calculate_reword(self.state, action, next_state)

    # 5. 判断游戏是否结束
    done = self.rewordUtil.is_game_over(next_state)

    return next_state, reward, done, info
```

## 1.5 工具类组件功能

### memory.py - 经验回放机制

**核心作用**：实现循环缓冲区，存储和采样训练经验。

**数据结构定义**：
```python
Transition = namedtuple('Transition',
                       ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)  # 循环缓冲区

    def push(self, *args):
        # 存储一条transition经验
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # 随机采样一批经验用于训练
        return random.sample(self.memory, batch_size)
```

**设计优势**：
- **循环缓冲区**：自动管理内存，避免无限增长
- **随机采样**：打破时间相关性，提高训练稳定性
- **固定大小**：控制内存使用，支持大规模训练

### getReword.py - 多维度奖励计算

**核心作用**：基于游戏状态计算奖励信号，指导AI学习正确策略。

**奖励设计策略**：
```python
def calculate_reword(self, state, action, next_state):
    reward = 0

    # 1. 基础状态奖励（胜利/失败/死亡）
    reward += self.get_base_state_reward(next_state)

    # 2. 攻击状态奖励（通过图像识别判断是否攻击）
    reward += self.get_attack_state_reward(next_state)

    # 3. 移动方向奖励（鼓励向前移动）
    reward += self.get_move_direction_reward(action)

    return reward
```

**并行状态检测**：
```python
with ThreadPoolExecutor(max_workers=3) as executor:
    # 并行检测多种游戏状态，提高效率
    victory_future = executor.submit(self.check_victory, next_state)
    defeat_future = executor.submit(self.check_defeat, next_state)
    death_future = executor.submit(self.check_death, next_state)
```

**技术亮点**：
- **多维度奖励**：结合游戏结果、战斗状态、移动策略
- **图像识别**：使用HSV颜色空间识别攻击状态条
- **并行处理**：同时检测多种状态，提高检测效率

## 1.6 配置参数说明

### argparses.py - 全局参数管理

**核心作用**：集中管理所有可调参数，便于实验和调优。

**关键参数分类**：

**训练参数**：
```python
parser.add_argument('--batch_size', type=int, default=64, help='训练批次大小')
parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
parser.add_argument('--gamma', type=float, default=0.95, help='奖励折扣因子')
```

**DQN参数**：
```python
parser.add_argument('--epsilon', type=float, default=1.5, help='初始探索率')
parser.add_argument('--epsilon_decay', type=float, default=0.995, help='探索率衰减')
parser.add_argument('--epsilon_min', type=float, default=0.01, help='最小探索率')
parser.add_argument('--target_update', type=int, default=10, help='目标网络更新频率')
```

**系统参数**：
```python
parser.add_argument('--device', type=str, default='cuda', help='计算设备')
parser.add_argument('--model_path', type=str, default='src/wzry_ai.pt', help='模型保存路径')
parser.add_argument('--window_title', type=str, default='MuMu安卓设备', help='模拟器窗口标题')
```

**参数调优建议**：
- **学习率**：建议从0.001开始，训练不稳定时调小
- **Batch Size**：根据GPU内存调整，一般32-128之间
- **Gamma值**：平衡当前与未来奖励，游戏类建议0.9-0.99
- **Epsilon衰减**：控制探索速度，建议0.995-0.999

## 1.7 小结

本章详细分析了项目的文件架构和功能模块，重点包括：

1. **train.py的双线程设计**：数据收集与训练分离，提高效率和稳定性
2. **dqnAgent.py的算法实现**：完整的DQN算法，包含探索策略和损失计算
3. **net_actor.py的网络结构**：CNN特征提取加多分支输出，适配复杂动作空间
4. **android_tool.py的设备控制**：支持ADB和Airtest双模式，兼容性强
5. **工具类组件**：经验回放、奖励计算等核心功能的实现

理解这些文件的功能和相互关系，为后续深入学习训练流程和算法原理打下基础。每个模块都经过精心设计，注释详尽，非常适合新手学习和实践强化学习项目开发。
# 第二章：训练流程代码级详解

## 2.1 主训练循环启动过程

### 程序入口分析

让我们从`train.py`的第一行开始，详细理解程序是如何启动的：

```python
# 导入必要的库
import threading
import time
import cv2
import numpy as np
from android_tool import AndroidTool
from argparses import args
from dqnAgent import DQNAgent
from getReword import GetRewordUtil
from globalInfo import GlobalInfo
from wzry_env import Environment
from onnxRunner import OnnxRunner
```

**新手注意**：这些导入语句展示了项目依赖的核心组件，每个都有其专门职责：
- `threading`：实现并发编程
- `cv2`：图像处理库OpenCV
- `DQNAgent`：强化学习智能体
- `AndroidTool`：设备控制接口

### 全局对象初始化

```python
# 创建全局状态管理器
globalInfo = GlobalInfo()

# 游戏开始状态检测器
class_names = ['started']
start_check = OnnxRunner('models/start.onnx', classes=class_names)

# 奖励计算工具
rewordUtil = GetRewordUtil()

# Android设备控制工具（关键配置）
airtest_config = "config.example.yaml"
tool = AndroidTool(airtest_config=airtest_config)
```

**重要理解**：这里的初始化顺序很重要，因为后面的组件依赖前面的组件：
1. `GlobalInfo`管理全局状态，必须最先创建
2. `AndroidTool`是硬件接口，需要正确配置
3. 使用airtest配置可以获得更好的游戏状态检测

### 核心组件实例化

```python
# 获取初始状态截图
state = tool.screenshot_window()
tool.show_scrcpy()  # 显示设备画面

# 创建环境封装器
env = Environment(tool, rewordUtil)

# 创建DQN智能体（核心算法）
agent = DQNAgent()
```

**关键概念**：
- **状态（state）**：游戏当前画面的截图，是AI决策的依据
- **环境（env）**：封装了游戏交互的逻辑，提供标准接口
- **智能体（agent）**：包含神经网络和训练算法的核心

## 2.2 双线程架构详解

### 线程启动逻辑

```python
if __name__ == '__main__':
    # 创建训练线程（后台运行）
    training_thread = threading.Thread(target=train_agent)
    training_thread.start()

    # 主线程运行数据收集
    data_collector()
```

**架构优势**：
- **数据收集线程**：持续监控游戏状态，不遗漏任何训练机会
- **训练线程**：只要有足够数据就开始训练，提高样本利用率
- **解耦合**：两个线程独立运行，互不影响

### 数据收集线程详细流程

让我们深入`data_collector()`函数的每一行：

```python
def data_collector():
    while True:  # 无限循环，持续收集数据
        # 获取当前游戏画面
        state = tool.screenshot_window()

        # 确保截图成功
        if state is None:
            time.sleep(0.01)  # 等待10ms重试
            continue
```

**新手要点**：
- 使用`while True`实现持续监控，这是实时系统的常见模式
- 截图失败处理很重要，避免程序崩溃
- `time.sleep(0.01)`避免CPU占用过高

### 游戏状态检测

```python
        # 初始化对局状态
        globalInfo.set_game_end()

        # 判断对局是否开始（关键优化）
        if tool.autowzry.判断对战中():
            print("-------------------------------对局开始-----------------------------------")
            globalInfo.set_game_start()
```

**技术演进**：作者从ONNX模型检测升级到airtest库检测：
```python
# 旧方法：基于ONNX模型
checkGameStart = start_check.get_max_label(state)

# 新方法：基于airtest库（更可靠）
tool.autowzry.判断对战中()
```

### 单局训练循环

当检测到游戏开始后，进入单局训练循环：

```python
            # 对局开始了，进行训练
            while globalInfo.is_start_game():
                # 1. 获取预测动作
                print("---> 获取预测动作")
                action = agent.select_action(state)
                print(f"---> env.step(action)={action}")
```

**动作选择过程**：
1. **输入处理**：将640x640截图转换为神经网络输入格式
2. **网络预测**：通过CNN提取特征，输出各动作的Q值
3. **动作决策**：根据epsilon-greedy策略选择最终动作

### 动作执行与环境交互

```python
                # 2. 执行动作并获取环境反馈
                next_state, reward, done, info = env.step(action)
                print("---> reward")
                print(info, reward)
```

**env.step()内部流程**：
1. **动作解析**：将8维动作向量分解为具体操作
2. **动作执行**：调用AndroidTool执行点击、滑动等操作
3. **状态获取**：截图获取新的游戏画面
4. **奖励计算**：基于新旧状态变化计算奖励值
5. **结束判断**：检测游戏是否结束（胜利/失败/死亡）

### 经验存储机制

```python
                # 3. 存储训练经验（关键设计）
                globalInfo.store_transition_dqn(state, action, reward, next_state, done)

                # 4. 状态更新
                state = next_state
```

**经验数据结构**：
```python
# 每条经验包含5个元素
(state, action, reward, next_state, done)

# 具体例子
state: 当前游戏截图 (640x640x3)
action: [1, 45, 0, 5, 1, 45, 50, 2]  # 8维动作向量
reward: -1  # 当前步的奖励值
next_state: 执行动作后的新截图
done: 0  # 游戏是否结束（1结束，0继续）
```

## 2.3 训练线程详细流程

### 训练触发条件

```python
def train_agent():
    count = 1
    while True:
        # 等待经验池有足够数据
        if not globalInfo.is_memory_bigger_batch_size_dqn():
            time.sleep(1)
            continue
        print("training")
```

**设计原理**：
- 经验池数据量必须≥batch_size才开始训练
- 避免早期数据不足时的无效训练
- 使用sleep避免CPU空转

### 核心训练过程

```python
        # 执行网络训练（核心算法）
        agent.replay()

        # 定期保存模型
        if count % args.num_episodes == 0:
            agent.save_model('src/wzry_ai.pt')
        count = count + 1
```

**训练频率控制**：
- 每次循环都执行训练，保证及时更新
- 定期保存模型，防止训练中断丢失
- 计数器循环使用，避免整数溢出

## 2.4 动作选择机制详解

### Epsilon-Greedy策略实现

让我们深入`agent.select_action()`方法：

```python
def select_action(self, state):
    # 生成随机数用于探索决策
    rand = np.random.rand()
    print(f"-->Random number for epsilon-greedy: {rand}, Epsilon: {self.epsilon}")

    # 探索：随机选择动作
    if rand <= self.epsilon:
        return [np.random.randint(size) for size in self.action_sizes]
```

**探索机制详解**：
- **epsilon值**：初始为1.5，表示极高的探索率
- **随机选择**：每个动作维度独立随机选择
- **重要性**：保证AI能充分探索动作空间，避免陷入局部最优

### 网络预测流程

```python
    else:
        # 利用：通过网络预测最优动作
        # 1. 图像预处理
        tmp_state_640_640 = self.preprocess_image(state).unsqueeze(0)

        # 2. 网络评估模式
        self.policy_net.eval()
        with torch.no_grad():
            # 3. 前向传播获取Q值
            q_values = self.policy_net(tmp_state_640_640)

        # 4. 选择每个维度Q值最大的动作
        return [np.argmax(q.detach().cpu().numpy()) for q in q_values]
```

**图像预处理过程**：
```python
def preprocess_image(self, image, target_size=(640, 640)):
    # 1. 调整图像大小
    resized_image = cv2.resize(image, target_size)

    # 2. 转换维度顺序 [H,W,C] -> [C,H,W]
    tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1)

    # 3. 移动到计算设备（CPU/GPU）
    return tensor_image.to(device)
```

**网络输出解读**：
```python
# 网络返回8个Q值矩阵，每个对应一个动作维度
move_action_q: [1, 2]      # 移动决策的Q值（移动/不移动）
angle_q: [1, 360]          # 移动角度的Q值（0-359度）
info_action_q: [1, 9]      # 信息操作的Q值（9种操作）
attack_action_q: [1, 11]   # 攻击对象的Q值（11种攻击目标）
# ... 其他维度

# 动作选择：每个维度选择Q值最大的索引
[ np.argmax(move_action_q), np.argmax(angle_q), ... ]
```

## 2.5 网络训练过程详解

### 经验采样机制

```python
def replay(self):
    # 从经验池随机采样一个batch的数据
    transitions = globalInfo.random_batch_size_memory_dqn()
    batch = Transition(*zip(*transitions))
```

**采样过程**：
1. **随机选择**：从10000条经验中随机选择64条
2. **数据重组**：将列表转换为张量格式
3. **设备转移**：将数据移动到GPU（如果可用）

### 数据预处理

```python
    # 批量图像预处理
    batch_state = torch.stack([self.preprocess_image(state) for state in batch.state]).to(device)
    batch_action = torch.LongTensor(batch.action).to(self.device)
    batch_reward = torch.FloatTensor(batch.reward).to(self.device)
    batch_next_state = torch.stack([self.preprocess_image(state) for state in batch.next_state]).to(device)
    batch_done = torch.FloatTensor(batch.done).to(self.device)
```

**关键操作**：
- **stack操作**：将多个单张图像堆叠成batch张量
- **数据类型转换**：确保数据类型与网络要求匹配
- **设备一致性**：所有数据在同一设备上计算

### Q值计算过程

#### 当前Q值计算（实际执行动作的Q值）

```python
# 获取网络对所有动作的Q值预测
state_action_values = self.policy_net(batch_state)
move_action_q, angle_q, info_action_q, attack_action_q, ... = state_action_values

# 提取实际执行动作的Q值（关键步骤）
state_action_q_values = move_action_q.gather(1, batch_action[:, 0].unsqueeze(1)) + \
                       angle_q.gather(1, batch_action[:, 1].unsqueeze(1)) + \
                       info_action_q.gather(1, batch_action[:, 2].unsqueeze(1)) + \
                       ...
```

**gather操作解释**：
```python
# gather的作用：从Q值矩阵中提取特定动作的Q值
# move_action_q形状: [batch_size, 2]（2个动作的Q值）
# batch_action[:, 0]: [batch_size]（实际选择的动作索引）
# gather结果: [batch_size, 1]（实际动作的Q值）

# 举例：
move_action_q = [[0.5, 0.8], [0.3, 0.9]]  # 2个样本的Q值
batch_action[:, 0] = [1, 0]               # 实际选择的动作
gather结果 = [[0.8], [0.3]]               # 实际动作的Q值
```

#### 目标Q值计算（Bellman方程）

```python
# 1. 筛选非终止状态
non_final_mask = (batch_done == 0)
non_final_next_states = batch_next_state[non_final_mask]

# 2. 使用目标网络计算下一状态的最大Q值
if non_final_next_states.size(0) > 0:
    next_state_action_values = self.target_net(non_final_next_states)
    next_move_action_q, next_angle_q, ... = next_state_action_values

    # 计算每个维度的最大Q值并相加
    next_state_values[non_final_mask] = torch.max(next_move_action_q, 1)[0] + \
                                       torch.max(next_angle_q, 1)[0] + \
                                       torch.max(next_info_action_q, 1)[0] + \
                                       ...

# 3. 计算目标Q值（Bellman方程）
expected_state_action_values = batch_reward + self.gamma * next_state_values * (1 - batch_done)
```

**Bellman方程详解**：
```python
# 目标Q值 = 即时奖励 + 折扣因子 × 下一状态最大Q值
# Q(s,a) = r + γ × max Q(s', a')

# 代码实现：
expected_q = reward + gamma * max_next_q * (1 - done)

# done=1时（游戏结束）：expected_q = reward
# done=0时（游戏继续）：expected_q = reward + gamma * max_next_q
```

### 损失计算与参数更新

```python
# 计算均方误差损失
loss = self.criterion(state_action_q_values, expected_state_action_values.unsqueeze(1))

# 梯度清零
self.optimizer.zero_grad()

# 反向传播
loss.backward()

# 参数更新
self.optimizer.step()
```

**训练目标**：
```python
# 最小化预测Q值与目标Q值的差异
# loss = (Q_predicted - Q_target)²

# 通过不断减小这个差异，网络学会准确预测每个动作的"价值"
# 从而能够在给定状态下选择最优动作
```

### 探索率更新

```python
# epsilon衰减：逐渐减少探索，增加利用
if self.epsilon > self.epsilon_min:
    self.epsilon *= self.epsilon_decay
```

**衰减策略**：
- **初始值**：1.5（高探索率，充分尝试各种动作）
- **衰减率**：0.995（每步衰减0.5%）
- **最小值**：0.01（保持最低探索，避免完全贪婪）

### 目标网络更新

```python
# 定期同步目标网络参数
if self.steps_done % self.target_update == 0:
    self.target_net.load_state_dict(self.policy_net.state_dict())
```

**设计原理**：
- **稳定性**：目标网络参数保持稳定，避免训练目标频繁变化
- **延迟更新**：每10步更新一次，平衡稳定性和实时性
- **软更新**：直接复制参数，简单有效

## 2.6 训练过程监控与调试

### 关键信息输出

程序运行时会输出关键调试信息：

```python
print("-------------------------------对局开始-----------------------------------")
print("---> 获取预测动作")
print(f"---> env.step(action)={action}")
print("---> reward")
print(info, reward)
print("loss", loss)
```

**调试要点**：
- **动作输出**：验证动作是否在合理范围内
- **奖励信号**：检查奖励计算是否正确
- **损失变化**：监控训练是否正常进行

### 训练状态判断

**正常训练的特征**：
1. epsilon值逐渐减小（从1.5降到0.01）
2. 损失值有波动但总体趋势下降
3. 动作从完全随机逐渐变得有策略性

**常见问题识别**：
1. **损失不下降**：学习率过高或网络结构问题
2. **epsilon不更新**：衰减参数设置错误
3. **动作为NaN**：网络输出异常，需要检查输入数据

## 2.7 小结

本章详细解析了训练流程的每个环节：

1. **双线程架构**：数据收集与训练分离，提高效率和稳定性
2. **动作选择机制**：epsilon-greedy策略平衡探索与利用
3. **网络训练过程**：从经验采样到参数更新的完整流程
4. **Bellman方程实现**：目标Q值计算的核心算法
5. **调试监控方法**：如何判断训练是否正常进行

理解这些流程细节，有助于：
- 快速定位训练中的问题
- 优化训练参数和超参数
- 开发新的算法改进
- 复现和调整实验结果

下一章我们将从监督学习的角度，深入理解强化学习的核心概念和思维转换。,
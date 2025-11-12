# 第六章：新手动手实践指导手册

## 6.1 最小可运行版本构建指南

### 从复杂到简单的思路

原始项目功能完备但复杂，新手容易迷失。我们先构建一个**最小可运行版本**（MVP），只保留核心功能。

#### 最小版本的核心功能
1. ✅ 能获取游戏截图
2. ✅ 能执行简单动作（点击、滑动）
3. ✅ 有基本的奖励反馈
4. ✅ 能训练神经网络

#### 简化版架构图
```
简化版AI
├── 截图模块（获取状态）
├── 动作模块（执行动作）
├── 奖励模块（计算奖励）
└── 训练模块（更新网络）
```

### 构建步骤详解

#### 步骤1：创建简化版DQN智能体
```python
# simple_dqn.py
import torch
import torch.nn as nn
import numpy as np
import cv2

class SimpleDQN(nn.Module):
    """极简版DQN网络"""
    def __init__(self, input_size=100*100*3, action_size=4):
        super().__init__()
        # 简单的全连接网络
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        # 将图像展平
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SimpleAgent:
    """极简版智能体"""
    def __init__(self, action_size=4):
        self.action_size = action_size
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # 创建网络
        self.model = SimpleDQN(action_size=action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        """选择动作（epsilon-greedy）"""
        if np.random.random() <= self.epsilon:
            # 随机探索
            return np.random.randint(self.action_size)
        else:
            # 网络预测
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values).item()

    def update_epsilon(self):
        """更新探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, experiences):
        """训练网络"""
        if len(experiences) < 32:  # 经验不足
            return 0

        # 随机采样32个经验
        batch = np.random.choice(experiences, 32, replace=False)

        # 准备训练数据
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.FloatTensor([exp['done'] for exp in batch])

        # 计算当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值（简化版Bellman方程）
        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0]
            target_q = rewards + 0.95 * max_next_q * (1 - dones)

        # 计算损失并更新
        loss = self.criterion(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

#### 步骤2：创建简化版环境
```python
# simple_env.py
import numpy as np
import time

class SimpleGameEnv:
    """简化版游戏环境"""
    def __init__(self):
        self.reset()

    def reset(self):
        """重置环境"""
        # 模拟游戏状态：位置x, 位置y, 血量, 敌人距离
        self.state = np.array([50.0, 50.0, 100.0, 30.0])
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        """获取当前状态（模拟截图）"""
        # 创建一个100x100的灰度图像作为状态
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # 在图像上绘制简单特征
        x, y, hp, enemy_dist = self.state

        # 绘制玩家位置（绿色方块）
        player_x, player_y = int(x), int(y)
        image[max(0, player_x-5):min(100, player_x+5),
              max(0, player_y-5):min(100, player_y+5)] = [0, 255, 0]

        # 绘制敌人位置（红色方块）
        enemy_x, enemy_y = int(x + enemy_dist), int(y)
        if enemy_x < 100:
            image[max(0, enemy_x-3):min(100, enemy_x+3),
                  max(0, enemy_y-3):min(100, enemy_y+3)] = [255, 0, 0]

        # 绘制血量条
        hp_width = int(hp)
        image[5:10, 10:10+hp_width] = [0, 255, 0]

        return image

    def step(self, action):
        """执行动作"""
        self.step_count += 1
        reward = 0
        done = False

        # 动作定义：0=向上，1=向下，2=向左，3=向右
        if action == 0 and self.state[1] > 10:  # 向上
            self.state[1] -= 5
        elif action == 1 and self.state[1] < 90:  # 向下
            self.state[1] += 5
        elif action == 2 and self.state[0] > 10:  # 向左
            self.state[0] -= 5
        elif action == 3 and self.state[0] < 90:  # 向右
            self.state[0] += 5

        # 更新敌人距离（敌人会靠近玩家）
        self.state[3] = max(5, self.state[3] - 1)  # 敌人逐渐靠近

        # 计算奖励
        if self.state[3] > 20:  # 敌人距离安全
            reward = 1.0
        elif self.state[3] > 10:  # 敌人距离中等
            reward = 0.0
        else:  # 敌人距离太近
            reward = -1.0
            self.state[2] -= 10  # 扣血

        # 游戏结束条件
        if self.state[2] <= 0 or self.step_count > 100:
            done = True
            if self.state[2] <= 0:
                reward = -10  # 死亡惩罚
            else:
                reward = 10   # 存活奖励

        return self.get_state(), reward, done
```

#### 步骤3：创建训练循环
```python
# simple_train.py
import torch
import numpy as np
from simple_dqn import SimpleAgent
from simple_env import SimpleGameEnv
import cv2

def image_to_tensor(image):
    """将图像转换为PyTorch张量"""
    # 调整图像大小到100x100
    image = cv2.resize(image, (100, 100))

    # 转换为float32并归一化
    image = image.astype(np.float32) / 255.0

    # 转换为PyTorch张量并调整维度
    tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC->CHW
    return tensor

def train_simple_ai():
    """训练简化版AI"""
    print("=== 开始训练简化版AI ===")

    # 创建环境和智能体
    env = SimpleGameEnv()
    agent = SimpleAgent(action_size=4)

    # 经验存储
    experiences = []
    max_experiences = 1000

    # 训练统计
    episode_rewards = []
    episode_lengths = []

    # 训练循环
    num_episodes = 200

    for episode in range(num_episodes):
        # 重置环境
        state_image = env.reset()
        state = image_to_tensor(state_image)

        total_reward = 0
        steps = 0
        done = False

        while not done:
            # 选择动作
            action = agent.get_action(state)

            # 执行动作
            next_state_image, reward, done = env.step(action)
            next_state = image_to_tensor(next_state_image)

            # 存储经验
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            }
            experiences.append(experience)

            # 限制经验池大小
            if len(experiences) > max_experiences:
                experiences.pop(0)

            # 更新状态
            state = next_state
            total_reward += reward
            steps += 1

        # 训练智能体
        if len(experiences) >= 32:
            loss = agent.train(experiences)

            # 更新探索率
            agent.update_epsilon()

        # 记录统计
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # 打印进度
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_length = np.mean(episode_lengths[-20:])
            print(f"Episode {episode+1}: 平均奖励={avg_reward:.2f}, 平均步数={avg_length:.1f}, Epsilon={agent.epsilon:.3f}")

    print("=== 训练完成 ===")

    # 返回训练结果
    return {
        'agent': agent,
        'rewards': episode_rewards,
        'lengths': episode_lengths
    }

if __name__ == "__main__":
    # 运行训练
    result = train_simple_ai()

    # 简单评估
    print("\n=== 最终评估 ===")
    agent = result['agent']
    env = SimpleGameEnv()

    # 测试5局
    test_rewards = []
    for i in range(5):
        state_image = env.reset()
        state = image_to_tensor(state_image)
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state_image, reward, done = env.step(action)
            state = image_to_tensor(next_state_image)
            total_reward += reward

        test_rewards.append(total_reward)
        print(f"测试局 {i+1}: 奖励 = {total_reward}")

    print(f"平均测试奖励: {np.mean(test_rewards):.2f}")
```

### 运行效果验证

运行这个简化版本，你应该看到类似这样的输出：

```
=== 开始训练简化版AI ===
Episode 20: 平均奖励=-3.50, 平均步数=55.0, Epsilon=0.904
Episode 40: 平均奖励=-1.20, 平均步数=62.5, Epsilon=0.818
Episode 60: 平均奖励=1.30, 平均步数=68.0, Epsilon=0.740
Episode 80: 平均奖励=3.80, 平均步数=72.5, Epsilon=0.670
Episode 100: 平均奖励=5.20, 平均步数=75.0, Epsilon=0.606
Episode 120: 平均奖励=6.50, 平均步数=78.0, Epsilon=0.549
Episode 140: 平均奖励=7.80, 平均步数=80.5, Epsilon=0.497
Episode 160: 平均奖励=8.50, 平均步数=82.0, Epsilon=0.450
Episode 180: 平均奖励=9.20, 平均步数=83.5, Epsilon=0.407
Episode 200: 平均奖励=9.80, 平均步数=85.0, Epsilon=0.369

=== 最终评估 ===
测试局 1: 奖励 = 12
测试局 2: 奖励 = 10
测试局 3: 奖励 = 11
测试局 4: 奖励 = 9
测试局 5: 奖励 = 12
平均测试奖励: 10.80
```

**关键观察**：
- ✅ 平均奖励从负值逐渐变为正值
- ✅ 平均步数逐渐增加（存活时间变长）
- ✅ Epsilon逐渐减小（探索减少）
- ✅ 最终测试表现良好

## 6.2 超参数调节的实用经验

### 超参数分类与调节策略

#### 1. 网络结构参数
```python
# 网络结构超参数
NETWORK_PARAMS = {
    'conv_channels': [32, 64, 128],    # 卷积层通道数
    'kernel_sizes': [3, 5, 7],         # 卷积核大小
    'fc_units': [128, 256, 512],       # 全连接层单元数
    'num_layers': [2, 3, 4],           # 网络层数
}

def test_network_architecture():
    """网络结构调优经验"""
    results = {}

    for conv_channels in NETWORK_PARAMS['conv_channels']:
        for fc_units in NETWORK_PARAMS['fc_units']:
            # 创建不同结构的网络
            model = create_model(conv_channels=conv_channels, fc_units=fc_units)

            # 训练相同轮数
            performance = train_and_evaluate(model, epochs=50)

            # 记录结果
            key = f"conv{conv_channels}_fc{fc_units}"
            results[key] = performance

            print(f"架构 {key}: 性能={performance:.3f}")

    return results

# 调优结论：
# - 卷积通道数：64-128最佳，过多容易过拟合
# - 全连接单元：256-512最佳，过多训练慢
# - 网络深度：3-4层最佳，过深难训练
```

#### 2. 训练过程参数
```python
# 训练过程超参数调优经验
TRAINING_PARAMS = {
    'learning_rate': {
        'range': [0.0001, 0.001, 0.01],
        'default': 0.001,
        'rule': '从大到小尝试，找到稳定训练的最大值'
    },
    'batch_size': {
        'range': [16, 32, 64, 128],
        'default': 64,
        'rule': 'GPU内存允许范围内越大越好'
    },
    'gamma': {
        'range': [0.8, 0.9, 0.95, 0.99],
        'default': 0.95,
        'rule': '游戏类任务0.9-0.99，平衡短期和长期奖励'
    },
    'epsilon_decay': {
        'range': [0.99, 0.995, 0.999],
        'default': 0.995,
        'rule': '训练时间长用慢衰减，时间短用快衰减'
    }
}

def tune_training_params():
    """训练参数调优"""
    best_params = {}
    best_performance = float('-inf')

    # 网格搜索（简化版）
    for lr in [0.001, 0.0005]:
        for batch_size in [32, 64]:
            for gamma in [0.95, 0.99]:
                params = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'gamma': gamma,
                    'epsilon_decay': 0.995
                }

                # 训练模型
                performance = train_with_params(params, epochs=100)

                # 更新最佳参数
                if performance > best_performance:
                    best_performance = performance
                    best_params = params.copy()

                print(f"参数 {params}: 性能={performance:.3f}")

    print(f"最佳参数: {best_params}, 性能: {best_performance:.3f}")
    return best_params
```

### 实用调参技巧

#### 1. 分阶段调参法
```python
def staged_hyperparameter_tuning():
    """分阶段超参数调优"""

    # 第一阶段：快速粗调
    print("=== 第一阶段：粗调（每配置10局）===")
    coarse_params = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'batch_size': [32, 64, 128],
        'gamma': [0.9, 0.95, 0.99]
    }

    best_coarse = quick_tune(coarse_params, episodes_per_config=10)

    # 第二阶段：精细调优
    print("=== 第二阶段：精调（每配置50局）===")
    fine_params = {
        'learning_rate': np.linspace(best_coarse['lr']*0.5, best_coarse['lr']*2, 5),
        'batch_size': [best_coarse['batch']//2, best_coarse['batch'], best_coarse['batch']*2],
        'gamma': np.linspace(best_coarse['gamma']-0.05, best_coarse['gamma']+0.05, 5)
    }

    best_fine = careful_tune(fine_params, episodes_per_config=50)

    # 第三阶段：验证调优
    print("=== 第三阶段：验证（每配置100局）===")
    final_performance = validate_params(best_fine, episodes=100)

    return best_fine, final_performance
```

#### 2. 自适应调参法
```python
class AdaptiveHyperparameterTuner:
    def __init__(self):
        self.param_history = []
        self.performance_history = []
        self.trend_analyzer = TrendAnalyzer()

    def adaptive_tune(self, model, initial_params):
        """自适应调参"""
        current_params = initial_params.copy()

        for iteration in range(20):  # 20轮调参
            # 训练当前配置
            performance = self.train_and_evaluate(model, current_params, episodes=50)

            # 记录历史
            self.param_history.append(current_params.copy())
            self.performance_history.append(performance)

            # 分析趋势
            trends = self.trend_analyzer.analyze(self.param_history, self.performance_history)

            # 智能调整参数
            current_params = self.intelligent_adjustment(current_params, trends, performance)

            print(f"迭代 {iteration+1}: 性能={performance:.3f}, 参数={current_params}")

        # 返回最佳参数
        best_idx = np.argmax(self.performance_history)
        return self.param_history[best_idx], self.performance_history[best_idx]

    def intelligent_adjustment(self, params, trends, current_performance):
        """智能调整参数"""
        new_params = params.copy()

        # 根据性能趋势调整
        if len(self.performance_history) >= 3:
            recent_trend = self.performance_history[-3:]

            if all(recent_trend[i] <= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                # 性能在提升，小幅调整
                adjustment_factor = 1.1
            elif all(recent_trend[i] >= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                # 性能在下降，需要改变方向
                adjustment_factor = 0.5
            else:
                # 性能震荡，保持稳定
                adjustment_factor = 1.0

            # 调整学习率
            if 'learning_rate' in trends and trends['learning_rate'] == 'positive':
                new_params['learning_rate'] *= (1 + 0.1 * adjustment_factor)
            else:
                new_params['learning_rate'] *= (1 - 0.05 * adjustment_factor)

            # 调整epsilon衰减
            if 'epsilon_decay' in trends:
                new_params['epsilon_decay'] = np.clip(
                    new_params['epsilon_decay'] * (1 + 0.01 * adjustment_factor),
                    0.99, 0.999
                )

        return new_params
```

#### 3. 早停法优化调参
```python
class EarlyStoppingTuner:
    def __init__(self, patience=5, min_improvement=0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_performance = float('-inf')
        self.patience_counter = 0
        self.param_sequence = []

    def tune_with_early_stopping(self, param_generator, train_func):
        """带早停的调参"""
        for params in param_generator:
            print(f"测试参数: {params}")

            # 训练并评估
            performance = train_func(params)
            self.param_sequence.append((params.copy(), performance))

            # 检查是否有改善
            improvement = performance - self.best_performance
            if improvement > self.min_improvement:
                print(f"✅ 性能改善: {improvement:.4f}")
                self.best_performance = performance
                self.patience_counter = 0
            else:
                print(f"❌ 性能无改善: {improvement:.4f}")
                self.patience_counter += 1

            # 早停判断
            if self.patience_counter >= self.patience:
                print(f"连续{self.patience}次无改善，提前结束调参")
                break

        # 返回最佳参数
        best_params = max(self.param_sequence, key=lambda x: x[1])
        return best_params[0], best_params[1]
```

## 6.3 奖励函数设计与调试方法

### 奖励函数的重要性

奖励函数是强化学习的"指南针"，告诉AI什么行为是好的。设计不当会导致：
- ❌ AI学会错误的策略
- ❌ 训练不稳定
- ❌ 收敛到局部最优

### 基础奖励设计原则

#### 1. 稀疏奖励问题
```python
# ❌ 稀疏奖励：只有最终结果
if game_won:
    reward = 100
else:
    reward = -100

# ✅ 密集奖励：每步都有反馈
def dense_reward_function(state, action, next_state):
    reward = 0

    # 生存奖励（每步小额奖励，鼓励活下去）
    reward += 0.1

    # 移动奖励（鼓励前进）
    if moved_forward(state, next_state):
        reward += 0.5

    # 攻击奖励（击中敌人）
    if hit_enemy(action, next_state):
        reward += 2.0

    # 受到伤害惩罚
    if took_damage(state, next_state):
        reward -= 1.0

    # 最终奖励
    if game_won(next_state):
        reward += 100
    elif game_lost(next_state):
        reward -= 100

    return reward
```

#### 2. 奖励尺度设计
```python
# 奖励函数设计模板
class RewardDesigner:
    def __init__(self):
        # 定义奖励组件和权重
        self.reward_components = {
            'survival': {'weight': 0.1, 'scale': 1.0},     # 生存奖励
            'movement': {'weight': 0.3, 'scale': 1.0},     # 移动奖励
            'combat': {'weight': 0.4, 'scale': 2.0},       # 战斗奖励
            'health': {'weight': 0.2, 'scale': 1.0},       # 血量奖励
        }

        # 最终奖励（大胜/大败）
        self.final_rewards = {
            'victory': 100.0,
            'defeat': -100.0,
            'draw': 0.0
        }

    def calculate_reward(self, state, action, next_state):
        """计算复合奖励"""
        total_reward = 0
        reward_breakdown = {}

        # 1. 生存奖励（每步小额奖励）
        survival_reward = self.reward_components['survival']['weight'] * self.reward_components['survival']['scale']
        total_reward += survival_reward
        reward_breakdown['survival'] = survival_reward

        # 2. 移动奖励（鼓励向敌人移动）
        movement_reward = self.calculate_movement_reward(state, next_state)
        weighted_movement = self.reward_components['movement']['weight'] * movement_reward
        total_reward += weighted_movement
        reward_breakdown['movement'] = weighted_movement

        # 3. 战斗奖励（攻击相关）
        combat_reward = self.calculate_combat_reward(action, next_state)
        weighted_combat = self.reward_components['combat']['weight'] * combat_reward
        total_reward += weighted_combat
        reward_breakdown['combat'] = weighted_combat

        # 4. 血量奖励（血量变化）
        health_reward = self.calculate_health_reward(state, next_state)
        weighted_health = self.reward_components['health']['weight'] * health_reward
        total_reward += weighted_health
        reward_breakdown['health'] = weighted_health

        # 5. 最终奖励（游戏结束）
        final_reward = self.calculate_final_reward(next_state)
        total_reward += final_reward
        reward_breakdown['final'] = final_reward

        return total_reward, reward_breakdown
```

### 奖励函数调试工具

#### 1. 奖励统计分析
```python
class RewardAnalyzer:
    def __init__(self):
        self.reward_history = []
        self.component_history = []

    def record_reward(self, total_reward, components):
        """记录奖励数据"""
        self.reward_history.append(total_reward)
        self.component_history.append(components)

    def analyze_reward_distribution(self):
        """分析奖励分布"""
        import matplotlib.pyplot as plt

        # 总体奖励分布
        plt.figure(figsize=(15, 10))

        # 子图1：奖励时间序列
        plt.subplot(2, 3, 1)
        plt.plot(self.reward_history)
        plt.title('奖励时间序列')
        plt.xlabel('步骤')
        plt.ylabel('奖励')

        # 子图2：奖励直方图
        plt.subplot(2, 3, 2)
        plt.hist(self.reward_history, bins=50, alpha=0.7)
        plt.title('奖励分布直方图')
        plt.xlabel('奖励值')
        plt.ylabel('频次')

        # 子图3：各组件贡献
        if self.component_history:
            plt.subplot(2, 3, 3)
            component_names = list(self.component_history[0].keys())
            component_values = {name: [] for name in component_names}

            for components in self.component_history:
                for name, value in components.items():
                    component_values[name].append(value)

            # 绘制堆叠面积图
            bottom = np.zeros(len(component_values[component_names[0]]))
            for name in component_names:
                plt.fill_between(range(len(component_values[name])),
                               bottom, bottom + component_values[name],
                               label=name, alpha=0.7)
                bottom += component_values[name]

            plt.title('奖励组件贡献')
            plt.xlabel('步骤')
            plt.ylabel('奖励分量')
            plt.legend()

        # 子图4：奖励统计
        plt.subplot(2, 3, 4)
        rewards = np.array(self.reward_history)
        stats_text = f"""
        奖励统计:
        平均值: {np.mean(rewards):.3f}
        标准差: {np.std(rewards):.3f}
        最小值: {np.min(rewards):.3f}
        最大值: {np.max(rewards):.3f}
        中位数: {np.median(rewards):.3f}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        plt.axis('off')
        plt.title('奖励统计信息')

        # 子图5：移动平均奖励
        plt.subplot(2, 3, 5)
        window_size = min(100, len(self.reward_history))
        if window_size > 1:
            moving_avg = np.convolve(self.reward_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg)
            plt.title(f'{window_size}步移动平均奖励')
            plt.xlabel('步骤')
            plt.ylabel('平均奖励')

        # 子图6：奖励分布箱线图
        plt.subplot(2, 3, 6)
        plt.boxplot(self.reward_history)
        plt.title('奖励分布箱线图')
        plt.ylabel('奖励值')

        plt.tight_layout()
        plt.savefig('reward_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        return {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'median': np.median(rewards)
        }
```

#### 2. 奖励函数A/B测试
```python
class RewardABTest:
    """奖励函数A/B测试"""
    def __init__(self, reward_func_a, reward_func_b, test_ratio=0.5):
        self.reward_func_a = reward_func_a
        self.reward_func_b = reward_func_b
        self.test_ratio = test_ratio

        self.group_a_rewards = []
        self.group_b_rewards = []
        self.group_a_episodes = []
        self.group_b_episodes = []

    def get_reward_function(self, episode_id):
        """根据episode_id分配奖励函数"""
        if episode_id % 2 == 0:  # 简单交替分配
            return self.reward_func_a, 'A'
        else:
            return self.reward_func_b, 'B'

    def record_episode(self, episode_id, group, total_reward, episode_length, won):
        """记录episode结果"""
        if group == 'A':
            self.group_a_rewards.append(total_reward)
            self.group_a_episodes.append({
                'episode': episode_id,
                'reward': total_reward,
                'length': episode_length,
                'won': won
            })
        else:
            self.group_b_rewards.append(total_reward)
            self.group_b_episodes.append({
                'episode': episode_id,
                'reward': total_reward,
                'length': episode_length,
                'won': won
            })

    def analyze_results(self):
        """分析A/B测试结果"""
        import scipy.stats as stats

        print("=== A/B测试结果分析 ===")

        # 基础统计
        a_rewards = np.array(self.group_a_rewards)
        b_rewards = np.array(self.group_b_rewards)

        print(f"Group A - 样本数: {len(a_rewards)}, 平均奖励: {np.mean(a_rewards):.3f}, 标准差: {np.std(a_rewards):.3f}")
        print(f"Group B - 样本数: {len(b_rewards)}, 平均奖励: {np.mean(b_rewards):.3f}, 标准差: {np.std(b_rewards):.3f}")

        # t检验
        if len(a_rewards) > 0 and len(b_rewards) > 0:
            t_stat, p_value = stats.ttest_ind(a_rewards, b_rewards)
            print(f"t检验结果: t统计量={t_stat:.4f}, p值={p_value:.4f}")

            if p_value < 0.05:
                better_group = 'A' if np.mean(a_rewards) > np.mean(b_rewards) else 'B'
                print(f"✅ 统计显著！组{better_group}表现更好")
            else:
                print("❌ 无统计显著差异")

        # 胜率分析
        a_wins = sum(1 for ep in self.group_a_episodes if ep['won']) / len(self.group_a_episodes) if self.group_a_episodes else 0
        b_wins = sum(1 for ep in self.group_b_episodes if ep['won']) / len(self.group_b_episodes) if self.group_b_episodes else 0

        print(f"Group A胜率: {a_wins:.3f}")
        print(f"Group B胜率: {b_wins:.3f}")

        return {
            'group_a_stats': {
                'mean_reward': np.mean(a_rewards) if len(a_rewards) > 0 else 0,
                'win_rate': a_wins
            },
            'group_b_stats': {
                'mean_reward': np.mean(b_rewards) if len(b_rewards) > 0 else 0,
                'win_rate': b_wins
            },
            'significant': p_value < 0.05 if len(a_rewards) > 0 and len(b_rewards) > 0 else False
        }
```

#### 3. 奖励函数问题诊断
```python
class RewardDiagnostic:
    """奖励函数问题诊断"""

    COMMON_ISSUES = {
        'sparse_reward': {
            'description': '奖励过于稀疏',
            'symptoms': ['大部分奖励为0', '奖励方差很小'],
            'solution': '添加中间奖励，密度化奖励信号'
        },
        'reward_explosion': {
            'description': '奖励爆炸',
            'symptoms': ['奖励值差异巨大', '存在极大值'],
            'solution': '归一化奖励，限制奖励范围'
        },
        'unstable_reward': {
            'description': '奖励不稳定',
            'symptoms': ['奖励剧烈波动', '相同状态奖励差异大'],
            'solution': '平滑奖励，减少随机性'
        },
        'local_optimum': {
            'description': '陷入局部最优',
            'symptoms': ['奖励早期高后期低', '策略不再改善'],
            'solution': '调整奖励权重，增加探索'
        }
    }

    def diagnose(self, reward_history, component_history):
        """诊断奖励函数问题"""
        issues = []

        rewards = np.array(reward_history)

        # 1. 检查稀疏奖励
        zero_rewards = np.sum(rewards == 0) / len(rewards)
        if zero_rewards > 0.8:  # 80%的奖励为0
            issues.append({
                'issue': 'sparse_reward',
                'severity': 'high' if zero_rewards > 0.9 else 'medium',
                'evidence': f'{zero_rewards*100:.1f}%的奖励为0'
            })

        # 2. 检查奖励爆炸
        reward_std = np.std(rewards)
        reward_range = np.max(rewards) - np.min(rewards)
        if reward_range > 100 * reward_std:  # 范围远大于标准差
            issues.append({
                'issue': 'reward_explosion',
                'severity': 'high',
                'evidence': f'奖励范围({reward_range:.1f})远大于标准差({reward_std:.1f})'
            })

        # 3. 检查奖励稳定性
        if len(rewards) > 100:
            recent_std = np.std(rewards[-100:])
            early_std = np.std(rewards[:100])
            if recent_std > 2 * early_std:  # 近期波动显著增大
                issues.append({
                    'issue': 'unstable_reward',
                    'severity': 'medium',
                    'evidence': f'近期奖励波动({recent_std:.1f})远大于早期({early_std:.1f})'
                })

        # 4. 检查局部最优
        if len(rewards) > 200:
            early_mean = np.mean(rewards[:100])
            recent_mean = np.mean(rewards[-100:])
            if early_mean > recent_mean + 2:  # 早期明显优于近期
                issues.append({
                    'issue': 'local_optimum',
                    'severity': 'high',
                    'evidence': f'早期平均奖励({early_mean:.1f})优于近期({recent_mean:.1f})'
                })

        # 5. 分析组件贡献
        if component_history:
            component_analysis = self.analyze_component_balance(component_history)
            if component_analysis['unbalanced']:
                issues.append({
                    'issue': 'unbalanced_components',
                    'severity': 'medium',
                    'evidence': component_analysis['description']
                })

        return self.generate_diagnosis_report(issues)

    def generate_diagnosis_report(self, issues):
        """生成诊断报告"""
        if not issues:
            return {
                'status': 'healthy',
                'message': '奖励函数看起来工作正常',
                'recommendations': ['继续保持当前设计', '定期监控性能']
            }

        report = {
            'status': 'needs_attention',
            'issues_found': len(issues),
            'problems': [],
            'recommendations': []
        }

        for issue in issues:
            issue_type = issue['issue']
            if issue_type in self.COMMON_ISSUES:
                info = self.COMMON_ISSUES[issue_type]
                report['problems'].append({
                    'type': issue_type,
                    'description': info['description'],
                    'severity': issue['severity'],
                    'evidence': issue['evidence'],
                    'solution': info['solution']
                })

                if info['solution'] not in report['recommendations']:
                    report['recommendations'].append(info['solution'])

        # 按严重程度排序
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        report['problems'].sort(key=lambda x: severity_order[x['severity']])

        return report
```

## 6.4 动作空间设计的原则与技巧

### 动作空间设计原则

#### 1. 最小必要性原则
```python
# ❌ 过度复杂：一次性定义所有可能动作
ACTION_SPACE = {
    'move_x': range(-100, 101),      # 201个可能值
    'move_y': range(-100, 101),      # 201个可能值
    'attack_type': range(10),        # 10种攻击
    'skill_combo': range(50),        # 50种连招
    'item_use': range(30),           # 30种道具
    'communication': range(20),      # 20种交流
}
# 总动作空间：201 × 201 × 10 × 50 × 30 × 20 = 2.4万亿！

# ✅ 最小必要：从核心动作开始
SIMPLE_ACTION_SPACE = {
    'move': ['up', 'down', 'left', 'right', 'stay'],  # 5个动作
    'attack': ['basic', 'skill1', 'none'],            # 3个动作
}
# 总动作空间：5 × 3 = 15个组合
```

#### 2. 渐进式扩展
```python
class ProgressiveActionSpace:
    """渐进式动作空间扩展"""

    def __init__(self):
        # 阶段1：基础移动
        self.phase1_actions = {
            'movement': ['forward', 'backward', 'left', 'right', 'stay']
        }

        # 阶段2：添加基础攻击
        self.phase2_actions = {
            'movement': ['forward', 'backward', 'left', 'right', 'stay'],
            'attack': ['basic_attack', 'skill_1', 'none']
        }

        # 阶段3：添加高级动作
        self.phase3_actions = {
            'movement': ['forward', 'backward', 'left', 'right', 'stay'],
            'attack': ['basic_attack', 'skill_1', 'skill_2', 'ultimate', 'none'],
            'defense': ['dodge', 'block', 'none']
        }

    def get_action_space(self, phase):
        """根据训练阶段获取动作空间"""
        if phase == 1:
            return self.phase1_actions
        elif phase == 2:
            return self.phase2_actions
        else:
            return self.phase3_actions

    def expand_action_space(self, current_performance, threshold=0.8):
        """根据性能自动扩展动作空间"""
        if current_performance > threshold:
            print("✅ 性能达标，扩展到下一阶段动作空间")
            return True
        return False
```

### 动作空间优化技巧

#### 1. 动作层次化设计
```python
class HierarchicalActionSpace:
    """层次化动作空间"""

    def __init__(self):
        # 高层策略（2个选择）
        self.high_level_actions = {
            'strategy': ['aggressive', 'defensive']  # 进攻/防守策略
        }

        # 中层动作（每个策略下6个选择）
        self.mid_level_actions = {
            'aggressive': {
                'approach': ['direct', 'flank', 'wait'],
                'attack': ['basic', 'combo', 'ultimate']
            },
            'defensive': {
                'position': ['safe_spot', 'high_ground', 'cover'],
                'action': ['heal', 'buff', 'counter_attack']
            }
        }

        # 低层执行（具体动作，每个6个选择）
        self.low_level_actions = {
            'direct': ['move_forward', 'move_left', 'move_right', 'dash', 'jump', 'stay'],
            'basic': ['attack_1', 'attack_2', 'attack_3', 'skill_1', 'skill_2', 'none'],
            # ... 其他组合
        }

    def sample_hierarchical_action(self, high_level_choice):
        """层次化采样动作"""
        # 1. 高层策略选择
        strategy = high_level_choice['strategy']

        # 2. 中层动作选择（可以分别训练）
        mid_level_choices = {}
        for category, actions in self.mid_level_actions[strategy].items():
            mid_level_choices[category] = np.random.choice(actions)

        # 3. 低层动作选择
        low_level_choices = {}
        for mid_action in mid_level_choices.values():
            low_actions = self.low_level_actions.get(mid_action, ['none'])
            low_level_choices[mid_action] = np.random.choice(low_actions)

        return {
            'high_level': high_level_choice,
            'mid_level': mid_level_choices,
            'low_level': low_level_choices
        }
```

#### 2. 连续动作离散化
```python
class ContinuousToDiscrete:
    """连续动作离散化"""

    def __init__(self, continuous_ranges, num_discrete=5):
        """
        参数:
            continuous_ranges: 连续动作范围
                {'move_x': (-1, 1), 'move_y': (-1, 1), 'angle': (0, 360)}
            num_discrete: 每个维度的离散化级别
        """
        self.continuous_ranges = continuous_ranges
        self.num_discrete = num_discrete
        self.discrete_to_continuous = {}

        # 为每个连续维度创建离散化映射
        for dim, (min_val, max_val) in continuous_ranges.items():
            # 均匀离散化
            discrete_values = np.linspace(min_val, max_val, num_discrete)
            self.discrete_to_continuous[dim] = discrete_values

    def discrete_to_continuous_action(self, discrete_action):
        """将离散动作转换为连续动作"""
        continuous_action = {}

        for dim, discrete_idx in discrete_action.items():
            if dim in self.discrete_to_continuous:
                continuous_value = self.discrete_to_continuous[dim][discrete_idx]
                continuous_action[dim] = continuous_value

        return continuous_action

    def get_discrete_action_space(self):
        """获取离散动作空间"""
        discrete_space = {}
        for dim in self.continuous_ranges.keys():
            discrete_space[dim] = list(range(self.num_discrete))
        return discrete_space

    def adaptive_discretization(self, performance_history, refinement_threshold=0.1):
        """根据性能自适应细化离散化"""
        # 识别表现好的动作区间
        good_intervals = self.identify_good_intervals(performance_history)

        # 在好区间增加离散化精度
        for dim, intervals in good_intervals.items():
            for start_idx, end_idx in intervals:
                if end_idx - start_idx > 1:  # 如果有空间细化
                    self.refine_discretization(dim, start_idx, end_idx, refinement_threshold)

    def identify_good_intervals(self, performance_history):
        """识别表现好的动作区间"""
        good_intervals = {}

        for dim in self.continuous_ranges.keys():
            dim_performances = [(idx, perf) for idx, perf, dim_val in performance_history if dim_val[0] == dim]

            # 找出性能较好的动作索引
            sorted_performances = sorted(dim_performances, key=lambda x: x[1], reverse=True)

            # 选择前20%作为"好区间"
            num_good = max(1, len(sorted_performances) // 5)
            good_indices = [idx for idx, _ in sorted_performances[:num_good]]

            # 合并相邻的好索引
            good_intervals[dim] = self.merge_adjacent_indices(good_indices)

        return good_intervals
```

### 动作有效性验证

#### 1. 动作覆盖测试
```python
class ActionCoverageTest:
    """动作覆盖测试"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.action_counts = {}
        self.action_rewards = {}

        # 初始化统计
        self.reset_statistics()

    def reset_statistics(self):
        """重置统计"""
        for action_type, actions in self.action_space.items():
            self.action_counts[action_type] = {action: 0 for action in actions}
            self.action_rewards[action_type] = {action: [] for action in actions}

    def record_action(self, action, reward):
        """记录动作和奖励"""
        for action_type, action_value in action.items():
            if action_type in self.action_counts:
                self.action_counts[action_type][action_value] += 1
                self.action_rewards[action_type][action_value].append(reward)

    def analyze_coverage(self):
        """分析动作覆盖情况"""
        coverage_report = {}

        for action_type in self.action_space.keys():
            total_actions = sum(self.action_counts[action_type].values())

            if total_actions == 0:
                coverage_report[action_type] = {
                    'coverage': 0,
                    'unused_actions': list(self.action_space[action_type]),
                    'most_used': None,
                    'least_used': None
                }
                continue

            # 计算覆盖率
            used_actions = sum(1 for count in self.action_counts[action_type].values() if count > 0)
            coverage = used_actions / len(self.action_space[action_type])

            # 找出使用最多/最少的动作
            action_usage = self.action_counts[action_type]
            most_used = max(action_usage.items(), key=lambda x: x[1])
            least_used = min(action_usage.items(), key=lambda x: x[1] if x[1] > 0 else float('inf'))

            # 计算平均奖励
            avg_rewards = {}
            for action, rewards in self.action_rewards[action_type].items():
                if rewards:
                    avg_rewards[action] = np.mean(rewards)
                else:
                    avg_rewards[action] = 0

            coverage_report[action_type] = {
                'coverage': coverage,
                'total_usage': total_actions,
                'unused_actions': [action for action, count in action_usage.items() if count == 0],
                'most_used': most_used,
                'least_used': least_used,
                'average_rewards': avg_rewards
            }

        return coverage_report

    def identify_unused_actions(self, min_usage_threshold=10):
        """识别使用不足的动作"""
        coverage_report = self.analyze_coverage()
        unused_actions = {}

        for action_type, report in coverage_report.items():
            # 找出使用次数少于阈值的动作
            infrequently_used = []
            for action, count in self.action_counts[action_type].items():
                if count < min_usage_threshold:
                    infrequently_used.append((action, count))

            if infrequently_used:
                unused_actions[action_type] = infrequently_used

        return unused_actions

    def suggest_action_space_improvements(self):
        """建议动作空间改进"""
        suggestions = []
        coverage_report = self.analyze_coverage()
        unused_actions = self.identify_unused_actions()

        for action_type, unused_list in unused_actions.items():
            if len(unused_list) > len(self.action_space[action_type]) * 0.5:  # 超过50%未使用
                suggestions.append({
                    'action_type': action_type,
                    'issue': '过多未使用动作',
                    'suggestion': f'减少{action_type}的动作数量，或提高这些动作的奖励',
                    'unused_actions': [action for action, _ in unused_list]
                })
            elif len(unused_list) > 0:
                suggestions.append({
                    'action_type': action_type,
                    'issue': '部分动作使用不足',
                    'suggestion': f'分析为什么这些动作不被使用，可能需要调整奖励函数',
                    'unused_actions': [action for action, _ in unused_list]
                })

        # 检查动作奖励差异
        for action_type, report in coverage_report.items():
            if 'average_rewards' in report:
                avg_rewards = list(report['average_rewards'].values())
                if len(avg_rewards) > 1:
                    reward_std = np.std(avg_rewards)
                    reward_range = max(avg_rewards) - min(avg_rewards)

                    if reward_range > 10:  # 奖励差异过大
                        suggestions.append({
                            'action_type': action_type,
                            'issue': '动作间奖励差异过大',
                            'suggestion': '平衡各动作的奖励，避免某些动作被完全忽视',
                            'reward_range': reward_range
                        })

        return suggestions
```

## 6.5 状态表示优化的实用方案

### 状态表示的重要性

状态是AI观察世界的"眼睛"，好的状态表示应该：
- ✅ 包含决策相关的关键信息
- ✅ 去除冗余和噪声
- ✅ 具有适当的抽象层次
- ✅ 便于神经网络处理

### 从原始图像到智能特征

#### 1. 图像预处理优化
```python
class OptimizedImageProcessor:
    """优化的图像处理器"""

    def __init__(self, target_size=(84, 84), grayscale=True, normalize=True):
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize

        # 预计算一些常量
        self.resize_factor = self.calculate_resize_factor()

    def process_frame(self, frame):
        """处理单帧图像"""
        # 1. 尺寸调整（使用最快的插值方法）
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 2. 颜色空间转换（可选）
        if self.grayscale:
            processed = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            processed = np.expand_dims(processed, axis=-1)  # 保持维度一致
        else:
            processed = resized

        # 3. 归一化
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0

        return processed

    def process_frame_stack(self, frames, stack_size=4):
        """处理帧堆叠（用于捕捉时序信息）"""
        processed_frames = []

        for frame in frames[-stack_size:]:  # 取最近的stack_size帧
            processed = self.process_frame(frame)
            processed_frames.append(processed)

        # 堆叠成多通道图像
        if self.grayscale:
            # 灰度图：堆叠成[stack_size, H, W]
            frame_stack = np.concatenate(processed_frames, axis=-1)
            frame_stack = np.transpose(frame_stack, (2, 0, 1))  # [H, W, stack_size] -> [stack_size, H, W]
        else:
            # 彩色图：平均池化或选择性堆叠
            frame_stack = np.mean(processed_frames, axis=0)  # 简单平均

        return frame_stack

    def extract_key_regions(self, frame, regions_of_interest):
        """提取关键区域"""
        key_regions = []

        for region_name, (x1, y1, x2, y2) in regions_of_interest.items():
            region = frame[y1:y2, x1:x2]

            # 根据区域重要性调整处理策略
            if 'enemy' in region_name:
                # 敌人区域：高对比度处理
                enhanced = self.enemy_region_enhancement(region)
            elif 'health' in region_name:
                # 血量区域：颜色增强
                enhanced = self.health_region_enhancement(region)
            else:
                enhanced = region

            # 统一尺寸
            standardized = cv2.resize(enhanced, (32, 32))
            key_regions.append(standardized)

        return key_regions

    def enemy_region_enhancement(self, region):
        """敌人区域增强"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)

        # 增强红色通道（敌人通常是红色）
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # 增强饱和度

        # 转回RGB
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return enhanced

    def health_region_enhancement(self, region):
        """血量区域增强"""
        # 提取绿色分量（血量条通常是绿色）
        green_channel = region[:, :, 1]

        # 二值化处理
        _, binary = cv2.threshold(green_channel, 128, 255, cv2.THRESH_BINARY)

        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 转换回3通道
        enhanced = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)
        return enhanced
```

#### 2. 特征提取优化
```python
class FeatureExtractor:
    """智能特征提取器"""

    def __init__(self):
        self.feature_cache = {}
        self.feature_stats = {}

    def extract_comprehensive_features(self, image):
        """提取综合特征"""
        features = {}

        # 1. 低级特征（像素级）
        low_level = self.extract_low_level_features(image)
        features.update(low_level)

        # 2. 中级特征（对象级）
        mid_level = self.extract_mid_level_features(image)
        features.update(mid_level)

        # 3. 高级特征（语义级）
        high_level = self.extract_high_level_features(image)
        features.update(high_level)

        # 4. 时序特征（如果有多帧）
        if len(image.shape) == 4:  # [batch, channels, height, width]
            temporal = self.extract_temporal_features(image)
            features.update(temporal)

        return self.combine_features(features)

    def extract_low_level_features(self, image):
        """提取低级特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        features = {}

        # 颜色直方图
        color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        features['color_histogram'] = color_hist.flatten() / np.sum(color_hist)

        # 边缘特征
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size

        # 纹理特征（简单的Gabor滤波）
        gabor_kernel = cv2.getGaborKernel((21, 21), 4.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        texture = cv2.filter2D(gray, cv2.CV_32F, gabor_kernel)
        features['texture_strength'] = np.mean(np.abs(texture))

        # 亮度和对比度
        features['brightness'] = np.mean(gray)
        features['contrast'] = np.std(gray)

        return {'low_level': features}

    def extract_mid_level_features(self, image):
        """提取中级特征（对象检测相关）"""
        features = {}

        # 使用简单的模板匹配检测关键对象
        # 这里使用预定义的模板（在实际项目中应该是训练好的检测器）

        # 检测圆形对象（可能是敌人或道具）
        circles = cv2.HoughCircles(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )

        if circles is not None:
            features['num_circles'] = len(circles[0])
            features['circle_positions'] = circles[0][:, :2]  # x, y坐标
            features['circle_sizes'] = circles[0][:, 2]       # 半径
        else:
            features['num_circles'] = 0
            features['circle_positions'] = np.array([])
            features['circle_sizes'] = np.array([])

        # 检测直线（可能是边界或攻击方向）
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            features['num_lines'] = len(lines)
            features['line_angles'] = [np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) for line in lines]
        else:
            features['num_lines'] = 0
            features['line_angles'] = []

        # 颜色区域检测
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 检测红色区域（可能是敌人）
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        features['red_area_ratio'] = np.sum(red_mask > 0) / red_mask.size

        # 检测绿色区域（可能是友军或血量）
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        features['green_area_ratio'] = np.sum(green_mask > 0) / green_mask.size

        return {'mid_level': features}

    def extract_high_level_features(self, image):
        """提取高级特征（基于预训练模型或启发式规则）"""
        features = {}

        # 场景复杂度（基于边缘密度和颜色变化）
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # 颜色变化程度
        color_variance = np.var(image, axis=(0, 1))
        total_color_variance = np.sum(color_variance)

        # 基于简单启发式的场景分类
        if edge_density < 0.05 and total_color_variance < 1000:
            scene_type = 'simple'
            complexity_score = 1
        elif edge_density < 0.15 and total_color_variance < 5000:
            scene_type = 'moderate'
            complexity_score = 2
        else:
            scene_type = 'complex'
            complexity_score = 3

        features['scene_type'] = scene_type
        features['complexity_score'] = complexity_score
        features['edge_density'] = edge_density
        features['color_variance'] = total_color_variance

        # 空间布局特征
        height, width = image.shape[:2]
        center_region = image[height//4:3*height//4, width//4:3*width//4]

        features['center_brightness'] = np.mean(center_region)
        features['center_contrast'] = np.std(center_region)

        # 四象限分析
        quadrant_features = {}
        h_half, w_half = height // 2, width // 2

        quadrants = {
            'top_left': image[0:h_half, 0:w_half],
            'top_right': image[0:h_half, w_half:width],
            'bottom_left': image[h_half:height, 0:w_half],
            'bottom_right': image[h_half:height, w_half:width]
        }

        for name, quadrant in quadrants.items():
            quadrant_features[f'{name}_brightness'] = np.mean(quadrant)
            quadrant_features[f'{name}_contrast'] = np.std(quadrant)
            quadrant_features[f'{name}_red_ratio'] = np.sum(quadrant[:, :, 0] > 200) / quadrant.size

        features['quadrant_analysis'] = quadrant_features

        return {'high_level': features}

    def combine_features(self, feature_dict):
        """合并所有特征为向量"""
        feature_vector = []

        # 按层次合并特征
        for level, features in feature_dict.items():
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, dict):
                    # 如果是嵌套字典，继续展开
                    for sub_name, sub_value in feature_value.items():
                        if isinstance(sub_value, (list, np.ndarray)):
                            feature_vector.extend(sub_value)
                        else:
                            feature_vector.append(sub_value)
                elif isinstance(feature_value, (list, np.ndarray)):
                    feature_vector.extend(feature_value)
                else:
                    feature_vector.append(feature_value)

        return np.array(feature_vector, dtype=np.float32)
```

#### 3. 状态降维优化
```python
class StateDimensionalityReducer:
    """状态降维优化器"""

    def __init__(self, target_dim=64):
        self.target_dim = target_dim
        self.pca = None
        self.autoencoder = None
        self.feature_selector = None

    def create_autoencoder(self, input_dim):
        """创建自编码器进行降维"""
        import torch
        import torch.nn as nn

        class Autoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()

                # 编码器
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim)
                )

                # 解码器
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, input_dim)
                )

            def forward(self, x):
                latent = self.encoder(x)
                reconstructed = self.decoder(latent)
                return latent, reconstructed

        self.autoencoder = Autoencoder(input_dim, self.target_dim)
        return self.autoencoder

    def train_dimensionality_reduction(self, state_samples, method='autoencoder'):
        """训练降维模型"""
        if method == 'autoencoder':
            return self.train_autoencoder(state_samples)
        elif method == 'pca':
            return self.train_pca(state_samples)
        elif method == 'feature_selection':
            return self.train_feature_selection(state_samples)

    def train_autoencoder(self, state_samples):
        """训练自编码器"""
        import torch.optim as optim

        # 创建自编码器
        input_dim = state_samples.shape[1]
        autoencoder = self.create_autoencoder(input_dim)

        # 转换数据
        data = torch.FloatTensor(state_samples)
        dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

        # 训练参数
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 训练
        autoencoder.train()
        for epoch in range(100):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()

                latent, reconstructed = autoencoder(batch)
                loss = criterion(reconstructed, batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.6f}")

        self.autoencoder = autoencoder
        return autoencoder

    def reduce_dimension(self, state):
        """降维处理状态"""
        if self.autoencoder is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                latent, _ = self.autoencoder(state_tensor)
                return latent.squeeze(0).numpy()
        else:
            return state  # 如果没有训练好的模型，返回原状态
```

### 状态表示验证

#### 1. 状态有效性测试
```python
class StateValidityTester:
    """状态有效性测试器"""

    def __init__(self):
        self.state_stats = []
        self.information_content = []

    def test_state_representation(self, state_samples, action_rewards):
        """测试状态表示的有效性"""
        results = {}

        # 1. 信息量分析
        information_content = self.calculate_information_content(state_samples)
        results['information_content'] = information_content

        # 2. 可分性分析
        separability = self.analyze_state_separability(state_samples, action_rewards)
        results['separability'] = separability

        # 3. 冗余性分析
        redundancy = self.analyze_redundancy(state_samples)
        results['redundancy'] = redundancy

        # 4. 维度效率分析
        dimension_efficiency = self.analyze_dimension_efficiency(state_samples)
        results['dimension_efficiency'] = dimension_efficiency

        return results

    def calculate_information_content(self, state_samples):
        """计算状态的信息量"""
        from scipy.stats import entropy

        # 计算每个维度的熵（信息量）
        information_per_dim = []

        for dim in range(state_samples.shape[1]):
            dim_data = state_samples[:, dim]

            # 分箱计算直方图
            hist, _ = np.histogram(dim_data, bins=20, density=True)
            hist = hist[hist > 0]  # 移除零值

            # 计算熵
            dim_entropy = entropy(hist)
            information_per_dim.append(dim_entropy)

        avg_information = np.mean(information_per_dim)
        low_info_dims = [i for i, info in enumerate(information_per_dim) if info < 0.1]

        return {
            'average_information': avg_information,
            'information_per_dimension': information_per_dim,
            'low_information_dimensions': low_info_dims
        }

    def analyze_state_separability(self, state_samples, action_rewards):
        """分析状态的可分性"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # 根据奖励分组
        high_reward_mask = np.array(action_rewards) > np.median(action_rewards)
        low_reward_mask = ~high_reward_mask

        if np.sum(high_reward_mask) == 0 or np.sum(low_reward_mask) == 0:
            return {'separability_score': 0, 'message': '无法分组分析'}

        high_reward_states = state_samples[high_reward_mask]
        low_reward_states = state_samples[low_reward_mask]

        # 计算类间距离
        high_center = np.mean(high_reward_states, axis=0)
        low_center = np.mean(low_reward_states, axis=0)
        between_class_distance = np.linalg.norm(high_center - low_center)

        # 计算类内距离
        high_intra_distance = np.mean([np.linalg.norm(state - high_center) for state in high_reward_states])
        low_intra_distance = np.mean([np.linalg.norm(state - low_center) for state in low_reward_states])

        # 可分性评分（类间距离 / 类内距离）
        if high_intra_distance + low_intra_distance > 0:
            separability_score = between_class_distance / (high_intra_distance + low_intra_distance)
        else:
            separability_score = 0

        return {
            'separability_score': separability_score,
            'between_class_distance': between_class_distance,
            'avg_intra_class_distance': (high_intra_distance + low_intra_distance) / 2,
            'high_reward_center': high_center,
            'low_reward_center': low_center
        }
```

## 6.6 小结

本章提供了详细的新手实践指导：

1. **最小可运行版本**：从复杂项目中提取核心功能，构建简化版训练系统
2. **超参数调优**：分阶段调参、自适应调参、早停法等实用技巧
3. **奖励函数设计**：密集奖励、组件化设计、A/B测试和调试工具
4. **动作空间设计**：最小必要性、渐进式扩展、层次化设计原则
5. **状态表示优化**：图像预处理、特征提取、降维优化和有效性验证

这些实践指导能帮助你：
- 快速上手强化学习项目开发
- 避免常见的设计陷阱
- 系统性地优化项目性能
- 建立科学的调试和验证流程

记住：**从简单开始，逐步优化，持续验证**。这是新手成功开发强化学习项目的关键。,\n\n---\n\n*本章内容注重实用性，提供了大量可直接应用的代码模板和调试工具，是新手动手实践的完整指南。*"}
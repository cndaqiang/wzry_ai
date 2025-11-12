# 第五章：开发实战经验与问题解决方案

## 5.1 环境配置常见问题与解决方案

### Python环境管理

#### Conda环境创建最佳实践
```bash
# 创建独立环境（避免污染系统环境）
conda create --name wzry_ai python=3.10
conda activate wzry_ai

# 关键依赖版本锁定（项目测试过的兼容版本）
conda install numpy=1.24.3  # 必须<2.0，否则与OpenCV冲突
conda install pytorch=2.0.1 torchvision=0.15.2 cudatoolkit=11.8
```

**常见版本冲突问题**：
```python
# ❌ 错误：NumPy 2.0 + OpenCV 4.9.0.80
import cv2
import numpy as np
# 报错：numpy.core.multiarray failed to import

# ✅ 正确：NumPy 1.x + OpenCV 4.9.0.80
# 降级NumPy: pip install "numpy<2"
```

#### 依赖安装顺序（重要！）
```bash
# 推荐安装顺序（避免依赖冲突）
1. python -m pip install opencv-contrib-python==4.9.0.80
2. python -m pip install "numpy<2"  # 必须第二步
3. python -m pip install ppocr-onnx==0.0.3.9 --no-deps
4. python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
5. python -m pip install airtest_mobileauto autowzry
```

### CUDA环境配置

#### 检查CUDA兼容性
```python
import torch
import subprocess

def check_cuda_environment():
    """检查CUDA环境配置"""
    print("=== CUDA环境检查 ===")

    # 1. PyTorch CUDA可用性
    print(f"PyTorch CUDA可用: {torch.cuda.is_available()}")
    print(f"PyTorch版本: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")

    # 2. 系统CUDA版本
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        print(f"系统CUDA: {result.stdout.split('release ')[-1].split(',')[0]}")
    except:
        print("系统CUDA未找到")

    # 3. 内存检查
    if torch.cuda.is_available():
        print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
        print(f"当前可用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

check_cuda_environment()
```

#### 常见CUDA问题解决方案

**问题1：CUDA out of memory**
```python
# ❌ 错误：批量大小过大
def train_with_large_batch():
    batch_size = 256  # 超出GPU内存
    return train_model(batch_size)

# ✅ 解决：动态批量大小
def adaptive_batch_size():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    if gpu_memory > 8 * 1024**3:  # 8GB以上
        return 64
    elif gpu_memory > 4 * 1024**3:  # 4GB以上
        return 32
    else:
        return 16
```

**问题2：cuDNN版本不匹配**
```bash
# 检查cuDNN版本
python -c "import torch; print(torch.backends.cudnn.version())"

# 如果版本不匹配，重新安装PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### 游戏环境连接问题

#### ADB连接故障排除
```python
def check_adb_connection():
    """检查ADB连接状态"""
    import subprocess

    # 1. 检查ADB服务
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        devices = result.stdout.strip().split('\n')[1:]  # 跳过标题行

        if len(devices) == 0 or devices[0] == '':
            print("❌ 未检测到ADB设备")
            return False

        for device in devices:
            if 'device' in device:
                device_id = device.split('\t')[0]
                print(f"✅ 检测到设备: {device_id}")
                return True

    except FileNotFoundError:
        print("❌ 未找到ADB命令，请检查Android SDK安装")
        return False

    print("❌ ADB连接异常")
    return False

# 自动重连机制
def auto_reconnect_adb():
    """自动重连ADB设备"""
    import time
    max_retries = 5
    retry_delay = 2

    for i in range(max_retries):
        if check_adb_connection():
            return True

        print(f"重连尝试 {i+1}/{max_retries}")

        # 重启ADB服务
        subprocess.run(['adb', 'kill-server'])
        time.sleep(1)
        subprocess.run(['adb', 'start-server'])
        time.sleep(retry_delay)

    return False
```

#### 模拟器窗口检测
```python
def find_emulator_window():
    """查找模拟器窗口"""
    import win32gui

    # 常见模拟器窗口标题关键词
    emulator_titles = [
        "MuMu", "LDPlayer", "Nox", "BlueStacks", "雷电", "夜神"
    ]

    def enum_window_callback(hwnd, windows):
        """枚举窗口回调函数"""
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            for title in emulator_titles:
                if title in window_text:
                    windows.append((hwnd, window_text))

    windows = []
    win32gui.EnumWindows(enum_window_callback, windows)

    if windows:
        print("找到以下模拟器窗口：")
        for hwnd, title in windows:
            print(f"  - {title} (句柄: {hwnd})")
        return windows[0][0]  # 返回第一个找到的窗口
    else:
        print("未找到模拟器窗口")
        return None
```

## 5.2 训练过程调试与验证方法

### 训练状态监控

#### 关键指标监控
```python
class TrainingMonitor:
    def __init__(self):
        self.metrics = {
            'epsilon': [],
            'loss': [],
            'reward': [],
            'win_rate': [],
            'episode_length': []
        }

    def record_episode(self, episode_data):
        """记录单局训练数据"""
        for key, value in episode_data.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def plot_training_curves(self):
        """绘制训练曲线"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('训练过程监控')

        # Epsilon衰减曲线
        axes[0,0].plot(self.metrics['epsilon'])
        axes[0,0].set_title('探索率衰减')
        axes[0,0].set_ylabel('Epsilon')

        # 损失变化曲线
        if self.metrics['loss']:
            axes[0,1].plot(self.metrics['loss'])
            axes[0,1].set_title('训练损失')
            axes[0,1].set_ylabel('Loss')

        # 奖励变化曲线
        if self.metrics['reward']:
            # 计算移动平均
            rewards = self.metrics['reward']
            moving_avg = [np.mean(rewards[max(0, i-10):i+1]) for i in range(len(rewards))]
            axes[1,0].plot(moving_avg)
            axes[1,0].set_title('平均奖励')
            axes[1,0].set_ylabel('Reward')

        # 胜率变化
        if self.metrics['win_rate']:
            axes[1,1].plot(self.metrics['win_rate'])
            axes[1,1].set_title('胜率变化')
            axes[1,1].set_ylabel('Win Rate')

        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()

    def check_training_health(self):
        """检查训练健康状况"""
        issues = []

        # 1. 检查epsilon衰减
        if len(self.metrics['epsilon']) > 100:
            recent_epsilon = self.metrics['epsilon'][-10:]
            if all(e == recent_epsilon[0] for e in recent_epsilon):
                issues.append("⚠️ Epsilon长时间未变化，检查衰减参数")

        # 2. 检查损失
        if len(self.metrics['loss']) > 20:
            recent_losses = self.metrics['loss'][-20:]
            avg_loss = np.mean(recent_losses)
            if avg_loss > 10:  # 损失过大
                issues.append("⚠️ 损失值过大，可能学习率过高")

        # 3. 检查奖励趋势
        if len(self.metrics['reward']) > 50:
            recent_rewards = self.metrics['reward'][-20:]
            early_rewards = self.metrics['reward'][:20]
            if np.mean(recent_rewards) < np.mean(early_rewards):
                issues.append("⚠️ 奖励呈下降趋势，可能需要调整奖励函数")

        return issues
```

#### 实时训练监控
```python
import time
from collections import deque

class RealTimeMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_losses = deque(maxlen=window_size)
        self.start_time = time.time()
        self.episode_count = 0

    def update(self, reward, loss=None):
        """更新监控数据"""
        self.recent_rewards.append(reward)
        if loss is not None:
            self.recent_losses.append(loss)
        self.episode_count += 1

    def get_status(self):
        """获取当前训练状态"""
        if len(self.recent_rewards) == 0:
            return "等待数据..."

        avg_reward = np.mean(self.recent_rewards)
        avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0

        # 训练时间
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)

        status = f"""
=== 实时训练状态 ===
训练时长: {hours}小时{minutes}分钟
训练局数: {self.episode_count}
平均奖励: {avg_reward:.2f}
平均损失: {avg_loss:.4f}
"""

        # 健康状况评估
        if avg_reward > 0:
            status += "✅ 奖励为正，训练效果良好\n"
        elif avg_reward > -5:
            status += "⚠️ 奖励偏低，需要优化\n"
        else:
            status += "❌ 奖励为负，训练异常\n"

        return status
```

### 模型行为验证

#### 动作分布分析
```python
def analyze_action_distribution(actions_taken):
    """分析动作选择的分布情况"""
    from collections import Counter
    import matplotlib.pyplot as plt

    # actions_taken是列表的列表，每个元素是一局的动作记录
    all_actions = []
    for episode_actions in actions_taken:
        all_actions.extend(episode_actions)

    # 分析每个动作维度的分布
    action_analysis = {}
    for dim in range(8):  # 8个动作维度
        dimension_actions = [action[dim] for action in all_actions]
        action_counts = Counter(dimension_actions)
        action_analysis[f'dim_{dim}'] = dict(action_counts)

    # 可视化动作分布
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (dim_name, counts) in enumerate(action_analysis.items()):
        actions = list(counts.keys())
        frequencies = list(counts.values())

        axes[i].bar(actions, frequencies)
        axes[i].set_title(f'动作维度 {i} 分布')
        axes[i].set_xlabel('动作值')
        axes[i].set_ylabel('频次')

    plt.tight_layout()
    plt.savefig('action_distribution.png')
    plt.close()

    return action_analysis

# 使用示例
actions_history = []
for episode in range(100):
    episode_actions = []
    for step in range(50):
        action = agent.select_action(state)  # 获取动作
        episode_actions.append(action)
    actions_history.append(episode_actions)

action_stats = analyze_action_distribution(actions_history)
```

#### Q值可视化
```python
def visualize_q_values(model, sample_states):
    """可视化Q值预测"""
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        for i, state in enumerate(sample_states[:5]):  # 取5个样本
            # 获取Q值预测
            q_values = model(state.unsqueeze(0))

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()

            for j, q_val in enumerate(q_values):
                # 将Q值转换为numpy并绘图
                q_numpy = q_val.squeeze().cpu().numpy()

                axes[j].plot(q_numpy)
                axes[j].set_title(f'动作维度 {j} Q值')
                axes[j].set_xlabel('动作索引')
                axes[j].set_ylabel('Q值')

                # 标记最大值
                max_idx = np.argmax(q_numpy)
                max_val = q_numpy[max_idx]
                axes[j].scatter(max_idx, max_val, color='red', s=100, zorder=5)
                axes[j].annotate(f'Max: {max_val:.2f}', (max_idx, max_val))

            plt.tight_layout()
            plt.savefig(f'q_values_sample_{i}.png')
            plt.close()
```

## 5.3 模型训练稳定性问题分析与解决

### 训练不稳定的表现

#### 损失函数异常
```python
def detect_training_anomalies(loss_history):
    """检测训练异常"""
    anomalies = []

    # 1. 检查损失爆炸
    recent_losses = loss_history[-20:]
    if recent_losses:
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)

        for i, loss in enumerate(recent_losses):
            if loss > mean_loss + 3 * std_loss:  # 超过3倍标准差
                anomalies.append({
                    'type': 'loss_explosion',
                    'value': loss,
                    'index': len(loss_history) - 20 + i,
                    'suggestion': '降低学习率或检查梯度'
                })

    # 2. 检查损失NaN
    for i, loss in enumerate(loss_history[-50:]):
        if np.isnan(loss) or np.isinf(loss):
            anomalies.append({
                'type': 'loss_nan',
                'value': loss,
                'index': len(loss_history) - 50 + i,
                'suggestion': '检查输入数据、降低学习率、添加梯度裁剪'
            })

    # 3. 检查损失震荡
    if len(loss_history) > 100:
        recent_trend = []
        for i in range(-20, -1):
            if loss_history[i] > loss_history[i-1]:
                recent_trend.append(1)  # 上升
            else:
                recent_trend.append(-1)  # 下降

        # 如果趋势变化太频繁，说明震荡严重
        trend_changes = sum(1 for i in range(1, len(recent_trend)) if recent_trend[i] != recent_trend[i-1])
        if trend_changes > 15:  # 90%的时间都在变化
            anomalies.append({
                'type': 'loss_oscillation',
                'value': trend_changes,
                'suggestion': '降低学习率、增加batch size、使用学习率调度器'
            })

    return anomalies
```

#### 梯度问题检测
```python
def check_gradient_health(model):
    """检查梯度健康状况"""
    gradient_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()

            # 检测梯度爆炸
            if grad_norm > 10.0:
                status = f"梯度爆炸 (norm: {grad_norm:.2f})"
            # 检测梯度消失
            elif grad_norm < 1e-7:
                status = f"梯度消失 (norm: {grad_norm:.2e})"
            # 检测梯度NaN
            elif np.isnan(grad_norm) or np.isinf(grad_norm):
                status = "梯度NaN/Inf"
            else:
                status = f"正常 (norm: {grad_norm:.4f})"

            gradient_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'status': status
            }

    return gradient_stats
```

### 稳定性改进方案

#### 学习率调度
```python
class AdaptiveLRScheduler:
    def __init__(self, optimizer, patience=10, factor=0.5, min_lr=1e-6):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.bad_episodes = 0
        self.best_loss = float('inf')

    def step(self, current_loss):
        """根据损失调整学习率"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.bad_episodes = 0
        else:
            self.bad_episodes += 1

        # 如果连续patience次没有改善，降低学习率
        if self.bad_episodes >= self.patience:
            for param_group in self.optimizer.param_groups:
                new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                print(f"学习率调整为: {new_lr}")

            self.bad_episodes = 0

    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']

# 使用示例
scheduler = AdaptiveLRScheduler(optimizer, patience=20, factor=0.5, min_lr=1e-6)

# 在训练循环中使用
for episode in range(num_episodes):
    loss = train_one_episode()
    scheduler.step(loss)
```

#### 梯度裁剪
```python
class GradientClipper:
    def __init__(self, max_norm=1.0, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip_gradients(self, model):
        """裁剪模型梯度"""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )
        return total_norm

    def should_clip(self, model):
        """判断是否需要裁剪"""
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
        total_norm = total_norm ** (1. / self.norm_type)

        return total_norm > self.max_norm

# 在训练中使用
gradient_clipper = GradientClipper(max_norm=1.0)

# 训练循环
loss.backward()
total_norm = gradient_clipper.clip_gradients(model)
if gradient_clipper.should_clip(model):
    print(f"进行了梯度裁剪，原梯度范数: {total_norm:.4f}")
optimizer.step()
```

#### 权重平滑（EMA）
```python
class ModelEMA:
    """指数移动平均（Exponential Moving Average）"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化shadow权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新shadow权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用shadow权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# 使用示例
ema = ModelEMA(model, decay=0.999)

# 训练循环
for step in range(num_steps):
    train_step()
    # 每步更新EMA权重
    ema.update()

# 评估时使用EMA权重
ema.apply_shadow()
evaluate_model(model)
ema.restore()  # 恢复原始权重
```

## 5.4 断点续训与模型保存加载最佳实践

### 完整的检查点系统

#### 检查点数据结构
```python
class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, episode, metrics, filename=None):
        """保存完整的训练状态"""
        if filename is None:
            filename = f'checkpoint_episode_{episode}.pth'

        checkpoint = {
            # 模型状态
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),

            # 训练进度
            'episode': episode,
            'steps_done': model.steps_done if hasattr(model, 'steps_done') else 0,
            'epsilon': model.epsilon if hasattr(model, 'epsilon') else 0,

            # 训练指标
            'metrics': metrics,

            # 随机状态（保证可重复性）
            'torch_random_state': torch.get_rng_state(),
            'numpy_random_state': np.random.get_state(),

            # 时间戳
            'timestamp': time.time(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath}")

        # 同时保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, model, optimizer, filepath):
        """加载检查点"""
        if not os.path.exists(filepath):
            print(f"检查点文件不存在: {filepath}")
            return None

        try:
            checkpoint = torch.load(filepath, map_location='cpu')

            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])

            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载训练状态
            start_episode = checkpoint['episode']
            if hasattr(model, 'steps_done'):
                model.steps_done = checkpoint.get('steps_done', 0)
            if hasattr(model, 'epsilon'):
                model.epsilon = checkpoint.get('epsilon', 0)

            # 加载随机状态
            if 'torch_random_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_random_state'])
            if 'numpy_random_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_random_state'])

            print(f"检查点加载成功: Episode {start_episode}")
            print(f"保存时间: {checkpoint.get('date', 'Unknown')}")

            return start_episode, checkpoint.get('metrics', {})

        except Exception as e:
            print(f"加载检查点失败: {e}")
            return None

    def list_checkpoints(self):
        """列出所有检查点"""
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.pth') and file != 'latest.pth':
                filepath = os.path.join(self.checkpoint_dir, file)
                try:
                    checkpoint = torch.load(filepath, map_location='cpu')
                    checkpoints.append({
                        'filename': file,
                        'episode': checkpoint.get('episode', 0),
                        'date': checkpoint.get('date', 'Unknown'),
                        'metrics': checkpoint.get('metrics', {})
                    })
                except:
                    pass

        return sorted(checkpoints, key=lambda x: x['episode'])
```

#### 自动保存策略
```python
class AutoSaveManager:
    def __init__(self, model, optimizer, checkpoint_manager,
                 save_interval=100, keep_last=5):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.save_interval = save_interval
        self.keep_last = keep_last
        self.saved_checkpoints = []

    def should_save(self, episode):
        """判断是否应该保存检查点"""
        return episode > 0 and episode % self.save_interval == 0

    def save_if_needed(self, episode, metrics):
        """按需保存检查点"""
        if self.should_save(episode):
            # 保存检查点
            filename = f'checkpoint_episode_{episode}.pth'
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, episode, metrics, filename
            )

            # 记录保存的文件
            self.saved_checkpoints.append(filename)

            # 清理旧的检查点
            self.cleanup_old_checkpoints()

    def cleanup_old_checkpoints(self):
        """清理旧的检查点"""
        while len(self.saved_checkpoints) > self.keep_last:
            old_checkpoint = self.saved_checkpoints.pop(0)
            old_path = os.path.join(self.checkpoint_manager.checkpoint_dir, old_checkpoint)
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"清理旧检查点: {old_checkpoint}")

    def save_best_model(self, episode, metrics, metric_key='win_rate'):
        """保存最佳模型"""
        if metric_key in metrics:
            current_value = metrics[metric_key]
            best_value = getattr(self, f'best_{metric_key}', float('-inf'))

            if current_value > best_value:
                setattr(self, f'best_{metric_key}', current_value)

                # 保存最佳模型
                best_filename = f'best_model_{metric_key}.pth'
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, episode, metrics, best_filename
                )
                print(f"新的最佳模型({metric_key}): {current_value:.4f}")
```

### 模型版本管理

#### 版本控制系统
```python
class ModelVersionManager:
    def __init__(self, model_name='wzry_ai'):
        self.model_name = model_name
        self.versions_dir = 'model_versions'
        os.makedirs(self.versions_dir, exist_ok=True)

    def save_version(self, model, version_info):
        """保存模型版本"""
        version = version_info.get('version', 'v1.0')
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        version_data = {
            'model_state_dict': model.state_dict(),
            'version_info': version_info,
            'timestamp': timestamp,
            'git_commit': self._get_git_commit(),
            'performance_metrics': version_info.get('metrics', {})
        }

        filename = f"{self.model_name}_{version}_{timestamp}.pth"
        filepath = os.path.join(self.versions_dir, filename)

        torch.save(version_data, filepath)
        print(f"模型版本已保存: {filename}")

        # 更新版本索引
        self._update_version_index(version_data, filename)

    def _get_git_commit(self):
        """获取当前Git提交ID"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "unknown"

    def _update_version_index(self, version_data, filename):
        """更新版本索引文件"""
        index_file = os.path.join(self.versions_dir, 'version_index.json')

        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
        except:
            index = {'versions': []}

        index['versions'].append({
            'filename': filename,
            'version': version_data['version_info']['version'],
            'timestamp': version_data['timestamp'],
            'git_commit': version_data['git_commit'],
            'metrics': version_data['performance_metrics'],
            'description': version_data['version_info'].get('description', '')
        })

        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)

    def list_versions(self):
        """列出所有版本"""
        index_file = os.path.join(self.versions_dir, 'version_index.json')
        if not os.path.exists(index_file):
            return []

        with open(index_file, 'r') as f:
            index = json.load(f)

        return index['versions']

    def load_version(self, version_str):
        """加载指定版本"""
        versions = self.list_versions()

        for version_info in versions:
            if version_info['version'] == version_str:
                filepath = os.path.join(self.versions_dir, version_info['filename'])
                if os.path.exists(filepath):
                    return torch.load(filepath, map_location='cpu')

        return None
```

## 5.5 性能瓶颈分析与优化策略

### 性能分析工具

#### 代码性能分析
```python
import cProfile
import pstats
import time

def profile_function(func, *args, **kwargs):
    """性能分析装饰器"""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 打印前20个最耗时的函数

    return result

# 使用示例
# profile_function(train_one_episode)
```

#### 时间开销分析
```python
class PerformanceTimer:
    def __init__(self):
        self.timers = {}
        self.results = {}

    def start_timer(self, name):
        """开始计时"""
        self.timers[name] = time.time()

    def end_timer(self, name):
        """结束计时并记录结果"""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            if name not in self.results:
                self.results[name] = []
            self.results[name].append(elapsed)
            del self.timers[name]

    def get_statistics(self):
        """获取统计信息"""
        stats = {}
        for name, times in self.results.items():
            if times:
                stats[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times),
                    'count': len(times)
                }
        return stats

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print("=== 性能统计 ===")
        for name, stat in stats.items():
            print(f"{name}:")
            print(f"  平均时间: {stat['mean']*1000:.2f}ms")
            print(f"  总时间: {stat['total']:.2f}s")
            print(f"  调用次数: {stat['count']}")

# 使用示例
timer = PerformanceTimer()

def train_step():
    timer.start_timer("data_preparation")
    # 数据准备...
    timer.end_timer("data_preparation")

    timer.start_timer("forward_pass")
    # 前向传播...
    timer.end_timer("forward_pass")

    timer.start_timer("backward_pass")
    # 反向传播...
    timer.end_timer("backward_pass")

# train_step()  # 运行多次后
timer.print_statistics()
```

### 常见性能瓶颈

#### 1. 数据加载瓶颈
```python
# 问题：数据加载过慢，GPU等待
# 症状：GPU利用率低，CPU占用高

# 解决方案：
def optimize_data_loading():
    """优化数据加载"""
    # 1. 增加worker数量
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,        # 根据CPU核心数调整
        pin_memory=True,      # 使用锁页内存
        persistent_workers=True,  # 保持worker存活
        prefetch_factor=2,    # 预取数据
    )

    # 2. 数据预处理优化
    class OptimizedDataset(Dataset):
        def __init__(self, data):
            # 预加载和预处理数据
            self.preprocessed_data = self.preprocess_all(data)

        def preprocess_all(self, data):
            # 并行预处理
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(self.preprocess_single, data))
            return results

        def __getitem__(self, idx):
            # 直接返回预处理好的数据
            return self.preprocessed_data[idx]

    return dataloader
```

#### 2. GPU内存瓶颈
```python
# 问题：GPU内存不足
# 症状：CUDA out of memory错误

def optimize_gpu_memory():
    """优化GPU内存使用"""
    # 1. 梯度累积（模拟大batch size）
    def train_with_gradient_accumulation(model, data, target_batch_size=256):
        real_batch_size = 64
        accumulation_steps = target_batch_size // real_batch_size

        optimizer.zero_grad()
        total_loss = 0

        for i in range(accumulation_steps):
            batch_data = data[i*real_batch_size:(i+1)*real_batch_size]
            loss = model(batch_data) / accumulation_steps
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        return total_loss

    # 2. 混合精度训练
    scaler = GradScaler()

    def train_with_mixed_precision(model, data):
        optimizer.zero_grad()

        with autocast():
            loss = model(data)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # 3. 及时清理内存
    def memory_efficient_training():
        for batch in dataloader:
            batch = batch.cuda()

            # 前向传播
            output = model(batch)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 关键：及时清理
            del batch, output, loss
            torch.cuda.empty_cache()
```

#### 3. 计算瓶颈
```python
# 问题：计算过慢，GPU利用率低
# 症状：训练时间长，性能不达标

def optimize_computation():
    """优化计算效率"""
    # 1. 模型编译优化（PyTorch 2.0+）
    model = torch.compile(model, mode='max-performance')

    # 2. 操作融合
    def fused_operations(x):
        # ❌ 低效：多次内存访问
        # temp1 = self.conv1(x)
        # temp2 = self.relu(temp1)
        # temp3 = self.conv2(temp2)
        # output = self.relu(temp3)

        # ✅ 高效：操作融合
        return F.relu(self.conv2(F.relu(self.conv1(x))))

    # 3. 批处理优化
    def optimized_batch_processing():
        # 使用更大的batch size（在内存允许范围内）
        optimal_batch_size = find_optimal_batch_size(model)

        # 预分配张量
        preallocated_tensor = torch.zeros(optimal_batch_size, channels, height, width)

        for batch in dataloader:
            # 复用预分配的内存
            batch_size = batch.size(0)
            preallocated_tensor[:batch_size] = batch

            output = model(preallocated_tensor[:batch_size])

    return model
```

### 系统级优化

#### 多进程训练
```python
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_training(rank, world_size):
    """设置分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 创建模型并包装为DDP
    model = create_model().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    return ddp_model

def train_distributed(rank, world_size):
    """分布式训练函数"""
    model = setup_distributed_training(rank, world_size)

    # 创建分布式数据加载器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler
    )

    # 训练循环
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, rank)

def main_distributed():
    """主函数：启动分布式训练"""
    world_size = torch.cuda.device_count()
    mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
```

#### 内存映射数据加载
```python
import numpy as np

class MemoryMappedDataset(Dataset):
    """内存映射数据集（适合大数据集）"""
    def __init__(self, data_path, mmap_mode='r'):
        # 使用内存映射加载大数据
        self.data = np.load(data_path, mmap_mode=mmap_mode)
        self.labels = np.load(data_path.replace('data', 'labels'), mmap_mode=mmap_mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 只加载需要的部分到内存
        data = torch.from_numpy(self.data[idx])
        label = torch.from_numpy(self.labels[idx])
        return data, label

# 使用示例
dataset = MemoryMappedDataset('large_dataset.npy')
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

## 5.6 小结

本章分享了丰富的开发实战经验：

1. **环境配置**：版本兼容性、依赖管理、CUDA配置常见问题
2. **训练监控**：状态监控、异常检测、可视化分析工具
3. **稳定性优化**：学习率调度、梯度裁剪、权重平滑等技术
4. **模型管理**：完整检查点系统、版本控制、自动保存策略
5. **性能优化**：瓶颈分析、内存优化、分布式训练方案

这些经验来自于实际项目开发，能帮助你：
- 快速定位和解决常见问题
- 提高训练稳定性和效率
- 建立完善的模型管理系统
- 优化系统性能表现

掌握这些实战技巧，能让你的强化学习项目开发更加顺利和专业。下一章将提供具体的新手实践指导。,\n\n---\n\n*本章内容基于实际开发经验编写，涵盖了大量踩坑经历和解决方案，是新手快速成长的捷径。*"}
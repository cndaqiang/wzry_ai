# 03 · 强化学习基础（面向监督学习转型者）

这里用你熟悉的监督学习思维切入：把 `state` 看作输入，`reward` + 下一状态的预期收益替代“标签”，而 Q 网络输出的每个动作值类似“多标签分类”的 logits。

## 1. 从监督学习到状态-动作-奖励
| 概念 | 监督学习对应 | 在本项目里的实现 |
| --- | --- | --- |
| 输入 `state` | 图片/特征向量 | `AndroidTool.screenshot_window()` 得到的 640×640×3 图像。|
| 标签 | 明确的 `y`（例如类别） | 强化学习没有直接标签，用 `reward + γ * max Q(next_state)` 作为学习目标。|
| 模型输出 | 分类概率 | `NetDQN` 输出 8 组 Q 值：移动/角度/信息/攻击等（`net_actor.py:5-48`）。|
| 损失 | `loss(y_pred, y_true)` | `nn.MSELoss(state_action_q, expected_q)`（`dqnAgent.py:139-158`）。|

理解这一点后，你只需把“标签”换成 Bellman 方程定义的期望价值即可。

## 2. Bellman 更新在代码中的步骤
以下对应 `dqnAgent.py:74-174`：
1. **采样 batch**：`transitions = globalInfo.random_batch_size_memory_dqn()`；`batch_state` 形状 `[B,3,640,640]`。
2. **前向计算**：`state_action_values = policy_net(batch_state)`，得到 8 个张量，例如 `move_action_q` 形状 `[B,2]`。
3. **选取执行过的动作的 Q 值**：
```python
move_q = move_action_q.gather(1, batch_action[:,0].unsqueeze(1))
...
state_action_q_values = move_q + angle_q + ... + arg3_q
```
   这里的 `gather` 相当于“用 one-hot 选中真实动作的预测值”，和监督学习里挑选对应类别的 logit 很像。
4. **计算下一状态的最大 Q**：
```python
next_state_values[non_final_mask] = (
    torch.max(next_move_action_q,1)[0] + ... + torch.max(next_arg3_q,1)[0]
)
```
5. **构造目标**：`expected_q = reward + gamma * next_state_values * (1 - done)`。
6. **回传**：`loss = MSE(state_action_q_values, expected_q.unsqueeze(1))`，然后 `loss.backward(); optimizer.step()`。

这就是 Bellman 方程 `Q(s,a) ← r + γ * max_a' Q(s', a')` 的代码化版本。

## 3. epsilon 策略与分阶段设定
- **epsilon-greedy**：`select_action()` 中 `rand <= epsilon` 时走随机动作，否则走网络预测。
- **参数**：`argparses.py:85-90` 默认 `epsilon=1.5`，`epsilon_decay=0.995`，`epsilon_min=0.01`。
- **实践建议**：
  1. **冷启动**：如果没有预训练模型，保持 `epsilon≈1` 收集尽量多的多样样本，尤其是胜/负/死亡等关键状态。
  2. **分阶段下降**：将衰减拆成多个阶段，例如先把 epsilon 降到 0.5，再根据 reward 表现继续下降，避免过早贪心导致陷入局部策略。
  3. **经验池加权**：可以优先采样“高信息量”片段，比如奖励绝对值大的样本，减小随机动作的噪声。
  4. **日志**：在 `select_action` 打印 `rand` 和 `epsilon` 已经帮你监控；建议把它写进文件或 TensorBoard 方便观察趋势。

## 4. “随机样本 vs Bellman”问题回答
> 早期模型几乎随机，采集到的样本不代表“真实动作价值”，这是否违背 Bellman 方程？

- **不会冲突**：Bellman 更新只要求 `(s,a,r,s')` 是真实体验，而不要求 `a` 是“最优动作”。随机探索得到的样本恰好提供了多样的状态转移，帮助估计真实的 `Q(s,a)`。
- **减轻噪声的方法**：
  1. **经验回放**：随机抽样打散时间相关性，避免连续随机动作产生偏差。
  2. **目标网络**：`target_net` 的延迟同步（`self.steps_done % target_update == 0`）让目标值更稳定。
  3. **奖励设计**：`getReword.py:94-141` 给移动方向额外加权，帮助模型更快学到“往前走更好”，即使动作随机也能通过奖励信号引导方向。
  4. **数据阶段化**：你可以先运行一段时间只收集经验，再开始训练；或设置“采集线程快于训练线程”，保证池子持续刷新。

## 5. 新手开发经验
1. **检查奖励来源**：随机动作阶段 reward 经常是 -1；如果长时间没有出现正奖励，优先调试 `GetRewordUtil`（比如攻击识别 HSV 阈值）。
2. **缩小动作空间**：初期可以只学习 `move_action` + `angle`，把其他维度固定为“无操作”，这样 Q 网络输出更易收敛。
3. **可视化奖励曲线**：利用 `GlobalInfo.update_data_file` 写入 JSON，再用 matplotlib 画出 reward/episode，帮助判断训练是否稳定。
4. **保存/加载模型**：`DQNAgent.save_model('src/wzry_ai.pt')`；当你重新开始训练时，记得把 `args.model_path` 指向旧模型并调整 epsilon（例如 0.1）。
5. **调 batch size 与 gamma**：
   - 帧率低、动作慢时，可将 `gamma` 降到 0.9 左右，避免过度依赖远期回报。
   - 如果训练易震荡，可以减小 `batch_size`（例如 32）并降低学习率。
6. **多线程同步**：在训练线程退出前要记得 `training_thread.join()` 或捕获异常，避免出现“采集线程继续往经验池写而训练已停”的情况。

掌握这些基础后，你就具备了实现和调试 DQN 的能力，也能更快理解 04 章里的扩展方案。

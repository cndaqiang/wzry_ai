import os

import cv2
import numpy as np
import torch
from torch import optim, nn

# 这个device是cuda设备, 参数(模型位置,epsilon等), batch参数等
from argparses import device, args, globalInfo
from memory import Transition
from net_actor import NetDQN

class DQNAgent:
    def __init__(self):
        torch.backends.cudnn.enabled = False

        self.action_sizes = [2, 360, 9, 11, 3, 360, 100, 5]
        # 读取 argparses 中的参数
        self.device = device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.learning_rate = args.learning_rate

        self.steps_done = 0
        self.target_update = args.target_update

        self.policy_net = NetDQN().to(self.device)
        self.target_net = NetDQN().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        if args.model_path and os.path.exists(args.model_path):
            # 加载用于预测的模型 `src/wzry_ai.pt`
            self.policy_net.load_state_dict(torch.load(args.model_path))
            # 复制权重，初始化 target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())
            #训练过程中（如 DQN 等强化学习算法），policy_net 会不断更新，
            # 而 target_net 会定期或延迟地从 policy_net 同步权重，以保持训练稳定性。
            print(f"Model loaded from {args.model_path}")

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def select_action(self, state):
        #cndaqiang debug
        # 默认情况下,  epsilon == 1, 所以总是随机选择动作
        rand = np.random.rand()
        print(f"-->Random number for epsilon-greedy: {rand}, Epsilon: {self.epsilon}, logical: {rand <= self.epsilon}")
        #if np.random.rand() <= self.epsilon:
        if rand <= self.epsilon:
            return [np.random.randint(size) for size in self.action_sizes]
        # 一组随机整数 self.action_sizes = [2, 360, 9, 11, 3, 360, 100, 5]
        # [0不动1动, 0-359移动角度, 0-8信息操作, 0-10攻击对象(普攻、小兵、回血、技能、不攻击), 0-2动作类型(点击、滑动、长按), 0-359参数1(滑动角度), 0-99参数2(滑动距离), 0-4参数3(长按时间)]
        # 处理图片, 并推送到device,[channels/RGB,h,w]
        tmp_state_640_640 = self.preprocess_image(state).unsqueeze(0)
        # 采用policy_net网络,进入评估模式
        self.policy_net.eval()
        with torch.no_grad():
            #输入图片,返回预测的动作，每个action返回的是个数组,这里提取数组的最大值位置
            q_values = self.policy_net(tmp_state_640_640)
        return [np.argmax(q.detach().cpu().numpy()) for q in q_values]

    def preprocess_image(self, image, target_size=(640, 640)):
        # 调整图像大小
        resized_image = cv2.resize(image, target_size)
        # 转换为张量并调整维度顺序 [height, width, channels] -> [channels, height, width]
        tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1)
        return tensor_image.to(device)

    def replay(self):
        transitions = globalInfo.random_batch_size_memory_dqn()
        batch = Transition(*zip(*transitions))

        # 将 batch 转换为张量，并移动到设备上
        batch_state = torch.stack([self.preprocess_image(state) for state in batch.state]).to(device)
        batch_action = torch.LongTensor(batch.action).to(self.device)
        batch_reward = torch.FloatTensor(batch.reward).to(self.device)
        batch_next_state = torch.stack([self.preprocess_image(state) for state in batch.next_state]).to(device)
        batch_done = torch.FloatTensor(batch.done).to(self.device)

        # 计算当前状态的 Q 值
        # 利用实时网络计算
        state_action_values = self.policy_net(batch_state)

        # 计算每个动作类别的 Q 值
        move_action_q, angle_q, info_action_q, attack_action_q, action_type_q, arg1_q, arg2_q, arg3_q = state_action_values

        # 与评估不同, 此时输入的是[batch]个state, 返回动作Q也是[batch维度]

        # 选择执行的动作的 Q 值
        # 模型输出的各个 *_q 是不同动作分量对应的 Q 值矩阵，例如 angle_q.shape = [batch, 360]，
        # 表示每个状态下，对应角度 1~360 的 Q 值。
        # batch_action 记录了每个样本实际执行的动作索引，比如 angle=120。
        # gather(1, batch_action[:, i].unsqueeze(1)) 用于提取该维度上实际执行动作的 Q 值，
        # 即 angle_q[batch, 120]。
        # 各维度的 Q 值相加，得到组合动作的总 Q 值。
        # 2. 是否需要归一化Q值矩阵？
        # 不需要显式归一化。模型会自动学习到各分量 Q 值的相对尺度。
        # 什么时候 可能 需要归一化？各动作维度的取值空间差异巨大
        # 比如 angle 取 1–360，而 move 只有 5 种；此时可能导致梯度不平衡或数值主导。
        # 若前期训练不稳定, 可以给每个分支加 learnable weight 或 LayerNorm，如angle_q / torch.norm(angle_q, dim=1, keepdim=True)
        # 
        # 这个Q矩阵是一个得分矩阵Q, 我们按照Q最大的位置操作(得分最多)得分Q[index]
        # ⭐ 模型预测的是不同操作index的得分Q[index]
        # 模型的得分Q[index]与实际得分的差值 Q[index]-rward就是损失函数
        #
        state_action_q_values = move_action_q.gather(1, batch_action[:, 0].unsqueeze(1)) + \
                                angle_q.gather(1, batch_action[:, 1].unsqueeze(1)) + \
                                info_action_q.gather(1, batch_action[:, 2].unsqueeze(1)) + \
                                attack_action_q.gather(1, batch_action[:, 3].unsqueeze(1)) + \
                                action_type_q.gather(1, batch_action[:, 4].unsqueeze(1)) + \
                                arg1_q.gather(1, batch_action[:, 5].unsqueeze(1)) + \
                                arg2_q.gather(1, batch_action[:, 6].unsqueeze(1)) + \
                                arg3_q.gather(1, batch_action[:, 7].unsqueeze(1))

        # 计算下一个状态的 Q 值
        # 挑选对局中的有效状态,(done = 1 胜利/失败)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        non_final_mask = (batch_done == 0)
        non_final_next_states = batch_next_state[non_final_mask]
        #
        # 都是有效数据时, 利用稳定网络计算操作
        if non_final_next_states.size(0) > 0:
            next_state_action_values = self.target_net(non_final_next_states)
            next_move_action_q, next_angle_q, next_info_action_q, next_attack_action_q, next_action_type_q, next_arg1_q, next_arg2_q, next_arg3_q = next_state_action_values
            next_state_values[non_final_mask] = torch.max(next_move_action_q, 1)[0] + \
                                                torch.max(next_angle_q, 1)[0] + \
                                                torch.max(next_info_action_q, 1)[0] + \
                                                torch.max(next_attack_action_q, 1)[0] + \
                                                torch.max(next_action_type_q, 1)[0] + \
                                                torch.max(next_arg1_q, 1)[0] + \
                                                torch.max(next_arg2_q, 1)[0] + \
                                                torch.max(next_arg3_q, 1)[0]

        # 计算期望的 Q 值
        # batch_done == 0，才参与比较
        # 为什么，要再期待的q值上加权?
        expected_state_action_values = batch_reward + self.gamma * next_state_values * (1 - batch_done)

        # 直观的理解是模型的得分Q[index]与实际得分的差值 Q[index]-rward就是损失函数
        # 通过这样可以学习怎样动作index,得分最高
        # 若不考虑后续结果就可以了
        # 但是这里, Q[index] 实际预测的是当前步和下一步可以获得的总分：**当前动作之后的长期累计收益的期望**
        # 下一步的分数r1通过gamma权重添加, Q = r[0] + \gamma*r[1] + \gamma^2*r[2] + ...
        # 帧数越高时, gamma应该取的越大
        # 较小（0.8 以下）收敛快，策略贪婪，容易陷入局部最优
        # 较大（>0.98）收敛慢，需更多经验 学习更稳定、泛化更好
        # GPT 建议 从 0.9 起，逐步尝试 0.95、0.97、0.99；观察收敛与稳定性
        # ⭐正因如此，模型输出的Q是不能进行归一化的，（可以）但没必要设置不同操作的权重，会自动优化
        # 计算损失
        # self.criterion = nn.MSELoss()
        loss = self.criterion(state_action_q_values, expected_state_action_values.unsqueeze(1))

        print("loss", loss)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon 决定接下来的动作是网络生成, 还是随机生成
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 定期更新稳定版模型参数
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1


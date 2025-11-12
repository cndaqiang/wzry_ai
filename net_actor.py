import torch
import torch.nn as nn
import torch.nn.functional as F


# Actor 网络
class NetDQN(nn.Module):
    def __init__(self):
        super(NetDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)

        # ，线性层 fc 的输入维度必须提前算好，
        # 而计算依据就是“卷积部分在 640×640 图像上拉平后有多少个元素”
        conv_output_size = self._get_conv_output_size(640, 640)
        self.fc = nn.Linear(conv_output_size, 256)

        self.fc1 = nn.Linear(256, 256)
        self.fc_move = nn.Linear(256, 2)  # move_action_list Q-values
        self.fc_angle = nn.Linear(256, 360)  # angle_list Q-values
        self.fc_info = nn.Linear(256, 9)  # info_action_list Q-values
        self.fc_attack = nn.Linear(256, 11)  # attack_action_list Q-values
        self.fc_action_type = nn.Linear(256, 3)  # action_type_list Q-values
        self.fc_arg1 = nn.Linear(256, 360)  # arg1_list Q-values
        self.fc_arg2 = nn.Linear(256, 100)  # arg2_list Q-values
        self.fc_arg3 = nn.Linear(256, 5)  # arg3_list Q-values
        self._initialize_weights()

    def _get_conv_output_size(self, height, width):
        dummy_input = torch.zeros(1, 3, height, width)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
        return x.view(x.size(0), -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 卷积 - 卷积 - 线性到256, relu激活
        # self.parameters 模型张量
        # next(self.parameters) 模型张量的第一个数据
        # 就是为了把x推送到模型所在的device
        x = x.to( next( self.parameters() ).device )
        # conv1(输入的x可以是任意分辨率)
        # conv1的参数空间, 由nn.Conv2d(3, 64, kernel_size=8, stride=4)决定
        # 滑动后的结果由x决定, 当前参数下, x[...,m640]滑动后为[...,159]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        # 前面的x输入可能是任意的,在进行线性变换之前,我们要确定卷积的结果才能变换
        # 所以在init中,采用 conv_output_size = self._get_conv_output_size(640, 640)
        # 计算两层卷积后的结果, 即self.fc = nn.Linear(conv_output_size, 256)
        x = self.fc(x)

        x = F.relu(self.fc1(x))

        # 线性降到预测维度
        move_action_q = self.fc_move(x)
        angle_q = self.fc_angle(x)
        info_action_q = self.fc_info(x)
        attack_action_q = self.fc_attack(x)
        action_type_q = self.fc_action_type(x)
        arg1_q = self.fc_arg1(x)
        arg2_q = self.fc_arg2(x)
        arg3_q = self.fc_arg3(x)

        return move_action_q, angle_q, info_action_q, attack_action_q, action_type_q, arg1_q, arg2_q, arg3_q

# config.py
import argparse

import torch

from globalInfo import GlobalInfo

# cndaqiang: 如果仍然使用这个框架的话, 下面存放的就是相对坐标, 后面点击时分辨率*相对坐标, 与airtest的里面的饿相对坐标规则不同
# 注释后, 我改为了16:9屏幕的相对坐标, duration参数单位毫秒
move_actions_detail = {
    1: {'action_name': '移动', 'position': (0.172, 0.803), 'radius': 200, 'duration': 1000}
}
# 点击坐标
#  购买装备1， 购买装备2，发起进攻，开始撤退，请求集合，升级1技能，升级2技能，升级3技能，升级4技能
# 左侧的购买装备回合移动位置有冲突(轮盘跟随手指),暂时改为(0.0,0.0)
info_actions_detail = {
    1: {'action_name': '购买装备1', 'position': (0.0, 0.0), 'radius': 0}, #左侧购买(0.050, 0.390)
    2: {'action_name': '购买装备2', 'position': (0.0,0.0), 'radius': 0},
    3: {'action_name': '发起进攻', 'position': (0.970, 0.155), 'radius': 0},
    4: {'action_name': '开始撤退', 'position': (0.970, 0.230), 'radius': 0},
    5: {'action_name': '请求集合', 'position': (0.970, 0.300), 'radius': 0},
    6: {'action_name': '升级1技能', 'position': (0.655, 0.773), 'radius': 0},
    7: {'action_name': '升级2技能', 'position': (0.701, 0.604), 'radius': 0},
    8: {'action_name': '升级3技能', 'position': (0.805, 0.500), 'radius': 0}
}

# 无操作, 攻击，攻击小兵，攻击塔，回城，恢复，装备技能, 1技能，2技能，3技能,
attack_actions_detail = {
    1: {'action_name': '攻击', 'position': (0.866, 0.841), 'radius': 0},
    2: {'action_name': '攻击小兵', 'position': (0.790, 0.894), 'radius': 0},
    3: {'action_name': '攻击塔', 'position': (0.921, 0.721), 'radius': 0},
    4: {'action_name': '回城', 'position': (0.458, 0.894), 'radius': 0},
    5: {'action_name': '恢复', 'position': (0.534, 0.894), 'radius': 0},
    6: {'action_name': '装备技能', 'position': (0.868, 0.388), 'radius': 0},
    7: {'action_name': '召唤师技能', 'position': (0.604, 0.894), 'radius': 50},
    8: {'action_name': '1技能', 'position': (0.697, 0.873), 'radius': 100},
    9: {'action_name': '2技能', 'position': (0.755, 0.706), 'radius': 100},
    10: {'action_name': '3技能', 'position': (0.866, 0.574), 'radius': 100}
}

"""
# 移动坐标和滑动半径
move_actions_detail = {
    1: {'action_name': '移动', 'position': (0.164, 0.798), 'radius': 200}
}
# 点击坐标
#  购买装备1， 购买装备2，发起进攻，开始撤退，请求集合，升级1技能，升级2技能，升级3技能，升级4技能
info_actions_detail = {
    1: {'action_name': '购买装备1', 'position': (0.133, 0.4), 'radius': 0},
    2: {'action_name': '购买装备2', 'position': (0.133, 0.51), 'radius': 0},
    3: {'action_name': '发起进攻', 'position': (0.926, 0.14), 'radius': 0},
    4: {'action_name': '开始撤退', 'position': (0.926, 0.22), 'radius': 0},
    5: {'action_name': '请求集合', 'position': (0.926, 0.31), 'radius': 0},
    6: {'action_name': '升级1技能', 'position': (0.668, 0.772), 'radius': 0},
    7: {'action_name': '升级2技能', 'position': (0.717, 0.59), 'radius': 0},
    8: {'action_name': '升级3技能', 'position': (0.8, 0.48), 'radius': 0}
}

# 无操作, 攻击，攻击小兵，攻击塔，回城，恢复，装备技能, 1技能，2技能，3技能,
attack_actions_detail = {
    1: {'action_name': '攻击', 'position': (0.85, 0.85), 'radius': 0},
    2: {'action_name': '攻击小兵', 'position': (0.776, 0.91), 'radius': 0},
    3: {'action_name': '攻击塔', 'position': (0.88, 0.71), 'radius': 0},
    4: {'action_name': '回城', 'position': (0.518, 0.9), 'radius': 0},
    5: {'action_name': '恢复', 'position': (0.579, 0.9), 'radius': 0},
    6: {'action_name': '装备技能', 'position': (0.84, 0.39), 'radius': 0},
    7: {'action_name': '召唤师技能', 'position': (0.64, 0.9), 'radius': 50},
    8: {'action_name': '1技能', 'position': (0.71, 0.874), 'radius': 100},
    9: {'action_name': '2技能', 'position': (0.76, 0.69), 'radius': 100},
    10: {'action_name': '3技能', 'position': (0.844, 0.58), 'radius': 100}
}
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iphone_id', type=str, default='127.0.0.1:5555', help="iphone_id")
    parser.add_argument('--real_iphone', type=bool, default=False, help="real_iphone")
    #parser.add_argument('--window_title', type=str, default='SM-S9210', help="window_title")
    parser.add_argument('--window_title', type=str, default='MuMu安卓设备', help="window_title")
    parser.add_argument('--device_id', type=str, default='cuda:0', help="device_id")
    parser.add_argument('--memory_size', type=int, default=10000, help="Replay memory size")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    #epsilon = 1.0适合从头开始训练,所有运动都是随机的; 如果加载了预训练模型, 则可以将其设置为较低的值, 例如0.1
    #parser.add_argument('--epsilon', type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument('--epsilon', type=float, default=0.5, help="Initial exploration rate")
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help="Exploration rate decay")
    parser.add_argument('--epsilon_min', type=float, default=0.01, help="Minimum exploration rate")
    parser.add_argument('--model_path', type=str, default="src/wzry_ai.pt", help="Path to the model to load")
    parser.add_argument('--num_episodes', type=int, default=10, help="Number of episodes to collect data")
    parser.add_argument('--target_update', type=int, default=10, help="Number of episodes to collect data")

    return parser.parse_args()


# 解析参数并存储在全局变量中
args = get_args()

device = torch.device(args.device_id if torch.cuda.is_available() else 'cpu')

# 全局状态
globalInfo = GlobalInfo(batch_size=args.batch_size, buffer_capacity=args.memory_size)

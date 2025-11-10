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

# 全局状态
globalInfo = GlobalInfo()

class_names = ['started']
start_check = OnnxRunner('models/start.onnx', classes=class_names)

rewordUtil = GetRewordUtil()
tool = AndroidTool(airtest_config="config.example.yaml")
state = tool.screenshot_window()
tool.show_scrcpy()
# tool.show_action_log()
env = Environment(tool, rewordUtil)

agent = DQNAgent()

def data_collector():
    while True:
        # 获取当前的图像
        state = tool.screenshot_window()
        # 保证图像能正常获取
        if state is None:
            time.sleep(0.01)
            continue
        # cv2.imwrite('output_image.jpg', state)
        # 初始化对局状态 对局未开始
        globalInfo.set_game_end()
        # 判断对局是否开始
        checkGameStart = start_check.get_max_label(state)

        if checkGameStart == 'started':
            print("-------------------------------对局开始-----------------------------------")
            globalInfo.set_game_start()

            # 对局开始了，进行训练
            while globalInfo.is_start_game():
                # 获取预测动作
                print("---> 获取预测动作")
                action = agent.select_action(state)
                print(f"---> env.step(action)={action}")
                # move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3 = action
                # 移动(只有一个和移动的角度), 发信号(信号,装备,升级技能), 攻击(小兵,英雄,塔,回血,回城,技能,召唤师技能) 
                #下面就是执行action列表，并读取state(同上)为next_state=screenshot_window()
                # action = [0不动1动, 0-359移动角度, 0-8信息操作, 0-10攻击对象(普攻、小兵、回血、技能、不攻击), 0-2动作类型(点击、滑动、长按), 0-359参数1(滑动角度), 0-99参数2(滑动距离), 0-4参数3(长按时间)]
                next_state, reward, done, info = env.step(action)
                print("---> reward")
                # # 以及根据新截图判断权重calculate_reword, 根据胜利失败死亡进行赋值
                #目前返回的 info=None, reward=-1
                print(info, reward)

                # 对局结束
                if done == 1:
                    print("-------------------------------对局结束-----------------------------------")
                    globalInfo.set_game_end()
                    break

                # 追加经验
                globalInfo.store_transition_dqn(state, action, reward, next_state, done)

                state = next_state

        else:
            print("对局未开始")
            time.sleep(0.1)


def train_agent():
    count = 1
    while True:
        if not globalInfo.is_memory_bigger_batch_size_dqn():
            time.sleep(1)
            continue
        print("training")
        agent.replay()
        if count % args.num_episodes == 0:
            agent.save_model('src/wzry_ai.pt')
        count = count + 1
        if count >= 100000:
            count = 1


if __name__ == '__main__':
    training_thread = threading.Thread(target=train_agent)
    training_thread.start()
    data_collector()

import datetime
import json
import os
import threading

from filelock import FileLock

from memory import ReplayMemory

#cnq: 
# @singleton 保证 GlobalInfo 全进程只有一份实例，多次 new 都返回同一对象，全局共享同一块内存与配置

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class GlobalInfo:
    def __init__(self, batch_size=64, buffer_capacity=10000):
        self.batch_size = batch_size
        self._info = {}
        self.ppo_memory = ReplayMemory(buffer_capacity)
        self.td3_memory = ReplayMemory(buffer_capacity)
        self.dqn_memory = ReplayMemory(buffer_capacity)
        self.lock = threading.Lock()

    def set_value(self, key, value):
        self._info[key] = value

    def get_value(self, key):
        return self._info.get(key, None)

    # -------------------------------对局状态-------------------------------------
    def set_game_start(self):
        self.set_value('start_game', True)

    def is_start_game(self):
        start_game = self.get_value('start_game')
        if start_game is None:
            return False
        else:
            return start_game

    def set_game_end(self):
        self.set_value('start_game', False)

    # -------------------------------ppo经验池-------------------------------------
    def store_transition_ppo(self, *args):
        self.lock.acquire()
        try:
            self.ppo_memory.push(*args)
        finally:
            self.lock.release()

    def is_memory_bigger_batch_size_ppo(self):
        self.lock.acquire()
        try:
            if len(self.ppo_memory) < self.batch_size:
                return False
            else:
                return True
        finally:
            self.lock.release()

    def random_batch_size_memory_ppo(self):
        self.lock.acquire()
        try:
            transitions = self.ppo_memory.sample(self.batch_size)
            return transitions
        finally:
            self.lock.release()

    # -------------------------------td3经验池-------------------------------------
    def store_transition_td3(self, *args):
        self.td3_memory.push(*args)

    def is_memory_bigger_batch_size_td3(self):
        if len(self.td3_memory) < self.batch_size:
            return False
        else:
            return True

    def random_batch_size_memory_td3(self):
        transitions = self.td3_memory.sample(self.batch_size)
        return transitions

    # -------------------------------dqn经验池-------------------------------------
    # 目前, 在 dqnAgent.py中会调用随机经验池 random_batch_size_memory_dqn
    # 在train.py中. 每控制一次英雄操作, 就存储一次经验
    #ReplayMemory 就是一个循环缓存区变量，帮你把数据按顺序存进来，
    # 满了自动从头覆盖，随时随机抽几条旧数据出来再用，省得程序自己管内存、管指针。
    # 就是临时存储训练数据只用
    # 就是一条自动滚动的“大变量”——来了新数据就往后追加，地方满了就回头覆盖，
    # 咱们只管往里放、往外拿，根本不用操心内存和指针。
    # 在train.py中会判断,是否存满了is_memory_bigger_batch_size_dqn


    def store_transition_dqn(self, *args):
        self.dqn_memory.push(*args)

    def is_memory_bigger_batch_size_dqn(self):
        if len(self.dqn_memory) < self.batch_size:
            return False
        else:
            return True

    def random_batch_size_memory_dqn(self):
        transitions = self.dqn_memory.sample(self.batch_size)
        return transitions

    # -------------------------------训练状态-------------------------------------
    def update_data_file(self, new_data):
        lock = FileLock("training_data.json.lock")

        with lock:
            if os.path.exists('training_data.json'):
                with open('training_data.json', 'r') as file:
                    data = json.load(file)
            else:
                data = []

            for new_plot in new_data:
                title_found = False
                for existing_plot in data:
                    if existing_plot['title'] == new_plot['title']:
                        existing_plot['x_data'].extend(new_plot['x_data'])
                        existing_plot['y_data'].extend(new_plot['y_data'])
                        title_found = True
                        break

                if not title_found:
                    data.append(new_plot)

            with open('training_data.json', 'w') as file:
                json.dump(data, file, indent=4)

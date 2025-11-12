# 王者荣耀 AI 技术分析与强化学习入门报告

## 报告概述

本报告旨在为有一定编程和监督学习基础的开发者，提供一份关于王者荣耀 AI 项目的实践性技术解读。报告将拆解项目代码，解释其工作原理，并以此为基础，引导读者入门强化学习（DQN）的核心概念。内容力求务实，从实际代码出发，辅以开发经验和注意事项，帮助新手快速建立从理论到实践的桥梁。

## 目录

*   **第一章：代码结构与执行流程**
    *   [1.1 各个 Python 文件的作用](./note/1-code-structure.md#11-各个-python-文件的作用)
    *   [1.2 程序如何从启动到训练：一步步解析](./note/1-code-structure.md#12-程序如何从启动到训练一步步解析)
    *   [1.3 `argparses.py` 中的超参数：它们是什么，如何影响训练](./note/1-code-structure.md#13-argparsespy-中的超参数它们是什么如何影响训练)

*   **第二章：项目中的实用 Python 技巧**
    *   [2.1 `memory.py`：如何用一个类(`ReplayMemory`)来管理训练数据](./note/2-python-techniques.md#21-memorypy如何用一个类replaymemory来管理训练数据)
    *   [2.2 `@singleton` 装饰器：为什么 `GlobalInfo` 在整个程序中只有一个实例？](./note/2-python-techniques.md#22-singleton-装饰器为什么-globalinfo-在整个程序中只有一个实例)
    *   [2.3 图像处理：`cv2` 和 `PyQt` 在项目中的应用](./note/2-python-techniques.md#23-图像处理cv2-和-pyqt-在项目中的应用)
    *   [2.4 并发编程：`threading` 和 `concurrent.futures` 如何让程序更高效](./note/2-python-techniques.md#24-并发编程threading-和-concurrentfutures-如何让程序更高效)

*   **第三章：强化学习入门：从监督学习到 DQN**
    *   [3.1 强化学习的目标：与监督学习的损失函数作类比](./note/3-rl-for-beginners.md#31-强化学习的目标与监督学习的损失函数作类比)
    *   [3.2 `dqnAgent.py` 剖析：智能体 (Agent) 是如何做出决策的？](./note/3-rl-for-beginners.md#32-dqnagentpy-剖析智能体-agent-是如何做出决策的)
    *   [3.3 Q值、奖励和损失：模型到底在学习什么？](./note/3-rl-for-beginners.md#33-q值奖励和损失模型到底在学习什么)
    *   [3.4 `epsilon` 参数：为什么有时要“随机乱玩”而不是永远“听模型的”？](./note/3-rl-for-beginners.md#34-epsilon-参数为什么有时要随机乱玩而不是永远听模型的)
    *   [3.5 新手常见问题：用“随机”数据来训练一个“最优”模型，这合理吗？](./note/3-rl-for-beginners.md#35-新手常见问题用随机数据来训练一个最优模型这合理吗)
    *   [3.6 新手开发经验与避坑指南](./note/3-rl-for-beginners.md#36-新手开发经验与避坑指南)

*   **第四章：下一步：二次开发与新项目规划**
    *   [4.1 整合现有思路：从 `myideal.txt` 和 `README.md` 出发](./note/4-next-steps.md#41-整合现有思路从-myidealtxt-和-readmemd-出发)
    *   [4.2 新项目技术选型：为什么用 `autowzry`？模型如何从简到繁？](./note/4-next-steps.md#42-新项目技术选型为什么用-autowzry模型如何从简到繁)
    *   [4.3 推荐的项目代码结构（附说明）](./note/4-next-steps.md#43-推荐的项目代码结构附说明)

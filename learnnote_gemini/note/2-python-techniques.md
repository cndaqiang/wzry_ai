# 第二章：项目中的实用 Python 技巧

本章将挑选出项目中几个对于 Python 新手来说可能比较陌生但非常实用的技术点。我们将通过代码实例，解释它们是什么，以及为什么在这里使用它们。

## 2.1 `memory.py`：如何用一个类(`ReplayMemory`)来管理训练数据

在监督学习中，我们通常有一个固定的数据集（比如一堆图片和它们的标签）。但在强化学习中，数据是动态产生的。`ReplayMemory` 这个类就是为了管理这些动态数据而设计的。

**它到底是什么？**

你可以把它想象成一个“**固定长度的先进先出列表**”。

-   **固定长度**：它有一个最大容量 `capacity`。当数据存满了，再存新数据时，最老的数据就会被挤出去。
-   **先进先出**：这保证了经验池里保存的总是最近的、最有价值的经验。

让我们看看代码：

```python
# memory.py

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
```

这里的 `self.position = (self.position + 1) % self.capacity` 是一个非常巧妙的技巧。`%` 是取余运算符，它让 `position` 索引在 `0` 到 `capacity-1` 之间循环。当 `position` 到达末尾后，会自动回到 `0`，从而覆盖掉最老的数据。

**和监督学习的类比**

`ReplayMemory` 的作用，非常类似于你在进行监督学习时的数据预处理步骤：

1.  **存储数据 (`push`)**: 相当于你不断收集和扩充你的数据集。
2.  **随机抽样 (`sample`)**:
    ```python
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    ```
    这一步和监督学习中 `DataLoader` 的 `shuffle=True` 异曲同工。我们不希望模型按顺序学习，因为连续的经验关联性太强。通过随机抽样，可以打破数据之间的相关性，让模型训练更稳定，学到的知识更具泛化能力。

**总结**：`ReplayMemory` 是一个专门为在线学习设计的、带自动淘汰和随机抽样功能的数据集容器。

## 2.2 `@singleton` 装饰器：为什么 `GlobalInfo` 在整个程序中只有一个实例？

在 `globalInfo.py` 中，你会看到一个 `@singleton` 装饰器被用在了 `GlobalInfo` 类上。

```python
# globalInfo.py

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class GlobalInfo:
    def __init__(self, ...):
        self.dqn_memory = ReplayMemory(buffer_capacity)
        # ...
```

**它做了什么？**

这个装饰器改变了类的行为。正常情况下，你每次调用 `GlobalInfo()` 都会创建一个新的、独立的对象。但被 `@singleton` 修饰后，**无论你在程序的任何地方、任何时间调用 `GlobalInfo()`，你得到的永远是同一个对象实例**。

**为什么需要它？**

回想一下第一章，我们有两个并行的线程：`data_collector` 和 `train_agent`。

-   `data_collector` 需要把采集到的经验存入经验池。
-   `train_agent` 需要从同一个经验池中取出经验进行训练。

如果它们各自创建自己的 `GlobalInfo` 实例，那么就会有两个独立的经验池，采集的数据永远无法被训练器使用。

通过单例模式，我们保证了整个程序中只有一个 `GlobalInfo` 实例，因此也只有一个 `dqn_memory` 经验池。这样，`data_collector` 存入的数据，`train_agent` 才能看得到、用得上。它就像一个所有线程都能访问的“**中央数据总站**”。

## 2.3 图像处理：`cv2` 和 `PyQt` 在项目中的应用

AI 的“眼睛”看到的是像素，但它需要理解这些像素的含义。`cv2` (OpenCV) 在这里扮演了“视觉神经”的角色。

**`cv2`：从像素到信息**

在 `getReword.py` 的 `calculate_attack_reword` 函数中，有一个典型的应用：**判断敌人血条还剩多少**。

```python
# getReword.py

# ... 截取屏幕中敌方血条的区域 ...
cropped_img = img[top:bottom, left:right]

# ... 将图片从 BGR 转换到 HSV 色彩空间 ...
hsv_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

# ... 定义红色的 HSV 范围 ...
lower_bound = np.array([hue - tolerance, 50, 50])
upper_bound = np.array([hue + tolerance, 255, 255])

# ... 制作一个只保留红色像素的“面具” ...
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# ... 计算面具中有多少非零像素，即红色区域的面积 ...
area = cv2.countNonZero(mask)
```

这个过程将一个视觉问题（“我打掉敌人多少血？”）转换成了一个数学问题（“血条图片中的红色像素有多少？”），从而为计算奖励 `reword` 提供了量化依据。这是在 CV 辅助的强化学习中非常常见的技巧。

**`PyQt`：开发辅助工具**

`showposition.py` 是一个用 `PyQt` 编写的桌面小工具。它的作用很简单：加载一张游戏截图，然后你用鼠标点击图片上的任何位置，它就能告诉你这个点的**绝对坐标**和**相对坐标**（百分比）。

这在你需要为 `argparses.py` 中的 `info_actions_detail` 和 `attack_actions_detail` 配置新的按键位置时非常有用，避免了手动猜测坐标的繁琐工作。这体现了“工欲善其事，必先利其器”的思想，为自己编写合适的辅助工具能极大提升开发效率。

## 2.4 并发编程：`threading` 和 `concurrent.futures` 如何让程序更高效

这个项目巧妙地运用了多线程来提升效率。

**`threading`：宏观任务并行**

在 `train.py` 中，数据采集和模型训练被放在了两个不同的线程中。这是一种经典的“**生产者-消费者**”模式：

-   **生产者**：`data_collector` 线程，负责玩游戏、产生数据，并放入“仓库”（`ReplayMemory`）。
-   **消费者**：`train_agent` 线程，负责从“仓库”中取数据，并用来训练模型。

**为什么这样做？**
想象一下，如果没有多线程，程序只能是：玩一步 -> 训练一步 -> 玩一步 -> 训练一步...
如果 `replay()` 训练需要 1 秒，那么游戏操作的反应就会延迟 1 秒，这在实时游戏中是不可接受的。

通过 `threading`，采集和训练可以“同时”进行。虽然由于 Python 的 GIL（全局解释器锁），它们并非真正在同一时刻运行，但在 I/O 操作（如 `time.sleep`、等待截图）和计算密集型操作（模型训练）之间，操作系统可以高效地切换，使得在宏观上，两个任务都在推进，互不阻塞。

**`concurrent.futures`：微观任务并行**

在 `android_tool.py` 和 `getReword.py` 中，你看到了 `ThreadPoolExecutor`。这是一个更高级、更易用的多线程工具。

例如，在 `getReword.py` 的 `get_reword` 函数中：

```python
# getReword.py

with ThreadPoolExecutor() as executor:
    # 同时提交三个任务到线程池
    future_class_name = executor.submit(self.check_finish, image) # 识别胜利/失败
    future_check_death = executor.submit(self.check_death, image) # 识别死亡
    future_md_class_name = executor.submit(self.predict, image)   # 识别血条

    # ... 等待并处理结果 ...
```

这里，三个独立的图像识别任务被同时“扔”进了线程池。线程池会自动分配线程去执行它们。这比手动创建和管理三个 `threading.Thread` 对象要简洁得多。对于需要并行执行多个相似的、独立的短任务场景，`ThreadPoolExecutor` 是一个绝佳的选择。

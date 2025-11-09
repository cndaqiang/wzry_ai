## README
* 我的初次配置过程: [Q_README.md](Q_README.md)
* 最新的运行环境配置: [Q_INSTALL.md](Q_INSTALL.md),基于便携版Python3.10
* 原项目[README_myBoris.md](README_myBoris.md)
* 二次开发进度[Q_Plan.md](Q_PLAN.md)


## 代码解读与开发规划

### 项目的运行流程
* 检测状态,直到进入对战状态
* 截屏 -> 随机/基于模型生成操作参数 -> 执行操作 -> 截屏并计算奖励 -> 重复以上步骤直到对战结束
* 检测状态,检测到胜利或者失败文字结束训练`from ppocronnx import TextSystem`

### 对局开始结束识别
`models/death.onnx`和`start.onnx`是用于检查页面是对战还是死亡的模型文件。
* 并不是强化学习的对战模型
* 后期可以考虑使用更简单的opencv图像识别来处理,并不需要机器学习来进行分辨.如[autowzry](https://github.com/cndaqiang/autowzry)中实现的那样.
* 目前这两个模型的识别还挺准确的,可以先用着.
* 即使没有这个模型,也可以直接修改代码,就让程序人为在对战了
* 或者在autowzry进入对战后,调用这个训练,在autowzry判断结束后,停止这个训练,逐渐进行
* 或者`import autowzry`,调用`autowzry`的对战状态判断函数
* 不依赖`onnx`可以省很对配置环境,也可以省去模型文件

### 获取状态/截图
目前获取对战状态,是windows系统的窗口截图实现
* 不用必须使用scrcpy的窗口,直接对MuMu窗口进行识别即可,分辨率也不用固定
* `parser.add_argument('--window_title', type=str, default='MuMu安卓设备', help="window_title")`
 计划改为[airtest_mobileauto](https://github.com/cndaqiang/airtest_mobileauto)返回screen,这样就不用在前台识别了,更好的融入自动训练流程

### 训练动作与执行
设备的操控是基于adb实现
* 目前支持的控制/action: `move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3`
* 其实重要的操作只有移动一个, 攻击和释放技能可以无脑点击即可
* **计划后面只训练移动方向**,是否应该前进,遇到敌人/地方防御塔,应该前进还是后退
* **只训练移动方向对于大多数的英雄也是适配的, 初期不可能适配技能和信号**
* 只用学习四个信号[下路、中路、上路、反向中路]四个方向即可
* 目前[autowzry](https://github.com/cndaqiang/autowzry)项目缺乏的也就是这个功能.

### 另一种训练方案: 监督学习
* 训练移动方向或许很难, 但是可以跳出架构, 但是可以观察TX的挂机操作,判断画面是前进了还是后退了
* 学习当前图像下, 前进和后退的概率是多少
* 没有强化,只有监督学习,应该是一个更简单的模型.
* 更适合作为库的存在,autowzry调用这个库,进行前进后退的操作.
* 这个方案或许更适合新手入门,也更容易实现.

## 测试阶段
* 测试学习代码期间, 进入训练营即可
* * 如在训练营中校验自己的位置按钮是否都正确
* 很多代码文件在运行过程中可以实时修改,比如android_tool.py,修改作为位置,会影响实时执行

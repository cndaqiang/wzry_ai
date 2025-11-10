1. [Done]在控制角色移动函数部分增加输出,矫正按键位置,测试adb位置
2. [Done][前方移动权重增加]增加移动权重, 让角色更倾向于上线,删除其他移动权重
3. [done][需要训练,提供的模型是图像识别,不是对抗模型]确定这个模型真的有让角色移动和对战的能力吗？是否需要训练新的模型.
4. [done]改用airtest提供接口,而非windows窗口截图
5. 改用airtest进行touch,采用autowzry进行控制
6. [done][支持960x540直接从MuMu模拟器的窗口读取,相对坐标计算的位置]调整分辨率
7. 打包成库,在autowzry中调用此库/训练此库
8. [Done]采用便携版Python3.10重新配置环境,并编写一键安装脚本[Q_INSTALL.md](Q_INSTALL.md)


## Airtest适配计划
* [done]接入[autowzry](https://github.com/cndaqiang/autowzry), 判断对战状态, 替换掉onnx模型
* [done]使用[airtest_mobileauto](https://github.com/cndaqiang/airtest_mobileauto)进行截图操作,替换掉windows窗口截图
* 在[autowzry](https://github.com/cndaqiang/autowzry)中开发死亡检查，以及基于opencv的胜利和失败判断,彻底替换掉onnx
* [done,只有android_tool.py中涉及,已替换]使用[airtest_mobileauto](https://github.com/cndaqiang/airtest_mobileauto)进行touch操作,替换掉adb命令操作
* 要先升级opencv和numpy吗？


### 截图操作拆解
#### pyqt截图窗口内容

原作者，这里采用了PyQt 来截图，兼容透明层、DPI 缩放、多屏环境，如果用 win32ui 自己截图，这些细节要自己处理，否则截图可能模糊或偏移。

但是这种方式会包含标题,如
![](pyqt_screenshot.png)

```
import cv2
import numpy as np
import win32gui
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
    def screenshot_window(self):
        """
        截取指定窗口的内容并返回图像数据。

        参数:
        window_name (str): 窗口标题的部分或全部字符串。

        返回:
        np.ndarray: 截图的图像数据，如果窗口未找到则返回 None。
        """
        try:
            # 获取窗口句柄
            handle = win32gui.FindWindow(None, args.window_title)
            if handle == 0:
                raise Exception(f"窗口 '{args.window_title}' 未找到。")

            # 初始化 QApplication
            app = QApplication(sys.argv)
            screen = QApplication.primaryScreen()

            # 截取指定窗口的内容
            img = screen.grabWindow(handle).toImage()

            # 将 QImage 转换为 numpy 数组
            img = img.convertToFormat(QImage.Format.Format_RGB32)
            width = img.width()
            height = img.height()
            ptr = img.bits()
            ptr.setsize(height * width * 4)
            arr = np.array(ptr).reshape(height, width, 4)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

            return arr
        except Exception as e:
            print(e)
            return None
```

#### airtest截图窗口内容
airtest的截图返回就是opencv数组,无需处理直接截图
![](air_screenshot.png)

```
from airtest_mobileauto.control import deviceOB, Settings
class AndroidTool:
    def __init__(self, scrcpy_dir="scrcpy-win64-v2.0",airtest_config=""):
        self.airtest_config = airtest_config
        self.airtest = False
        if len(self.airtest_config) > 0:
            print(f"---> AndroidTool: airtest_config={airtest_config}")
            print(f"---> cndaqiang: 将基于airtest执行")
            self.airtest_init()
        self.scrcpy_dir = scrcpy_dir
        self.device_serial = args.iphone_id
        self.actual_height, self.actual_width = self.get_device_resolution()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        #self._show_action_log = False
        self._show_action_log = True #cndaqang debug

    def airtest_init(self):
        Settings.Config(self.airtest_config)
        # copy from autowzry
        # 静态资源
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, 'assets')
        Settings.figdirs.append(assets_dir)
        seen = set()
        Settings.figdirs = [x for x in Settings.figdirs if not (x in seen or seen.add(x))]
        #
        # device
        self.mynode = Settings.mynode
        self.totalnode = Settings.totalnode
        self.totalnode_bak = self.totalnode
        self.LINK = Settings.LINK_dict[Settings.mynode]
        self.移动端 = deviceOB(mynode=self.mynode, totalnode=self.totalnode, LINK=self.LINK)
        self.移动端.连接设备()
        self.airtest = True
    #...
    def screenshot_airtest(self):
        """
        使用 Airtest 截取屏幕并返回图像数据。

        返回:
        np.ndarray: 截图的图像数据。
        """
        try:
            arr = self.移动端.device.snapshot()
            # cndaqiang debug
            cv2.imwrite("screenshot_airtest.png", arr)
            #
            return arr
        except Exception as e:
            print(e)
            return None
```
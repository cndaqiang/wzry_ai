
## 构建绿色便携环境

### 便携Python
```
# https://cyfuture.dl.sourceforge.net/project/winpython/WinPython_3.10/3.10.9.0/Winpython64-3.10.9.0.exe?viasf=1
# 激活
$baseDir = 'D:\GreenSoft\WPy64-31090\python-3.10.9.amd64'
$env:PATH = "$baseDir;$baseDir\Scripts;" + $env:PATH
```

### 依赖

**重要说明：** 由于 ppocr-onnx 的依赖会自动安装最新版 opencv-python（需要 NumPy 2.x），而项目使用的 opencv-contrib-python 4.9.0.80 只兼容 NumPy 1.x，因此需要按以下步骤安装以避免版本冲突：

```bash
python -m pip cache purge

# 1. 先安装 opencv-contrib-python（会自动安装 NumPy，可能是 2.x）
python -m pip install opencv-contrib-python==4.9.0.80

# 2. 立即降级 NumPy 到 1.x（重要！否则 cv2 无法正常工作）
python -m pip install "numpy<2"

# 3. 安装其他依赖
python -m pip install PyQt5==5.15.10 pywin32==306 keyboard==0.13.5 matplotlib==3.9.2 filelock==3.13.1

# 4. 安装 PyTorch (CUDA 11.8)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# 只有onnxruntime-gpu才需要cuda和cudnn, 若采用cpu版本,则无需安装cuda和cudnn
# 5. 使用 --no-deps 安装 ppocr-onnx 避免自动安装冲突的 opencv-python
python -m pip install ppocr-onnx==0.0.3.9 --no-deps

# 5.1 安装 ppocr-onnx 的其他依赖（不包括 opencv-python）
python -m pip install shapely pyclipper pillow requests

# 5.2 安装 onnxruntime
# python -m pip install onnxruntime
# onnxruntime 是CPU实现, 若选择安装GPU版本, 这里就需要onnxruntime
# onnxruntime-gpu 需要安装 CUDA Toolkit cuDNN
# https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_527.41_windows.exe
# https://developer.download.nvidia.com/compute/cudnn/9.15.0/local_installers/cudnn_9.15.0_windows.exe
# 只有onnxruntime-gpu才需要cuda和cudnn, 若采用cpu版本,则无需安装cuda和cudnn. 仅用于判断页面状态, 可以不安装
# python -m pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
# 直接安装导入库dll异常
# 根据 https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
# CUDA Toolkit 12.x 和 cuDNN 9.x 需要 ONNX 1.18.1 ~ 1.20.x
# 这样安装没有任何问题可以直接使用
python -m pip install onnxruntime-gpu==1.19.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```


#### 验证安装
```bash
# 验证核心库可以正常导入并使用
python -c "import cv2; import numpy as np; print('NumPy:', np.__version__); print('Has cvtColor:', hasattr(cv2, 'cvtColor')); print('All imports successful!')"

python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

python -c "import onnxruntime as ort; print('ONNX Runtime version:', ort.__version__); print('Available providers:', ort.get_available_providers())"
```


## 运行


```
$baseDir = 'D:\GreenSoft\WPy64-31090\python-3.10.9.amd64'
$env:PATH = "$baseDir;$baseDir\Scripts;" + $env:PATH

.\scrcpy-win64-v2.0\adb.exe connect 127.0.0.1:5555
python train.py



# 暂停，方便查看结果
$timer=Start-Job {Start-Sleep 20}; Write-Host "Press any key to continue..."; while(-not [console]::KeyAvailable -and (Get-Job -Id $timer.Id).State -eq 'Running'){Start-Sleep 0.1}; Stop-Job $timer
```
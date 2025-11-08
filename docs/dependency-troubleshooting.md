# 依赖问题排查与解决记录

**日期:** 2025-11-08
**协作:** cndaqiang + Claude (via [AnyRouter](https://anyrouter.top/register?aff=TFIa))

## 问题概述

在配置 wzry_ai 项目环境时遇到多个依赖冲突问题，本文档记录了完整的排查和解决过程。

## 遇到的问题

### 1. NumPy 版本冲突

**错误信息:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
AttributeError: _ARRAY_API not found
module 'cv2' has no attribute 'cvtColor'
```

**问题根源:**
- ppocr-onnx 0.0.3.9 依赖声明中包含 `opencv-python`（无版本限制）
- pip 自动安装最新版 opencv-python 4.12.x，要求 NumPy 2.x
- opencv-contrib-python 4.9.0.80 虽能安装 NumPy 2.x，但运行时 cv2 模块无法正常加载
- 必须使用 NumPy 1.x 才能正常工作

**解决方案:**
1. 先安装 opencv-contrib-python==4.9.0.80（允许它安装 NumPy 2.x）
2. 立即降级 NumPy 到 1.x：`pip install "numpy<2"`
3. 使用 `--no-deps` 安装 ppocr-onnx，避免自动安装 opencv-python
4. 手动安装 ppocr-onnx 的其他依赖

### 2. ONNX Runtime GPU 警告

**错误信息:**
```
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*
Error loading "onnxruntime_providers_cuda.dll" which depends on "cublasLt64_12.dll"
```

**问题原因:**
- PyTorch 自带 CUDA 库，但 onnxruntime-gpu 需要系统安装 CUDA Toolkit 12.x
- 缺少系统级 CUDA 12 和 cuDNN 9 支持

**性能影响:**
- CPU 推理：约 20-50 ms/帧
- GPU 推理：约 2-5 ms/帧
- **提升 5-10 倍**

**解决方案:**
安装 CUDA Toolkit 12.0 和 cuDNN 9.15：
- CUDA: https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_527.41_windows.exe
- cuDNN: https://developer.download.nvidia.com/compute/cudnn/9.15.0/local_installers/cudnn_9.15.0_windows.exe

## 为什么会出现依赖冲突？

**时间差导致的环境不一致:**
- 原开发者安装时（可能 2024 年初）：opencv-python 最新版可能是 4.9.x（兼容 NumPy 1.x）
- 现在安装（2025 年）：opencv-python 最新版是 4.12.x（要求 NumPy 2.x）
- ppocr-onnx 的依赖声明中 `opencv-python` 没有版本限制，导致不同时间安装结果不同

**这是典型的依赖地狱（Dependency Hell）问题**

## 最终解决方案

### 完整安装步骤

```bash
# 1. 创建 Python 3.10 环境
conda create --name wzry_ai python=3.10
conda activate wzry_ai

# 2. 安装 opencv-contrib-python（会自动安装 NumPy 2.x）
python -m pip install opencv-contrib-python==4.9.0.80

# 3. 立即降级 NumPy 到 1.x（重要！）
python -m pip install "numpy<2"

# 4. 使用 --no-deps 安装 ppocr-onnx
python -m pip install ppocr-onnx==0.0.3.9 --no-deps

# 5. 安装其他依赖
python -m pip install PyQt5==5.15.10 pywin32==306 keyboard==0.13.5 matplotlib==3.9.2 filelock==3.13.1

# 6. 安装 ppocr-onnx 的其他依赖
python -m pip install onnxruntime shapely pyclipper pillow requests

# 7. 安装 PyTorch (CUDA 11.8)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 8. 安装 onnxruntime-gpu (CUDA 12)
python -m pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 9. 安装 CUDA 12.0 和 cuDNN 9.15（系统级）
# 下载并安装上述链接中的安装包
```

### 验证安装

```bash
# 验证核心库
python -c "import cv2; import numpy as np; print('NumPy:', np.__version__); print('Has cvtColor:', hasattr(cv2, 'cvtColor'))"

# 验证 GPU 支持
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 验证 ONNX Runtime
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"
```

## 关键经验教训

1. **依赖锁定的重要性:**
   - 只锁定直接依赖不够，传递依赖也会变化
   - 应使用 `pip freeze` 或 `poetry.lock` 锁定所有依赖

2. **PyTorch vs ONNX Runtime 的 CUDA 差异:**
   - PyTorch 自带 CUDA 库（独立运行）
   - ONNX Runtime 需要系统安装 CUDA Toolkit
   - 两者可以共存，使用不同版本的 CUDA

3. **实时推理项目需要 GPU:**
   - 本项目 ONNX 模型每帧都调用
   - GPU 加速是必需的，不是可选的

4. **opencv-python vs opencv-contrib-python:**
   - 两者功能相似（contrib 包含更多模块）
   - 不能同时安装，会冲突
   - ppocr-onnx 可以在 opencv-contrib-python 上工作

## 最终环境状态

```
Python: 3.10
NumPy: 1.26.4
opencv-contrib-python: 4.9.0.80
ppocr-onnx: 0.0.3.9
PyTorch: 2.7.1+cu118
onnxruntime-gpu: 1.23.2
CUDA: 12.0
cuDNN: 9.15
GPU: NVIDIA GeForce RTX 4060 Ti (16GB)
Driver: 581.57
```

## 参考资料

- [ONNX Runtime CUDA Requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)

## 致谢

本次问题排查和解决得益于：
- [Claude](https://claude.ai) (Anthropic) 提供的技术支持
- [AnyRouter](https://anyrouter.top/register?aff=TFIa) 提供的免费 Claude API 服务

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Honor of Kings (王者荣耀) AI training project** that implements a Deep Q-Network (DQN) reinforcement learning system to play the mobile game. The AI learns to control game characters through screen analysis and action execution using Android automation.

## Key Architecture Components

### Core Training System
- **DQN Agent** (`dqnAgent.py`): Main reinforcement learning agent implementing Deep Q-Network algorithm
- **Neural Network** (`net_actor.py`): NetDQN model - CNN architecture that processes 640x640 game screenshots
- **Environment** (`wzry_env.py`): Game environment wrapper that handles action space (movement, attacks, skills)

### Game Interface & Control
- **Android Tool** (`android_tool.py`): Handles device connection, screenshot capture, and action execution via ADB
- **Airtest Integration**: Uses `airtest_mobileauto` for mobile device control and `autowzry` for game state detection
- **Action Space**: 8-dimensional action vector [move, angle, info, attack, action_type, arg1, arg2, arg3]

### State Recognition
- **ONNX Models** (`models/`): Pre-trained models for game state detection (`start.onnx`, `death.onnx`)
- **Reward System** (`getReword.py`): Calculates rewards based on game outcomes and state changes
- **Screenshot Processing**: Windows-based window capture targeting MuMu emulator

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create --name wzry_ai python=3.10
conda activate wzry_ai

# Install dependencies (follow Q_README.md for precise order)
python -m pip install opencv-contrib-python==4.9.0.80
python -m pip install "numpy<2"
python -m pip install ppocr-onnx==0.0.3.9 --no-deps
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Running the Training
```bash
# Connect to emulator via ADB
./scrcpy-win64-v2.0/adb.exe connect 127.0.0.1:5555

# Start training
python train.py

# Or use the PowerShell script
./run.ps1
```

### Key Parameters
- `--window_title`: Emulator window title (default: 'MuMu安卓设备')
- `--iphone_id`: ADB device ID (default: '127.0.0.1:5555')
- `--model_path`: Model save/load path (default: "src/wzry_ai.pt")
- `--batch_size`: Training batch size (default: 64)
- `--learning_rate`: Learning rate (default: 0.001)

## Important Implementation Details

### Action Execution
- Movement: Relative coordinates based on 16:9 screen ratio
- Skills/Attacks: Predefined click positions for different game elements
- Multi-threading: Separate threads for data collection and training

### Game State Detection
- Uses `autowzry.判断对战中()` for battle state detection
- Fallback to ONNX models for start/end game recognition
- Reward calculation based on game outcomes (victory/defeat)

### Known Issues & Limitations
- Model corruption: Delete `src/wzry_ai.pt` if loading fails
- Dependency conflicts: NumPy 1.x required for OpenCV compatibility
- GPU acceleration: Requires CUDA 12.x + cuDNN 9.x for optimal performance

## Development Notes

- The project is transitioning from pure ADB control to Airtest-based automation
- Current focus is on movement training rather than complex skill combinations
- Real-time modification of `android_tool.py` affects live execution
- Training can be tested in practice mode before real matches
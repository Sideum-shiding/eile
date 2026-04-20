# MSI App Player Subway Surfers Avoidance Bot

Native C++ and Python scripts that watch Subway Surfers in the `MSI App Player` window, detect risky objects with OpenCV, and use lane changes, jumps, and rolls to avoid obstacles.

## Features

- Native C++ screen capture with WinAPI `BitBlt`
- Python fallback with `mss` and `pygetwindow`
- Gameplay ROI from 45% to 92% of the window height
- HSV coin mask, saturated-object mask, dark-object mask, and Canny edge mask
- Three-lane danger scoring with `cv2.countNonZero()`
- `left` / `right` lane changes toward the safest lane
- `up` jump when danger is concentrated low in the current lane
- `down` roll when danger is concentrated higher in the current lane
- Separate cooldowns for horizontal and vertical actions
- OpenCV mask preview with lane borders and action text

## Python Install

```powershell
pip install -r requirements.txt
```

## Python Run

```powershell
python main.py --activate
```

The script opens an OpenCV preview window. Press `q` inside that window to quit.

```powershell
python main.py --activate --start-lane center
```

Check whether the emulator receives all controls:

```powershell
python main.py --activate --test-press
```

Print lane danger counters:

```powershell
python main.py --activate --print-stats
```

Maximum FPS mode:

```powershell
python main.py --activate --no-debug
```

Tune sensitivity:

```powershell
python main.py --activate --warning-threshold 1200 --lane-danger-threshold 4000
```

If the emulator misses short taps, increase key hold time:

```powershell
python main.py --activate --key-hold 0.08
```

Move the mouse to a screen corner to trigger PyAutoGUI fail-safe.

## C++ Build

The C++ version needs a C++ OpenCV SDK and a CMake-compatible build tool, not only the Python `opencv-python` wheel.
Install OpenCV for C++ and point CMake at its `OpenCVConfig.cmake`.

```powershell
cmake -S . -B build -DOpenCV_DIR="C:\path\to\opencv\build"
cmake --build build --config Release
```

## C++ Run

```powershell
.\build\Release\subway_bot_cpp.exe --activate
```

If you build with MinGW or a single-config generator, the executable may be here instead:

```powershell
.\build\subway_bot_cpp.exe --activate
```

Maximum FPS mode:

```powershell
.\build\Release\subway_bot_cpp.exe --activate --no-debug
```

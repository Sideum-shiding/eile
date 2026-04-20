# MSI App Player Subway Surfers Avoidance Bot

Python script that watches Subway Surfers in the `MSI App Player` window, detects risky objects with OpenCV, and uses lane changes, jumps, and rolls to avoid obstacles.

## Features

- Fast screen capture with `mss`
- Window lookup with `pygetwindow`
- Gameplay ROI from 45% to 92% of the window height
- HSV coin mask, saturated-object mask, dark-object mask, and Canny edge mask
- Three-lane danger scoring with `cv2.countNonZero()`
- `left` / `right` lane changes toward the safest lane
- `up` jump when danger is concentrated low in the current lane
- `down` roll when danger is concentrated higher in the current lane
- Separate cooldowns for horizontal and vertical actions
- OpenCV mask preview with lane borders and action text

## Install

```powershell
pip install -r requirements.txt
```

## Run

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

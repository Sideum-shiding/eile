# MSI App Player No Coin Tester

Python script that watches Subway Surfers in the `MSI App Player` window, detects gold coins with OpenCV, and moves the player away from coin-heavy lanes.

## Features

- Fast screen capture with `mss`
- Window lookup with `pygetwindow`
- Decision ROI from 60% to 85% of the window height
- HSV gold mask with `[20, 100, 100]` to `[30, 255, 255]`
- Three-sector `cv2.countNonZero()` scoring
- Lane change only when the current lane contains yellow pixels
- 200ms cooldown between key presses
- OpenCV mask preview with sector borders

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
python main.py --activate --start-sector center
```

Tune noise sensitivity:

```powershell
python main.py --activate --coin-threshold 20
```

Check whether the emulator receives keys and whether the mask sees coins:

```powershell
python main.py --activate --test-press --print-stats
```

If the emulator misses short taps, increase key hold time:

```powershell
python main.py --activate --key-hold 0.08
```

Move the mouse to a screen corner to trigger PyAutoGUI fail-safe.

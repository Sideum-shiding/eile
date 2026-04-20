# MSI App Player No Coin Tester

Python script that watches the `MSI App Player` window, detects gold-colored UI objects with OpenCV, and keeps the player in the safest vertical sector.

## Features

- Fast screen capture with `mss`
- Window lookup with `pygetwindow`
- Lower-half ROI scanning
- HSV gold mask with `[20, 100, 100]` to `[30, 255, 255]`
- Three-sector `cv2.countNonZero()` scoring
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

Move the mouse to a screen corner to trigger PyAutoGUI fail-safe.

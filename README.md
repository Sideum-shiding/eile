# MSI App Player Gold Detector Bot

Python bot that watches the `MSI App Player` window, detects gold objects with OpenCV, and sends keyboard input through PyAutoGUI.

## Features

- Fast screen capture with `mss`
- ROI scanning from 50% to 80% of the window height
- OpenCV color detection for `RGB(255, 215, 0)`
- Cooldown after each key press to avoid input spam
- Optional debug preview

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python main.py --activate
```

With debug preview:

```powershell
python main.py --activate --debug
```

Move the mouse to a screen corner to trigger PyAutoGUI fail-safe.

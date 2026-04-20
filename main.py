import argparse
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mss
import numpy as np
import pyautogui


WINDOW_TITLE = "MSI App Player"
TARGET_RGB = (255, 215, 0)
ROI_TOP_RATIO = 0.50
ROI_BOTTOM_RATIO = 0.80


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    area: float
    sector: str


@dataclass(frozen=True)
class RoiRegion:
    left: int
    top: int
    width: int
    height: int
    offset_y: int


@dataclass(frozen=True)
class ColorThreshold:
    lower: np.ndarray
    upper: np.ndarray
    kernel: np.ndarray


def find_window(title: str):
    windows = pyautogui.getWindowsWithTitle(title)
    visible_windows = [window for window in windows if window.width > 0 and window.height > 0]

    if not visible_windows:
        raise RuntimeError(f'Window with title "{title}" was not found.')

    return visible_windows[0]


def get_window_region(window) -> tuple[int, int, int, int]:
    left = max(0, int(window.left))
    top = max(0, int(window.top))
    width = max(1, int(window.width))
    height = max(1, int(window.height))
    return left, top, width, height


def get_roi_region(window) -> RoiRegion:
    left, top, width, height = get_window_region(window)
    roi_offset_y = int(height * ROI_TOP_RATIO)
    roi_bottom = int(height * ROI_BOTTOM_RATIO)
    roi_height = max(1, roi_bottom - roi_offset_y)
    return RoiRegion(
        left=left,
        top=top + roi_offset_y,
        width=width,
        height=roi_height,
        offset_y=roi_offset_y,
    )


def roi_to_mss_monitor(roi: RoiRegion) -> dict[str, int]:
    return {
        "left": roi.left,
        "top": roi.top,
        "width": roi.width,
        "height": roi.height,
    }


def grab_roi_frame(sct, roi: RoiRegion) -> np.ndarray:
    return np.asarray(sct.grab(roi_to_mss_monitor(roi)), dtype=np.uint8)


def build_gold_threshold(tolerance: int) -> ColorThreshold:
    target = np.uint8([[TARGET_RGB]])
    target_hsv = cv2.cvtColor(target, cv2.COLOR_RGB2HSV)[0][0]

    hue_tolerance = max(4, tolerance // 5)
    saturation_tolerance = tolerance
    value_tolerance = tolerance

    lower = np.array(
        [
            max(0, int(target_hsv[0]) - hue_tolerance),
            max(0, int(target_hsv[1]) - saturation_tolerance),
            max(0, int(target_hsv[2]) - value_tolerance),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            min(179, int(target_hsv[0]) + hue_tolerance),
            min(255, int(target_hsv[1]) + saturation_tolerance),
            min(255, int(target_hsv[2]) + value_tolerance),
        ],
        dtype=np.uint8,
    )

    kernel = np.ones((5, 5), dtype=np.uint8)
    return ColorThreshold(lower=lower, upper=upper, kernel=kernel)


def build_gold_mask(frame_bgra: np.ndarray, threshold: ColorThreshold) -> np.ndarray:
    frame_bgr = frame_bgra[:, :, :3]
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, threshold.lower, threshold.upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, threshold.kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, threshold.kernel)
    return mask


def classify_sector(x: int, frame_width: int) -> str:
    left_boundary = frame_width / 3
    right_boundary = frame_width * 2 / 3

    if x < left_boundary:
        return "left"
    if x < right_boundary:
        return "center"
    return "right"


def detect_gold_object(
    frame_bgra: np.ndarray,
    *,
    min_area: float,
    roi_offset_y: int,
    full_width: int,
    threshold: ColorThreshold,
) -> Optional[Detection]:
    mask = build_gold_mask(frame_bgra, threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"]) + roi_offset_y
    sector = classify_sector(x, full_width)
    return Detection(x=x, y=y, area=area, sector=sector)


def action_for_detection(detection: Detection) -> Optional[str]:
    if detection.sector == "center":
        return "left"
    if detection.sector == "left":
        return "right"
    return None


def draw_debug(
    frame_bgra: np.ndarray,
    detection: Optional[Detection],
    *,
    roi_offset_y: int,
) -> np.ndarray:
    debug = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    height, width = debug.shape[:2]

    cv2.line(debug, (width // 3, 0), (width // 3, height), (255, 255, 255), 1)
    cv2.line(debug, (width * 2 // 3, 0), (width * 2 // 3, height), (255, 255, 255), 1)

    if detection:
        roi_y = detection.y - roi_offset_y
        cv2.circle(debug, (detection.x, roi_y), 12, (0, 255, 255), 2)
        cv2.putText(
            debug,
            f"{detection.sector} area={int(detection.area)}",
            (max(0, detection.x - 80), max(25, roi_y - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return debug


def run_bot(args: argparse.Namespace) -> None:
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True

    window = find_window(args.title)
    if args.activate:
        window.activate()
        time.sleep(0.2)

    input_locked_until = 0.0
    threshold = build_gold_threshold(args.tolerance)
    print(
        f'Watching "{args.title}" ROI {ROI_TOP_RATIO:.0%}-{ROI_BOTTOM_RATIO:.0%}. '
        "Move the mouse to a screen corner to stop PyAutoGUI."
    )

    with mss.mss() as sct:
        while True:
            try:
                roi = get_roi_region(window)
                frame_bgra = grab_roi_frame(sct, roi)
                now = time.monotonic()

                detection = detect_gold_object(
                    frame_bgra,
                    min_area=args.min_area,
                    roi_offset_y=roi.offset_y,
                    full_width=roi.width,
                    threshold=threshold,
                )
                action = action_for_detection(detection) if detection else None

                if action and now >= input_locked_until:
                    pyautogui.press(action)
                    input_locked_until = now + args.cooldown
                    print(
                        f"Detected gold object in {detection.sector} sector "
                        f"at ({detection.x}, {detection.y}); pressed {action}."
                    )

                if args.debug:
                    cv2.imshow(
                        "MSI App Player detector ROI",
                        draw_debug(frame_bgra, detection, roi_offset_y=roi.offset_y),
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.interval > 0:
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                break
            except pyautogui.FailSafeException:
                print("PyAutoGUI fail-safe triggered.")
                break

    if args.debug:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect a gold RGB(255, 215, 0) object in the MSI App Player window "
            "and press left/right according to its sector."
        )
    )
    parser.add_argument("--title", default=WINDOW_TITLE, help="Target window title.")
    parser.add_argument("--min-area", type=float, default=80.0, help="Minimum contour area.")
    parser.add_argument("--tolerance", type=int, default=45, help="HSV color tolerance.")
    parser.add_argument("--cooldown", type=float, default=0.2, help="Seconds to ignore input after a key press.")
    parser.add_argument("--interval", type=float, default=0.0, help="Delay between frames.")
    parser.add_argument("--activate", action="store_true", help="Activate the target window on start.")
    parser.add_argument("--debug", action="store_true", help="Show detection preview. Press q to exit.")
    return parser.parse_args()


if __name__ == "__main__":
    run_bot(parse_args())

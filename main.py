import argparse
import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np
import pyautogui
import pygetwindow as gw


WINDOW_TITLE = "MSI App Player"
GOLD_LOWER_HSV = np.array([20, 100, 100], dtype=np.uint8)
GOLD_UPPER_HSV = np.array([30, 255, 255], dtype=np.uint8)
SECTORS = ("left", "center", "right")
ROI_TOP_RATIO = 0.60
ROI_BOTTOM_RATIO = 0.85
COOLDOWN_SECONDS = 0.20


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class SectorStats:
    name: str
    index: int
    x1: int
    x2: int
    gold_pixels: int


def find_window(title: str):
    windows = [window for window in gw.getWindowsWithTitle(title) if window.width > 0 and window.height > 0]
    if not windows:
        raise RuntimeError(f'Window with title "{title}" was not found.')
    return windows[0]


def get_decision_region(window) -> CaptureRegion:
    left = max(0, int(window.left))
    top = max(0, int(window.top))
    width = max(1, int(window.width))
    height = max(1, int(window.height))
    roi_top = top + int(height * ROI_TOP_RATIO)
    roi_bottom = top + int(height * ROI_BOTTOM_RATIO)
    roi_height = max(1, roi_bottom - roi_top)
    return CaptureRegion(left=left, top=roi_top, width=width, height=roi_height)


def region_to_monitor(region: CaptureRegion) -> dict[str, int]:
    return {
        "left": region.left,
        "top": region.top,
        "width": region.width,
        "height": region.height,
    }


def grab_frame(sct, region: CaptureRegion) -> np.ndarray:
    return np.asarray(sct.grab(region_to_monitor(region)), dtype=np.uint8)


def build_gold_mask(frame_bgra: np.ndarray) -> np.ndarray:
    frame_bgr = frame_bgra[:, :, :3]
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(frame_hsv, GOLD_LOWER_HSV, GOLD_UPPER_HSV)


def split_sector_bounds(width: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    first = width // 3
    second = (width * 2) // 3
    return ((0, first), (first, second), (second, width))


def count_gold_by_sector(mask: np.ndarray, bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> list[SectorStats]:
    stats = []

    for index, (x1, x2) in enumerate(bounds):
        sector_mask = mask[:, x1:x2]
        stats.append(
            SectorStats(
                name=SECTORS[index],
                index=index,
                x1=x1,
                x2=x2,
                gold_pixels=cv2.countNonZero(sector_mask),
            )
        )

    return stats


def choose_escape_sector(stats: list[SectorStats], current_sector_index: int) -> SectorStats:
    zero_sectors = [sector for sector in stats if sector.gold_pixels == 0]
    candidates = zero_sectors or stats
    return min(
        candidates,
        key=lambda sector: (
            sector.gold_pixels,
            abs(sector.index - current_sector_index),
            abs(sector.index - 1),
        ),
    )


def next_move(current_sector_index: int, target_sector_index: int) -> str | None:
    if target_sector_index < current_sector_index:
        return "left"
    if target_sector_index > current_sector_index:
        return "right"
    return None


def draw_debug_mask(
    mask: np.ndarray,
    stats: list[SectorStats],
    current_sector_index: int,
    target_sector_index: int,
) -> np.ndarray:
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    height, width = mask.shape[:2]

    for x in (width // 3, (width * 2) // 3):
        cv2.line(debug, (x, 0), (x, height), (0, 255, 255), 2)

    for sector in stats:
        color = (0, 255, 0) if sector.index == target_sector_index else (255, 255, 255)
        if sector.index == current_sector_index:
            cv2.rectangle(debug, (sector.x1 + 3, 3), (sector.x2 - 3, height - 3), (255, 0, 0), 2)

        cv2.putText(
            debug,
            f"{sector.name}: {sector.gold_pixels}",
            (sector.x1 + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        debug,
        f"current={SECTORS[current_sector_index]} target={SECTORS[target_sector_index]}",
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return debug


def run(args: argparse.Namespace) -> None:
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True

    window = find_window(args.title)
    if args.activate:
        window.activate()
        time.sleep(0.2)

    current_sector_index = SECTORS.index(args.start_sector)
    last_press_at = 0.0

    print(
        f'Watching "{args.title}" ROI {ROI_TOP_RATIO:.0%}-{ROI_BOTTOM_RATIO:.0%}. '
        "Press q in the OpenCV window to quit, or move mouse to a screen corner."
    )

    with mss.mss() as sct:
        cached_width = 0
        cached_bounds = split_sector_bounds(1)

        while True:
            try:
                region = get_decision_region(window)
                frame_bgra = grab_frame(sct, region)
                mask = build_gold_mask(frame_bgra)
                if region.width != cached_width:
                    cached_width = region.width
                    cached_bounds = split_sector_bounds(region.width)

                stats = count_gold_by_sector(mask, cached_bounds)
                current_gold_pixels = stats[current_sector_index].gold_pixels
                target_sector = (
                    choose_escape_sector(stats, current_sector_index)
                    if current_gold_pixels >= args.coin_threshold
                    else stats[current_sector_index]
                )

                now = time.time()
                move = next_move(current_sector_index, target_sector.index)
                if move and now - last_press_at >= args.cooldown:
                    pyautogui.press(move)
                    current_sector_index += -1 if move == "left" else 1
                    last_press_at = now
                    print(
                        f"gold pixels: left={stats[0].gold_pixels}, "
                        f"center={stats[1].gold_pixels}, right={stats[2].gold_pixels}; "
                        f"pressed {move}, current={SECTORS[current_sector_index]}"
                    )

                debug = draw_debug_mask(mask, stats, current_sector_index, target_sector.index)
                cv2.imshow("No Coin sector mask", debug)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except KeyboardInterrupt:
                break
            except pyautogui.FailSafeException:
                print("PyAutoGUI fail-safe triggered.")
                break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Watch the "MSI App Player" window and keep the player in the sector '
            "with no coins, or with the fewest gold pixels."
        )
    )
    parser.add_argument("--title", default=WINDOW_TITLE, help="Target window title.")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=COOLDOWN_SECONDS,
        help="Minimum seconds between left/right key presses.",
    )
    parser.add_argument(
        "--start-sector",
        choices=SECTORS,
        default="center",
        help="Initial player sector assumed by the bot.",
    )
    parser.add_argument(
        "--coin-threshold",
        type=int,
        default=1,
        help="Minimum yellow pixels in the current sector before the bot changes lanes.",
    )
    parser.add_argument("--activate", action="store_true", help="Activate the target window on start.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

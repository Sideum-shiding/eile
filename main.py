import argparse
import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np
import pyautogui
import pygetwindow as gw


WINDOW_TITLE = "MSI App Player"
LANES = ("left", "center", "right")

ROI_TOP_RATIO = 0.45
ROI_BOTTOM_RATIO = 0.92
COOLDOWN_SECONDS = 0.20
VERTICAL_COOLDOWN_SECONDS = 0.35
KEY_HOLD_SECONDS = 0.04

COIN_LOWER_HSV = np.array([18, 80, 120], dtype=np.uint8)
COIN_UPPER_HSV = np.array([36, 255, 255], dtype=np.uint8)

SATURATED_LOWER_HSV = np.array([0, 70, 45], dtype=np.uint8)
SATURATED_UPPER_HSV = np.array([179, 255, 255], dtype=np.uint8)
DARK_LOWER_HSV = np.array([0, 0, 0], dtype=np.uint8)
DARK_UPPER_HSV = np.array([179, 255, 80], dtype=np.uint8)


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class LaneStats:
    name: str
    index: int
    x1: int
    x2: int
    danger_pixels: int
    coin_pixels: int
    top_danger_pixels: int
    bottom_danger_pixels: int


@dataclass(frozen=True)
class VisionMasks:
    danger: np.ndarray
    coins: np.ndarray
    edges: np.ndarray


def find_window(title: str):
    windows = [window for window in gw.getWindowsWithTitle(title) if window.width > 0 and window.height > 0]
    if not windows:
        raise RuntimeError(f'Window with title "{title}" was not found.')
    return windows[0]


def focus_window(window) -> None:
    try:
        if window.isMinimized:
            window.restore()
            time.sleep(0.05)
        window.activate()
    except Exception as exc:
        print(f"Could not activate target window: {exc}")


def get_gameplay_region(window) -> CaptureRegion:
    left = max(0, int(window.left))
    top = max(0, int(window.top))
    width = max(1, int(window.width))
    height = max(1, int(window.height))
    roi_top = top + int(height * ROI_TOP_RATIO)
    roi_bottom = top + int(height * ROI_BOTTOM_RATIO)
    return CaptureRegion(left=left, top=roi_top, width=width, height=max(1, roi_bottom - roi_top))


def region_to_monitor(region: CaptureRegion) -> dict[str, int]:
    return {
        "left": region.left,
        "top": region.top,
        "width": region.width,
        "height": region.height,
    }


def grab_frame(sct, region: CaptureRegion) -> np.ndarray:
    return np.asarray(sct.grab(region_to_monitor(region)), dtype=np.uint8)


def split_lane_bounds(width: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    first = width // 3
    second = (width * 2) // 3
    return ((0, first), (first, second), (second, width))


def build_masks(frame_bgra: np.ndarray) -> VisionMasks:
    frame_bgr = frame_bgra[:, :, :3]
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    coin_mask = cv2.inRange(frame_hsv, COIN_LOWER_HSV, COIN_UPPER_HSV)
    saturated_mask = cv2.inRange(frame_hsv, SATURATED_LOWER_HSV, SATURATED_UPPER_HSV)
    dark_mask = cv2.inRange(frame_hsv, DARK_LOWER_HSV, DARK_UPPER_HSV)
    edge_mask = cv2.Canny(gray, 70, 150)

    kernel = np.ones((3, 3), dtype=np.uint8)
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
    coin_mask = cv2.morphologyEx(coin_mask, cv2.MORPH_OPEN, kernel)

    raw_danger = cv2.bitwise_or(saturated_mask, dark_mask)
    raw_danger = cv2.bitwise_or(raw_danger, edge_mask)
    raw_danger = cv2.bitwise_and(raw_danger, cv2.bitwise_not(coin_mask))
    danger_mask = cv2.morphologyEx(raw_danger, cv2.MORPH_OPEN, kernel)
    danger_mask = cv2.morphologyEx(danger_mask, cv2.MORPH_CLOSE, kernel)

    return VisionMasks(danger=danger_mask, coins=coin_mask, edges=edge_mask)


def count_lane_stats(
    masks: VisionMasks,
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> list[LaneStats]:
    height = masks.danger.shape[0]
    split_y = int(height * 0.48)
    stats = []

    for index, (x1, x2) in enumerate(bounds):
        danger_lane = masks.danger[:, x1:x2]
        coin_lane = masks.coins[:, x1:x2]
        top_danger = danger_lane[:split_y, :]
        bottom_danger = danger_lane[split_y:, :]

        stats.append(
            LaneStats(
                name=LANES[index],
                index=index,
                x1=x1,
                x2=x2,
                danger_pixels=cv2.countNonZero(danger_lane),
                coin_pixels=cv2.countNonZero(coin_lane),
                top_danger_pixels=cv2.countNonZero(top_danger),
                bottom_danger_pixels=cv2.countNonZero(bottom_danger),
            )
        )

    return stats


def danger_score(lane: LaneStats) -> int:
    return lane.danger_pixels + lane.bottom_danger_pixels * 2


def choose_best_lane(stats: list[LaneStats], current_lane_index: int) -> LaneStats:
    return min(
        stats,
        key=lambda lane: (
            danger_score(lane),
            lane.coin_pixels,
            abs(lane.index - current_lane_index),
            abs(lane.index - 1),
        ),
    )


def horizontal_move(current_lane_index: int, target_lane_index: int) -> str | None:
    if target_lane_index < current_lane_index:
        return "left"
    if target_lane_index > current_lane_index:
        return "right"
    return None


def vertical_move(current_lane: LaneStats, *, jump_bias: float) -> str:
    if current_lane.bottom_danger_pixels >= current_lane.top_danger_pixels * jump_bias:
        return "up"
    return "down"


def press_game_key(window, key: str, *, focus_before_press: bool, key_hold: float) -> None:
    if focus_before_press:
        focus_window(window)
        time.sleep(0.02)

    pyautogui.keyDown(key)
    time.sleep(key_hold)
    pyautogui.keyUp(key)


def choose_action(
    stats: list[LaneStats],
    current_lane_index: int,
    *,
    lane_danger_threshold: int,
    warning_threshold: int,
    jump_bias: float,
) -> tuple[str | None, int]:
    current_lane = stats[current_lane_index]
    best_lane = choose_best_lane(stats, current_lane_index)
    current_score = danger_score(current_lane)
    best_score = danger_score(best_lane)

    if current_score < warning_threshold:
        return None, current_lane_index

    move = horizontal_move(current_lane_index, best_lane.index)
    if move and best_score < current_score and best_score < lane_danger_threshold:
        return move, best_lane.index

    if current_score >= lane_danger_threshold:
        return vertical_move(current_lane, jump_bias=jump_bias), current_lane_index

    if move and best_score + warning_threshold < current_score:
        return move, best_lane.index

    return None, current_lane_index


def print_stats(stats: list[LaneStats], current_lane_index: int, action: str | None, prefix: str = "stats") -> None:
    lane_text = ", ".join(
        f"{lane.name}=danger:{lane.danger_pixels}/score:{danger_score(lane)}/coins:{lane.coin_pixels}"
        for lane in stats
    )
    print(f"{prefix}: {lane_text}; current={LANES[current_lane_index]}; action={action or 'none'}")


def draw_debug(
    masks: VisionMasks,
    stats: list[LaneStats],
    current_lane_index: int,
    target_lane_index: int,
    action: str | None,
) -> np.ndarray:
    danger_bgr = cv2.cvtColor(masks.danger, cv2.COLOR_GRAY2BGR)
    coin_layer = np.zeros_like(danger_bgr)
    coin_layer[:, :, 1] = masks.coins
    debug = cv2.addWeighted(danger_bgr, 1.0, coin_layer, 0.7, 0)
    height, width = masks.danger.shape[:2]

    for x in (width // 3, (width * 2) // 3):
        cv2.line(debug, (x, 0), (x, height), (0, 255, 255), 2)

    cv2.line(debug, (0, int(height * 0.48)), (width, int(height * 0.48)), (255, 255, 0), 1)

    for lane in stats:
        color = (0, 255, 0) if lane.index == target_lane_index else (255, 255, 255)
        if lane.index == current_lane_index:
            cv2.rectangle(debug, (lane.x1 + 3, 3), (lane.x2 - 3, height - 3), (255, 0, 0), 2)

        cv2.putText(
            debug,
            f"{lane.name} d:{danger_score(lane)} c:{lane.coin_pixels}",
            (lane.x1 + 8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        debug,
        f"current={LANES[current_lane_index]} target={LANES[target_lane_index]} action={action or 'none'}",
        (10, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
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
        focus_window(window)
        time.sleep(0.2)

    current_lane_index = LANES.index(args.start_lane)
    last_horizontal_at = 0.0
    last_vertical_at = 0.0
    last_stats_at = 0.0

    if args.test_press:
        print("Sending test presses: left, right, up, down.")
        for key in ("left", "right", "up", "down"):
            press_game_key(window, key, focus_before_press=True, key_hold=args.key_hold)
            time.sleep(args.cooldown)

    print(
        f'Watching "{args.title}" ROI {ROI_TOP_RATIO:.0%}-{ROI_BOTTOM_RATIO:.0%}. '
        "Move mouse to a screen corner or press q in debug window to stop."
    )

    with mss.mss() as sct:
        cached_width = 0
        cached_bounds = split_lane_bounds(1)

        while True:
            try:
                region = get_gameplay_region(window)
                frame_bgra = grab_frame(sct, region)
                masks = build_masks(frame_bgra)

                if region.width != cached_width:
                    cached_width = region.width
                    cached_bounds = split_lane_bounds(region.width)

                stats = count_lane_stats(masks, cached_bounds)
                action, target_lane_index = choose_action(
                    stats,
                    current_lane_index,
                    lane_danger_threshold=args.lane_danger_threshold,
                    warning_threshold=args.warning_threshold,
                    jump_bias=args.jump_bias,
                )

                now = time.time()
                can_press_horizontal = action in ("left", "right") and now - last_horizontal_at >= args.cooldown
                can_press_vertical = action in ("up", "down") and now - last_vertical_at >= args.vertical_cooldown

                if can_press_horizontal or can_press_vertical:
                    press_game_key(
                        window,
                        action,
                        focus_before_press=not args.no_focus_before_press,
                        key_hold=args.key_hold,
                    )
                    if action in ("left", "right"):
                        current_lane_index += -1 if action == "left" else 1
                        current_lane_index = max(0, min(len(LANES) - 1, current_lane_index))
                        last_horizontal_at = now
                    else:
                        last_vertical_at = now

                    print_stats(stats, current_lane_index, action, prefix=f"pressed {action}")
                elif args.print_stats and now - last_stats_at >= args.stats_interval:
                    print_stats(stats, current_lane_index, action)
                    last_stats_at = now

                if not args.no_debug:
                    debug = draw_debug(masks, stats, current_lane_index, target_lane_index, action)
                    cv2.imshow("Subway Surfers avoidance mask", debug)
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
        description='Avoid obstacles in Subway Surfers running inside the "MSI App Player" window.'
    )
    parser.add_argument("--title", default=WINDOW_TITLE, help="Target window title.")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=COOLDOWN_SECONDS,
        help="Minimum seconds between left/right lane changes.",
    )
    parser.add_argument(
        "--vertical-cooldown",
        type=float,
        default=VERTICAL_COOLDOWN_SECONDS,
        help="Minimum seconds between jump/roll actions.",
    )
    parser.add_argument(
        "--key-hold",
        type=float,
        default=KEY_HOLD_SECONDS,
        help="Seconds to hold a key down. Some emulators miss ultra-short taps.",
    )
    parser.add_argument(
        "--start-lane",
        choices=LANES,
        default="center",
        help="Initial player lane assumed by the bot.",
    )
    parser.add_argument(
        "--lane-danger-threshold",
        type=int,
        default=4500,
        help="Danger score that triggers jump/roll if no safer lane is available.",
    )
    parser.add_argument(
        "--warning-threshold",
        type=int,
        default=1400,
        help="Danger score that starts proactive lane selection.",
    )
    parser.add_argument(
        "--jump-bias",
        type=float,
        default=1.20,
        help="Use jump when bottom danger is this many times stronger than top danger.",
    )
    parser.add_argument("--activate", action="store_true", help="Activate the target window on start.")
    parser.add_argument(
        "--no-focus-before-press",
        action="store_true",
        help="Do not activate the MSI App Player window before each key press.",
    )
    parser.add_argument(
        "--test-press",
        action="store_true",
        help="Send left, right, up, and down once at startup to verify emulator controls.",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print lane danger counters periodically.",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=0.5,
        help="Seconds between --print-stats log lines.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable cv2.imshow for maximum FPS.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

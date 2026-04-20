#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace {

constexpr wchar_t kDefaultWindowTitle[] = L"MSI App Player";
constexpr double kRoiTopRatio = 0.45;
constexpr double kRoiBottomRatio = 0.92;

constexpr int kDefaultHorizontalCooldownMs = 200;
constexpr int kDefaultVerticalCooldownMs = 350;
constexpr int kDefaultKeyHoldMs = 40;
constexpr int kDefaultWarningThreshold = 1400;
constexpr int kDefaultLaneDangerThreshold = 4500;
constexpr double kDefaultJumpBias = 1.20;

const cv::Scalar kCoinLowerHsv(18, 80, 120);
const cv::Scalar kCoinUpperHsv(36, 255, 255);
const cv::Scalar kSaturatedLowerHsv(0, 70, 45);
const cv::Scalar kSaturatedUpperHsv(179, 255, 255);
const cv::Scalar kDarkLowerHsv(0, 0, 0);
const cv::Scalar kDarkUpperHsv(179, 255, 80);

enum class Lane : int {
    Left = 0,
    Center = 1,
    Right = 2,
};

struct Args {
    std::wstring title = kDefaultWindowTitle;
    Lane start_lane = Lane::Center;
    int horizontal_cooldown_ms = kDefaultHorizontalCooldownMs;
    int vertical_cooldown_ms = kDefaultVerticalCooldownMs;
    int key_hold_ms = kDefaultKeyHoldMs;
    int warning_threshold = kDefaultWarningThreshold;
    int lane_danger_threshold = kDefaultLaneDangerThreshold;
    double jump_bias = kDefaultJumpBias;
    bool activate = false;
    bool focus_before_press = true;
    bool test_press = false;
    bool print_stats = false;
    int stats_interval_ms = 500;
    bool debug = true;
};

struct CaptureRegion {
    int left = 0;
    int top = 0;
    int width = 1;
    int height = 1;
};

struct LaneStats {
    std::string name;
    int index = 0;
    int x1 = 0;
    int x2 = 0;
    int danger_pixels = 0;
    int coin_pixels = 0;
    int top_danger_pixels = 0;
    int bottom_danger_pixels = 0;
};

struct Masks {
    cv::Mat danger;
    cv::Mat coins;
    cv::Mat edges;
};

std::string laneName(int index) {
    switch (index) {
        case 0:
            return "left";
        case 1:
            return "center";
        case 2:
            return "right";
        default:
            return "unknown";
    }
}

std::wstring widen(std::string_view text) {
    if (text.empty()) {
        return {};
    }

    const int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring result(size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), size);
    return result;
}

std::optional<Lane> parseLane(std::string_view value) {
    if (value == "left") {
        return Lane::Left;
    }
    if (value == "center") {
        return Lane::Center;
    }
    if (value == "right") {
        return Lane::Right;
    }
    return std::nullopt;
}

bool readValue(int argc, char** argv, int& i, std::string_view option, std::string& value) {
    if (std::string_view(argv[i]) != option) {
        return false;
    }
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + std::string(option));
    }
    value = argv[++i];
    return true;
}

Args parseArgs(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        std::string value;

        if (readValue(argc, argv, i, "--title", value)) {
            args.title = widen(value);
        } else if (readValue(argc, argv, i, "--start-lane", value)) {
            auto lane = parseLane(value);
            if (!lane) {
                throw std::runtime_error("--start-lane must be left, center, or right");
            }
            args.start_lane = *lane;
        } else if (readValue(argc, argv, i, "--cooldown", value)) {
            args.horizontal_cooldown_ms = static_cast<int>(std::stod(value) * 1000.0);
        } else if (readValue(argc, argv, i, "--vertical-cooldown", value)) {
            args.vertical_cooldown_ms = static_cast<int>(std::stod(value) * 1000.0);
        } else if (readValue(argc, argv, i, "--key-hold", value)) {
            args.key_hold_ms = static_cast<int>(std::stod(value) * 1000.0);
        } else if (readValue(argc, argv, i, "--warning-threshold", value)) {
            args.warning_threshold = std::stoi(value);
        } else if (readValue(argc, argv, i, "--lane-danger-threshold", value)) {
            args.lane_danger_threshold = std::stoi(value);
        } else if (readValue(argc, argv, i, "--jump-bias", value)) {
            args.jump_bias = std::stod(value);
        } else if (readValue(argc, argv, i, "--stats-interval", value)) {
            args.stats_interval_ms = static_cast<int>(std::stod(value) * 1000.0);
        } else if (option == "--activate") {
            args.activate = true;
        } else if (option == "--no-focus-before-press") {
            args.focus_before_press = false;
        } else if (option == "--test-press") {
            args.test_press = true;
        } else if (option == "--print-stats") {
            args.print_stats = true;
        } else if (option == "--no-debug") {
            args.debug = false;
        } else if (option == "--help" || option == "-h") {
            std::cout
                << "Usage: subway_bot_cpp [options]\n"
                << "  --title TEXT\n"
                << "  --activate\n"
                << "  --start-lane left|center|right\n"
                << "  --cooldown SECONDS\n"
                << "  --vertical-cooldown SECONDS\n"
                << "  --key-hold SECONDS\n"
                << "  --warning-threshold N\n"
                << "  --lane-danger-threshold N\n"
                << "  --jump-bias FLOAT\n"
                << "  --test-press\n"
                << "  --print-stats\n"
                << "  --stats-interval SECONDS\n"
                << "  --no-debug\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + std::string(option));
        }
    }

    return args;
}

HWND findTargetWindow(const std::wstring& title) {
    HWND hwnd = FindWindowW(nullptr, title.c_str());
    if (!hwnd) {
        throw std::runtime_error("Target window was not found.");
    }
    return hwnd;
}

void focusWindow(HWND hwnd) {
    if (IsIconic(hwnd)) {
        ShowWindow(hwnd, SW_RESTORE);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    SetForegroundWindow(hwnd);
}

CaptureRegion getGameplayRegion(HWND hwnd) {
    RECT rect{};
    if (!GetWindowRect(hwnd, &rect)) {
        throw std::runtime_error("GetWindowRect failed.");
    }

    const int window_width = static_cast<int>(std::max<LONG>(1, rect.right - rect.left));
    const int window_height = static_cast<int>(std::max<LONG>(1, rect.bottom - rect.top));
    const int roi_top = rect.top + static_cast<int>(window_height * kRoiTopRatio);
    const int roi_bottom = rect.top + static_cast<int>(window_height * kRoiBottomRatio);

    return CaptureRegion{
        static_cast<int>(std::max<LONG>(0, rect.left)),
        std::max(0, roi_top),
        static_cast<int>(window_width),
        std::max(1, roi_bottom - roi_top),
    };
}

class ScreenCapture {
public:
    ScreenCapture() {
        screen_dc_ = GetDC(nullptr);
        memory_dc_ = CreateCompatibleDC(screen_dc_);
        if (!screen_dc_ || !memory_dc_) {
            throw std::runtime_error("Failed to create screen capture DCs.");
        }
    }

    ~ScreenCapture() {
        if (bitmap_) {
            DeleteObject(bitmap_);
        }
        if (memory_dc_) {
            DeleteDC(memory_dc_);
        }
        if (screen_dc_) {
            ReleaseDC(nullptr, screen_dc_);
        }
    }

    cv::Mat grab(const CaptureRegion& region) {
        ensureBuffer(region.width, region.height);

        if (!BitBlt(memory_dc_, 0, 0, region.width, region.height, screen_dc_, region.left, region.top, SRCCOPY | CAPTUREBLT)) {
            throw std::runtime_error("BitBlt failed.");
        }

        return cv::Mat(region.height, region.width, CV_8UC4, pixels_);
    }

private:
    void ensureBuffer(int width, int height) {
        if (bitmap_ && width == width_ && height == height_) {
            return;
        }

        if (bitmap_) {
            DeleteObject(bitmap_);
            bitmap_ = nullptr;
            pixels_ = nullptr;
        }

        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = width;
        bmi.bmiHeader.biHeight = -height;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        bitmap_ = CreateDIBSection(screen_dc_, &bmi, DIB_RGB_COLORS, &pixels_, nullptr, 0);
        if (!bitmap_ || !pixels_) {
            throw std::runtime_error("CreateDIBSection failed.");
        }

        SelectObject(memory_dc_, bitmap_);
        width_ = width;
        height_ = height;
    }

    HDC screen_dc_ = nullptr;
    HDC memory_dc_ = nullptr;
    HBITMAP bitmap_ = nullptr;
    void* pixels_ = nullptr;
    int width_ = 0;
    int height_ = 0;
};

Masks buildMasks(const cv::Mat& frame_bgra) {
    cv::Mat frame_bgr;
    cv::Mat frame_hsv;
    cv::Mat gray;
    cv::cvtColor(frame_bgra, frame_bgr, cv::COLOR_BGRA2BGR);
    cv::cvtColor(frame_bgr, frame_hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);

    Masks masks;
    cv::Mat saturated;
    cv::Mat dark;
    cv::Mat raw_danger;
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});

    cv::inRange(frame_hsv, kCoinLowerHsv, kCoinUpperHsv, masks.coins);
    cv::inRange(frame_hsv, kSaturatedLowerHsv, kSaturatedUpperHsv, saturated);
    cv::inRange(frame_hsv, kDarkLowerHsv, kDarkUpperHsv, dark);
    cv::Canny(gray, masks.edges, 70, 150);

    cv::dilate(masks.edges, masks.edges, kernel);
    cv::morphologyEx(masks.coins, masks.coins, cv::MORPH_OPEN, kernel);

    cv::Mat not_coins;
    cv::bitwise_or(saturated, dark, raw_danger);
    cv::bitwise_or(raw_danger, masks.edges, raw_danger);
    cv::bitwise_not(masks.coins, not_coins);
    cv::bitwise_and(raw_danger, not_coins, raw_danger);
    cv::morphologyEx(raw_danger, masks.danger, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(masks.danger, masks.danger, cv::MORPH_CLOSE, kernel);

    return masks;
}

std::array<std::pair<int, int>, 3> splitLaneBounds(int width) {
    const int first = width / 3;
    const int second = (width * 2) / 3;
    return {{{0, first}, {first, second}, {second, width}}};
}

std::vector<LaneStats> countLaneStats(const Masks& masks, const std::array<std::pair<int, int>, 3>& bounds) {
    const int split_y = static_cast<int>(masks.danger.rows * 0.48);
    std::vector<LaneStats> stats;
    stats.reserve(3);

    for (int index = 0; index < 3; ++index) {
        const auto [x1, x2] = bounds[index];
        const cv::Range x_range(x1, x2);
        const cv::Mat danger_lane = masks.danger(cv::Range::all(), x_range);
        const cv::Mat coin_lane = masks.coins(cv::Range::all(), x_range);
        const cv::Mat top_danger = danger_lane(cv::Range(0, split_y), cv::Range::all());
        const cv::Mat bottom_danger = danger_lane(cv::Range(split_y, danger_lane.rows), cv::Range::all());

        stats.push_back(LaneStats{
            laneName(index),
            index,
            x1,
            x2,
            cv::countNonZero(danger_lane),
            cv::countNonZero(coin_lane),
            cv::countNonZero(top_danger),
            cv::countNonZero(bottom_danger),
        });
    }

    return stats;
}

int dangerScore(const LaneStats& lane) {
    return lane.danger_pixels + lane.bottom_danger_pixels * 2;
}

const LaneStats& chooseBestLane(const std::vector<LaneStats>& stats, int current_lane) {
    return *std::min_element(stats.begin(), stats.end(), [current_lane](const LaneStats& a, const LaneStats& b) {
        const auto key_a = std::tuple(dangerScore(a), a.coin_pixels, std::abs(a.index - current_lane), std::abs(a.index - 1));
        const auto key_b = std::tuple(dangerScore(b), b.coin_pixels, std::abs(b.index - current_lane), std::abs(b.index - 1));
        return key_a < key_b;
    });
}

std::optional<WORD> horizontalMove(int current_lane, int target_lane) {
    if (target_lane < current_lane) {
        return VK_LEFT;
    }
    if (target_lane > current_lane) {
        return VK_RIGHT;
    }
    return std::nullopt;
}

WORD verticalMove(const LaneStats& current_lane, double jump_bias) {
    if (current_lane.bottom_danger_pixels >= static_cast<int>(current_lane.top_danger_pixels * jump_bias)) {
        return VK_UP;
    }
    return VK_DOWN;
}

std::string keyName(WORD key) {
    switch (key) {
        case VK_LEFT:
            return "left";
        case VK_RIGHT:
            return "right";
        case VK_UP:
            return "up";
        case VK_DOWN:
            return "down";
        default:
            return "unknown";
    }
}

std::pair<std::optional<WORD>, int> chooseAction(
    const std::vector<LaneStats>& stats,
    int current_lane,
    const Args& args
) {
    const LaneStats& current = stats[current_lane];
    const LaneStats& best = chooseBestLane(stats, current_lane);
    const int current_score = dangerScore(current);
    const int best_score = dangerScore(best);

    if (current_score < args.warning_threshold) {
        return {std::nullopt, current_lane};
    }

    const std::optional<WORD> move = horizontalMove(current_lane, best.index);
    if (move && best_score < current_score && best_score < args.lane_danger_threshold) {
        return {move, best.index};
    }

    if (current_score >= args.lane_danger_threshold) {
        return {verticalMove(current, args.jump_bias), current_lane};
    }

    if (move && best_score + args.warning_threshold < current_score) {
        return {move, best.index};
    }

    return {std::nullopt, current_lane};
}

void pressKey(HWND hwnd, WORD key, const Args& args) {
    if (args.focus_before_press) {
        focusWindow(hwnd);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    INPUT inputs[2]{};
    inputs[0].type = INPUT_KEYBOARD;
    inputs[0].ki.wVk = key;
    inputs[1].type = INPUT_KEYBOARD;
    inputs[1].ki.wVk = key;
    inputs[1].ki.dwFlags = KEYEVENTF_KEYUP;

    SendInput(1, &inputs[0], sizeof(INPUT));
    std::this_thread::sleep_for(std::chrono::milliseconds(args.key_hold_ms));
    SendInput(1, &inputs[1], sizeof(INPUT));
}

void printStats(const std::vector<LaneStats>& stats, int current_lane, std::optional<WORD> action, std::string_view prefix) {
    std::cout << prefix << ": ";
    for (const LaneStats& lane : stats) {
        std::cout << lane.name << "=danger:" << lane.danger_pixels << "/score:" << dangerScore(lane)
                  << "/coins:" << lane.coin_pixels << " ";
    }
    std::cout << "current=" << laneName(current_lane) << " action="
              << (action ? keyName(*action) : "none") << "\n";
}

cv::Mat drawDebug(const Masks& masks, const std::vector<LaneStats>& stats, int current_lane, int target_lane, std::optional<WORD> action) {
    cv::Mat danger_bgr;
    cv::cvtColor(masks.danger, danger_bgr, cv::COLOR_GRAY2BGR);

    cv::Mat coin_layer = cv::Mat::zeros(danger_bgr.size(), danger_bgr.type());
    std::vector<cv::Mat> channels;
    cv::split(coin_layer, channels);
    channels[1] = masks.coins;
    cv::merge(channels, coin_layer);

    cv::Mat debug;
    cv::addWeighted(danger_bgr, 1.0, coin_layer, 0.7, 0.0, debug);

    const int height = debug.rows;
    const int width = debug.cols;
    cv::line(debug, {width / 3, 0}, {width / 3, height}, {0, 255, 255}, 2);
    cv::line(debug, {(width * 2) / 3, 0}, {(width * 2) / 3, height}, {0, 255, 255}, 2);
    cv::line(debug, {0, static_cast<int>(height * 0.48)}, {width, static_cast<int>(height * 0.48)}, {255, 255, 0}, 1);

    for (const LaneStats& lane : stats) {
        const cv::Scalar color = lane.index == target_lane ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 255);
        if (lane.index == current_lane) {
            cv::rectangle(debug, {lane.x1 + 3, 3}, {lane.x2 - 3, height - 3}, {255, 0, 0}, 2);
        }
        cv::putText(
            debug,
            lane.name + " d:" + std::to_string(dangerScore(lane)) + " c:" + std::to_string(lane.coin_pixels),
            {lane.x1 + 8, 28},
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv::LINE_AA
        );
    }

    cv::putText(
        debug,
        "current=" + laneName(current_lane) + " target=" + laneName(target_lane) + " action=" + (action ? keyName(*action) : "none"),
        {10, height - 14},
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        {0, 255, 255},
        2,
        cv::LINE_AA
    );

    return debug;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        HWND hwnd = findTargetWindow(args.title);

        if (args.activate) {
            focusWindow(hwnd);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        if (args.test_press) {
            std::cout << "Sending test presses: left, right, up, down.\n";
            for (const WORD key : {VK_LEFT, VK_RIGHT, VK_UP, VK_DOWN}) {
                pressKey(hwnd, key, args);
                std::this_thread::sleep_for(std::chrono::milliseconds(args.horizontal_cooldown_ms));
            }
        }

        int current_lane = static_cast<int>(args.start_lane);
        auto last_horizontal = std::chrono::steady_clock::now() - std::chrono::seconds(1);
        auto last_vertical = std::chrono::steady_clock::now() - std::chrono::seconds(1);
        auto last_stats = std::chrono::steady_clock::now() - std::chrono::seconds(1);

        ScreenCapture capture;
        int cached_width = 0;
        auto lane_bounds = splitLaneBounds(1);

        std::cout << "Watching \"MSI App Player\" ROI 45%-92%. Press q in debug window or Ctrl+C to stop.\n";

        while (true) {
            hwnd = findTargetWindow(args.title);
            const CaptureRegion region = getGameplayRegion(hwnd);
            const cv::Mat frame = capture.grab(region);
            const Masks masks = buildMasks(frame);

            if (region.width != cached_width) {
                cached_width = region.width;
                lane_bounds = splitLaneBounds(region.width);
            }

            const std::vector<LaneStats> stats = countLaneStats(masks, lane_bounds);
            const auto [action, target_lane] = chooseAction(stats, current_lane, args);
            const auto now = std::chrono::steady_clock::now();

            const bool horizontal_ready =
                action && (*action == VK_LEFT || *action == VK_RIGHT) &&
                now - last_horizontal >= std::chrono::milliseconds(args.horizontal_cooldown_ms);
            const bool vertical_ready =
                action && (*action == VK_UP || *action == VK_DOWN) &&
                now - last_vertical >= std::chrono::milliseconds(args.vertical_cooldown_ms);

            if (horizontal_ready || vertical_ready) {
                pressKey(hwnd, *action, args);
                if (*action == VK_LEFT || *action == VK_RIGHT) {
                    current_lane += *action == VK_LEFT ? -1 : 1;
                    current_lane = std::clamp(current_lane, 0, 2);
                    last_horizontal = now;
                } else {
                    last_vertical = now;
                }
                printStats(stats, current_lane, action, "pressed " + keyName(*action));
            } else if (args.print_stats && now - last_stats >= std::chrono::milliseconds(args.stats_interval_ms)) {
                printStats(stats, current_lane, action, "stats");
                last_stats = now;
            }

            if (args.debug) {
                cv::imshow("Subway Surfers C++ avoidance mask", drawDebug(masks, stats, current_lane, target_lane, action));
                if ((cv::waitKey(1) & 0xFF) == 'q') {
                    break;
                }
            }
        }

        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}

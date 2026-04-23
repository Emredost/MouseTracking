#pragma once

#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <windowsx.h>
// Undefine Windows macros that conflict with our enum
#ifdef ERROR
#undef ERROR
#endif
#endif

// Version information
#define MOUSETRACKING_VERSION_MAJOR 1
#define MOUSETRACKING_VERSION_MINOR 0
#define MOUSETRACKING_VERSION_PATCH 0

// Performance constants
constexpr int SCREEN_WIDTH = 1920;
constexpr int SCREEN_HEIGHT = 1080;
constexpr double SAMPLING_RATE_HZ = 15.0;
constexpr double SLEEP_TIME_MS = (1000.0 / SAMPLING_RATE_HZ) * 0.9; // 90% for responsiveness

// Research-based gaze tracking parameters (from literature)
constexpr double DISPERSION_THRESHOLD = 1.0;           // degrees of visual angle
constexpr double MIN_FIXATION_DURATION_MS = 150.0;     // Salvucci & Goldberg, 2000
constexpr double SACCADE_VELOCITY_THRESHOLD = 30.0;    // degrees/second
constexpr double PURSUIT_MIN_DURATION_MS = 300.0;      // minimum smooth pursuit duration

// Coordinate bounds
constexpr double MIN_NORMALIZED_COORD = 0.05;
constexpr double MAX_NORMALIZED_COORD = 0.95;

// Utility macros
#define SAFE_DELETE(ptr) do { delete ptr; ptr = nullptr; } while(0)
#define SAFE_DELETE_ARRAY(ptr) do { delete[] ptr; ptr = nullptr; } while(0)

// High-resolution timing
using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::duration<double>;

inline double GetCurrentTimeSeconds() {
    static auto start = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    return duration.count() / 1000000.0;
}

// Thread-safe logging level
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

// Forward declarations
class MouseTracker;
class TobiiConsumerTracker;
class SyncTracker;
class FileLogger;
class GUI; 
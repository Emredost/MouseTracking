#include "MouseTracker.h"
#include "FileLogger.h"

// Static instance for callback access
MouseTracker* MouseTracker::instance_ = nullptr;

MouseTracker::MouseTracker() 
    : isRunning_(false), mouseHook_(nullptr), startTime_(0.0), totalDistance_(0.0),
      lastPosition_{0, 0}, screenWidth_(SCREEN_WIDTH), screenHeight_(SCREEN_HEIGHT),
      lastEventTime_(0.0) {
    
    instance_ = this;
    events_.reserve(10000); // Pre-allocate for performance
    
    // Detect actual screen resolution
    RECT desktop;
    const HWND hDesktop = GetDesktopWindow();
    if (GetWindowRect(hDesktop, &desktop)) {
        screenWidth_ = desktop.right;
        screenHeight_ = desktop.bottom;
        FileLogger::Log(LogLevel::INFO, "Screen resolution detected: " + 
                       std::to_string(screenWidth_) + "x" + std::to_string(screenHeight_));
    }
}

MouseTracker::~MouseTracker() {
    Stop();
    instance_ = nullptr;
}

bool MouseTracker::Start() {
    if (isRunning_) {
        FileLogger::Log(LogLevel::WARNING, "MouseTracker already running");
        return true;
    }

    // Install low-level mouse hook for maximum performance
    mouseHook_ = SetWindowsHookEx(WH_MOUSE_LL, MouseHookProc, GetModuleHandle(nullptr), 0);
    if (!mouseHook_) {
        FileLogger::Log(LogLevel::ERROR, "Failed to install mouse hook. Error: " + std::to_string(GetLastError()));
        return false;
    }

    isRunning_ = true;
    startTime_ = GetCurrentTimeSeconds();
    totalDistance_ = 0.0;
    
    FileLogger::Log(LogLevel::INFO, "MouseTracker started with high-performance Windows API hooks");
    return true;
}

void MouseTracker::Stop() {
    if (!isRunning_) {
        return;
    }

    isRunning_ = false;

    if (mouseHook_) {
        UnhookWindowsHookEx(mouseHook_);
        mouseHook_ = nullptr;
    }

    auto stats = GetStats();
    FileLogger::Log(LogLevel::INFO, "MouseTracker stopped. Events: " + std::to_string(stats.mouseEventCount) +
                   ", Duration: " + std::to_string(stats.totalDuration) + "s, Distance: " + 
                   std::to_string(stats.totalMouseDistance) + " pixels");
}

LRESULT CALLBACK MouseTracker::MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0 && instance_ && instance_->isRunning_) {
        auto* hookStruct = reinterpret_cast<MSLLHOOKSTRUCT*>(lParam);
        
        // High-performance coordinate extraction
        int x = static_cast<int>(hookStruct->pt.x);
        int y = static_cast<int>(hookStruct->pt.y);
        
        switch (wParam) {
            case WM_MOUSEMOVE:
                instance_->OnMouseMove(x, y);
                break;
                
            case WM_LBUTTONDOWN:
                instance_->OnMouseClick(x, y, "left", true);
                break;
                
            case WM_LBUTTONUP:
                instance_->OnMouseClick(x, y, "left", false);
                break;
                
            case WM_RBUTTONDOWN:
                instance_->OnMouseClick(x, y, "right", true);
                break;
                
            case WM_RBUTTONUP:
                instance_->OnMouseClick(x, y, "right", false);
                break;
                
            case WM_MBUTTONDOWN:
                instance_->OnMouseClick(x, y, "middle", true);
                break;
                
            case WM_MBUTTONUP:
                instance_->OnMouseClick(x, y, "middle", false);
                break;
                
            case WM_MOUSEWHEEL: {
                int delta = GET_WHEEL_DELTA_WPARAM(hookStruct->mouseData);
                instance_->OnMouseScroll(x, y, delta);
                break;
            }
        }
    }
    
    return CallNextHookEx(nullptr, nCode, wParam, lParam);
}

void MouseTracker::OnMouseMove(int x, int y) {
    double currentTime = GetCurrentTimeSeconds();
    
    // Performance optimization: limit event frequency
    if (currentTime - lastEventTime_ < MIN_EVENT_INTERVAL) {
        return;
    }
    
    // Clamp coordinates to screen bounds
    auto [clampedX, clampedY] = ClampCoordinates(x, y);
    
    // Calculate distance for statistics
    if (lastPosition_.first != 0 || lastPosition_.second != 0) {
        double dx = clampedX - lastPosition_.first;
        double dy = clampedY - lastPosition_.second;
        totalDistance_ += std::sqrt(dx * dx + dy * dy);
    }
    
    lastPosition_ = {clampedX, clampedY};
    lastEventTime_ = currentTime;
    
    // Store event with move optimization
    events_.push_back(MouseEvent(currentTime, "move", clampedX, clampedY));
}

void MouseTracker::OnMouseClick(int x, int y, const std::string& button, bool pressed) {
    double currentTime = GetCurrentTimeSeconds();
    auto [clampedX, clampedY] = ClampCoordinates(x, y);
    
    MouseEvent event(currentTime, "click", clampedX, clampedY);
    event.button = button;
    event.pressed = pressed;
    
    events_.push_back(std::move(event));
    
    std::string action = pressed ? "pressed" : "released";
    FileLogger::Log(LogLevel::INFO, "Mouse " + action + " " + button + " at (" + 
                   std::to_string(clampedX) + ", " + std::to_string(clampedY) + ")");
}

void MouseTracker::OnMouseScroll(int x, int y, int delta) {
    double currentTime = GetCurrentTimeSeconds();
    auto [clampedX, clampedY] = ClampCoordinates(x, y);
    
    MouseEvent event(currentTime, "scroll", clampedX, clampedY);
    event.scrollDy = delta / WHEEL_DELTA; // Normalize scroll delta
    
    events_.push_back(std::move(event));
}

std::pair<int, int> MouseTracker::ClampCoordinates(int x, int y) const {
    int clampedX = std::max(0, std::min(screenWidth_ - 1, x));
    int clampedY = std::max(0, std::min(screenHeight_ - 1, y));
    
    // Log only if coordinates were actually clamped
    if (x != clampedX || y != clampedY) {
        FileLogger::Log(LogLevel::DEBUG, "Clamped coordinates from (" + std::to_string(x) + 
                       ", " + std::to_string(y) + ") to (" + std::to_string(clampedX) + 
                       ", " + std::to_string(clampedY) + ")");
    }
    
    return {clampedX, clampedY};
}

TrackingStats MouseTracker::GetStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    TrackingStats stats;
    stats.totalDuration = isRunning_ ? GetCurrentTimeSeconds() - startTime_ : 0.0;
    stats.mouseEventCount = events_.size();
    stats.totalMouseDistance = totalDistance_;
    
    return stats;
}

void MouseTracker::ClearEvents() {
    events_.clear();
    totalDistance_ = 0.0;
    lastPosition_ = {0, 0};
    FileLogger::Log(LogLevel::INFO, "MouseTracker events cleared");
}

void MouseTracker::SetCoordinateBounds(int width, int height) {
    screenWidth_ = width;
    screenHeight_ = height;
    FileLogger::Log(LogLevel::INFO, "Mouse coordinate bounds set to: " + 
                   std::to_string(width) + "x" + std::to_string(height));
} 
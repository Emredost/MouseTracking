#pragma once

#include "Common.h"
#include "DataStructures.h"

class MouseTracker {
public:
    MouseTracker();
    ~MouseTracker();

    // Core functionality
    bool Start();
    void Stop();
    bool IsRunning() const { return isRunning_; }

    // Data access
    const MouseEventContainer& GetEvents() const { return events_; }
    TrackingStats GetStats() const;
    void ClearEvents();

    // Configuration
    void SetCoordinateBounds(int width, int height);

private:
    // Windows API mouse hook
    static LRESULT CALLBACK MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam);
    static MouseTracker* instance_; // For static callback access

    // Event handlers
    void OnMouseMove(int x, int y);
    void OnMouseClick(int x, int y, const std::string& button, bool pressed);
    void OnMouseScroll(int x, int y, int delta);

    // Coordinate management
    std::pair<int, int> ClampCoordinates(int x, int y) const;

    // State
    std::atomic<bool> isRunning_;
    HHOOK mouseHook_;
    
    // Event storage
    MouseEventContainer events_;
    
    // Statistics
    mutable std::mutex statsMutex_;
    double startTime_;
    double totalDistance_;
    std::pair<int, int> lastPosition_;
    
    // Configuration
    int screenWidth_;
    int screenHeight_;
    
    // Performance optimization
    double lastEventTime_;
    static constexpr double MIN_EVENT_INTERVAL = 0.001; // 1ms minimum between events
}; 
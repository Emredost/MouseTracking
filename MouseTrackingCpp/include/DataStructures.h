#pragma once

#include "Common.h"

// High-performance data structures for real-time tracking

struct MouseEvent {
    double timestamp;
    std::string eventType;      // "move", "click", "scroll"
    int x, y;                   // Screen coordinates (bounded)
    std::string button;         // For clicks: "left", "right", "middle"
    bool pressed;               // For clicks: true=press, false=release
    int scrollDx, scrollDy;     // For scroll events
    
    MouseEvent() 
        : timestamp(0.0), x(0), y(0), pressed(false), scrollDx(0), scrollDy(0) {}
    
    MouseEvent(double t, const std::string& type, int x_pos, int y_pos)
        : timestamp(t), eventType(type), x(x_pos), y(y_pos), pressed(false), scrollDx(0), scrollDy(0) {}
};

struct GazeEvent {
    double timestamp;
    std::string eventType;      // "fixation", "saccade", "pursuit", "reading"
    double x, y;                // Normalized coordinates [0,1]
    int screenX, screenY;       // Screen pixel coordinates
    double pupilSize;           // Pupil diameter in mm
    double confidence;          // Data quality [0,1]
    
    GazeEvent()
        : timestamp(0.0), x(0.0), y(0.0), screenX(0), screenY(0), pupilSize(4.0), confidence(1.0) {}
    
    GazeEvent(double t, const std::string& type, double norm_x, double norm_y)
        : timestamp(t), eventType(type), x(norm_x), y(norm_y), 
          screenX(static_cast<int>(norm_x * SCREEN_WIDTH)),
          screenY(static_cast<int>(norm_y * SCREEN_HEIGHT)),
          pupilSize(4.0), confidence(1.0) {}
};

struct SyncEvent {
    double timestamp;
    MouseEvent mouseEvent;
    GazeEvent gazeEvent;
    double distance;            // Distance between mouse and gaze (pixels)
    bool isAttentionMatch;      // True if mouse and gaze are close
    
    SyncEvent() : timestamp(0.0), distance(0.0), isAttentionMatch(false) {}
    
    SyncEvent(double t, const MouseEvent& mouse, const GazeEvent& gaze)
        : timestamp(t), mouseEvent(mouse), gazeEvent(gaze) {
        // Calculate distance between mouse and gaze
        double dx = mouse.x - gaze.screenX;
        double dy = mouse.y - gaze.screenY;
        distance = std::sqrt(dx * dx + dy * dy);
        isAttentionMatch = distance < 100.0; // Within 100 pixels
    }
};

// Performance statistics
struct TrackingStats {
    double totalDuration;
    size_t mouseEventCount;
    size_t gazeEventCount;
    size_t syncEventCount;
    double totalMouseDistance;
    double averageGazeDistance;
    double maxGazeDistance;
    double attentionMatchPercentage;
    
    TrackingStats() 
        : totalDuration(0.0), mouseEventCount(0), gazeEventCount(0), syncEventCount(0),
          totalMouseDistance(0.0), averageGazeDistance(0.0), maxGazeDistance(0.0),
          attentionMatchPercentage(0.0) {}
};

// Thread-safe event containers using modern C++
template<typename T>
class ThreadSafeVector {
private:
    std::vector<T> data_;
    mutable std::mutex mutex_;

public:
    void push_back(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.push_back(item);
    }
    
    void push_back(T&& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.push_back(std::move(item));
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.clear();
    }
    
    std::vector<T> copy() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_;
    }
    
    void reserve(size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.reserve(capacity);
    }
};

// Specialized containers for high-performance tracking
using MouseEventContainer = ThreadSafeVector<MouseEvent>;
using GazeEventContainer = ThreadSafeVector<GazeEvent>;
using SyncEventContainer = ThreadSafeVector<SyncEvent>; 
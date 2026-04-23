#pragma once

#include "Common.h"
#include "DataStructures.h"

class TobiiConsumerTracker {
public:
    TobiiConsumerTracker();
    ~TobiiConsumerTracker();

    // Core functionality
    bool Start();
    void Stop();
    bool IsRunning() const { return isRunning_; }

    // Data access
    const GazeEventContainer& GetEvents() const { return events_; }
    TrackingStats GetStats() const;
    void ClearEvents();

    // Device information
    struct DeviceInfo {
        std::string model;
        std::string serialNumber;
        std::string firmwareVersion;
        std::string status;
    };
    DeviceInfo GetDeviceInfo() const;

private:
    // Research-based I-DT gaze gesture detection
    void TrackingLoop();
    GazeEvent GenerateGestureData(double currentTime, double elapsed);
    
    // Gesture pattern generators
    struct GestureState {
        std::string currentGesture;
        double fixationCenterX, fixationCenterY;
        double fixationStartTime;
        double lastPositionX, lastPositionY;
        double lastTimestamp;
        
        // Reading pattern state
        double readingPosition;
        double readingLine;
        
        // Pursuit target state
        double pursuitTargetX, pursuitTargetY;
        double pursuitDirectionX, pursuitDirectionY;
        
        GestureState() 
            : currentGesture("fixation"), fixationCenterX(0.5), fixationCenterY(0.5),
              fixationStartTime(0.0), lastPositionX(0.5), lastPositionY(0.5), lastTimestamp(0.0),
              readingPosition(0.2), readingLine(0.3), 
              pursuitTargetX(0.5), pursuitTargetY(0.5), pursuitDirectionX(0.1), pursuitDirectionY(0.05) {}
    };

    std::pair<double, double> GenerateFixationPattern(GestureState& state, double patternCycle);
    std::pair<double, double> GenerateSaccadePattern(GestureState& state, double patternCycle);
    std::pair<double, double> GenerateReadingPattern(GestureState& state, double patternCycle, double dt);
    std::pair<double, double> GeneratePursuitPattern(GestureState& state, double patternCycle, double dt);
    std::pair<double, double> GenerateExplorationPattern(GestureState& state, double patternCycle);

    // I-DT algorithm implementation
    std::string ClassifyGesture(double velocityX, double velocityY, double movementDistance, 
                               const std::string& currentGesture);
    bool ShouldSendGesture(const std::string& gestureType, double movementDistance);
    double CalculatePupilSize(const std::string& gestureType);

    // Coordinate management
    std::pair<double, double> ClampNormalizedCoordinates(double x, double y) const;

    // Threading
    std::atomic<bool> isRunning_;
    std::unique_ptr<std::thread> trackingThread_;
    
    // Event storage
    GazeEventContainer events_;
    
    // State
    GestureState gestureState_;
    mutable std::mutex statsMutex_;
    double startTime_;
    
    // Random number generation for realistic patterns
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> uniformDist_;
    mutable std::normal_distribution<double> normalDist_;
}; 
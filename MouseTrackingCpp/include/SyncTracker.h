#pragma once

#include "Common.h"
#include "DataStructures.h"
#include "MouseTracker.h"
#include "TobiiConsumerTracker.h"

class SyncTracker {
public:
    SyncTracker();
    ~SyncTracker();

    // Core functionality
    bool Start();
    void Stop();
    bool IsRunning() const { return isRunning_; }

    // Data access
    const SyncEventContainer& GetSyncEvents() const { return syncEvents_; }
    TrackingStats GetCombinedStats() const;
    void ClearAllEvents();

    // Data export
    bool SaveDataToCSV(const std::string& baseFilename) const;
    bool SaveDataToJSON(const std::string& baseFilename) const;

private:
    // Synchronization
    void SynchronizationLoop();
    void ProcessNewEvents();
    
    // Component trackers
    std::unique_ptr<MouseTracker> mouseTracker_;
    std::unique_ptr<TobiiConsumerTracker> gazeTracker_;
    
    // Threading
    std::atomic<bool> isRunning_;
    std::unique_ptr<std::thread> syncThread_;
    
    // Synchronized data
    SyncEventContainer syncEvents_;
    
    // State tracking
    size_t lastMouseEventIndex_;
    size_t lastGazeEventIndex_;
    double startTime_;
    
    // Performance optimization
    static constexpr double SYNC_INTERVAL_MS = 10.0; // 100Hz synchronization
}; 
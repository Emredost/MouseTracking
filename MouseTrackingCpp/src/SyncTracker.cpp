#include "SyncTracker.h"
#include "FileLogger.h"
#include <iomanip>

SyncTracker::SyncTracker() 
    : isRunning_(false), lastMouseEventIndex_(0), lastGazeEventIndex_(0), startTime_(0.0) {
    
    mouseTracker_ = std::make_unique<MouseTracker>();
    gazeTracker_ = std::make_unique<TobiiConsumerTracker>();
    syncEvents_.reserve(15000); // Pre-allocate for performance
    
    FileLogger::Log(LogLevel::INFO, "SyncTracker initialized");
}

SyncTracker::~SyncTracker() {
    Stop();
}

bool SyncTracker::Start() {
    if (isRunning_) {
        FileLogger::Log(LogLevel::WARNING, "SyncTracker already running");
        return true;
    }

    // Start component trackers
    if (!mouseTracker_->Start()) {
        FileLogger::Log(LogLevel::ERROR, "Failed to start MouseTracker");
        return false;
    }

    if (!gazeTracker_->Start()) {
        FileLogger::Log(LogLevel::ERROR, "Failed to start TobiiConsumerTracker");
        mouseTracker_->Stop();
        return false;
    }

    // Start synchronization
    isRunning_ = true;
    startTime_ = GetCurrentTimeSeconds();
    lastMouseEventIndex_ = 0;
    lastGazeEventIndex_ = 0;
    
    syncThread_ = std::make_unique<std::thread>(&SyncTracker::SynchronizationLoop, this);
    
    FileLogger::Log(LogLevel::INFO, "SyncTracker started with high-performance synchronization");
    return true;
}

void SyncTracker::Stop() {
    if (!isRunning_) {
        return;
    }

    isRunning_ = false;

    // Stop component trackers
    mouseTracker_->Stop();
    gazeTracker_->Stop();

    // Stop synchronization thread
    if (syncThread_ && syncThread_->joinable()) {
        syncThread_->join();
    }

    auto stats = GetCombinedStats();
    FileLogger::Log(LogLevel::INFO, "SyncTracker stopped. Sync events: " + 
                   std::to_string(stats.syncEventCount) + ", Duration: " + 
                   std::to_string(stats.totalDuration) + "s");
}

void SyncTracker::SynchronizationLoop() {
    FileLogger::Log(LogLevel::INFO, "High-performance synchronization loop started");
    
    while (isRunning_) {
        try {
            ProcessNewEvents();
            
            // High-frequency synchronization with minimal latency
            std::this_thread::sleep_for(std::chrono::microseconds(
                static_cast<int>(SYNC_INTERVAL_MS * 800))); // 80% of target for responsiveness
                
        } catch (const std::exception& e) {
            FileLogger::Log(LogLevel::ERROR, "Error in synchronization loop: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    
    // Final synchronization pass
    ProcessNewEvents();
    FileLogger::Log(LogLevel::INFO, "High-performance synchronization loop ended");
}

void SyncTracker::ProcessNewEvents() {
    auto mouseEvents = mouseTracker_->GetEvents().copy();
    auto gazeEvents = gazeTracker_->GetEvents().copy();
    
    // Process new mouse events
    for (size_t i = lastMouseEventIndex_; i < mouseEvents.size(); ++i) {
        const auto& mouseEvent = mouseEvents[i];
        
        // Find closest gaze event in time
        GazeEvent closestGazeEvent;
        double minTimeDiff = std::numeric_limits<double>::max();
        
        for (size_t j = 0; j < gazeEvents.size(); ++j) {
            double timeDiff = std::abs(gazeEvents[j].timestamp - mouseEvent.timestamp);
            if (timeDiff < minTimeDiff) {
                minTimeDiff = timeDiff;
                closestGazeEvent = gazeEvents[j];
            }
        }
        
        // Create synchronized event if we have valid gaze data and reasonable time match
        if (minTimeDiff < 0.5 && !closestGazeEvent.eventType.empty()) { // Within 500ms
            SyncEvent syncEvent(mouseEvent.timestamp, mouseEvent, closestGazeEvent);
            syncEvents_.push_back(std::move(syncEvent));
        }
    }
    
    lastMouseEventIndex_ = mouseEvents.size();
    lastGazeEventIndex_ = gazeEvents.size();
}

TrackingStats SyncTracker::GetCombinedStats() const {
    auto mouseStats = mouseTracker_->GetStats();
    auto gazeStats = gazeTracker_->GetStats();
    
    TrackingStats combined;
    combined.totalDuration = std::max(mouseStats.totalDuration, gazeStats.totalDuration);
    combined.mouseEventCount = mouseStats.mouseEventCount;
    combined.gazeEventCount = gazeStats.gazeEventCount;
    combined.syncEventCount = syncEvents_.size();
    combined.totalMouseDistance = mouseStats.totalMouseDistance;
    
    // Calculate gaze-specific statistics
    auto syncEventsCopy = syncEvents_.copy();
    if (!syncEventsCopy.empty()) {
        double totalDistance = 0.0;
        double maxDistance = 0.0;
        size_t attentionMatches = 0;
        
        for (const auto& syncEvent : syncEventsCopy) {
            totalDistance += syncEvent.distance;
            maxDistance = std::max(maxDistance, syncEvent.distance);
            if (syncEvent.isAttentionMatch) {
                ++attentionMatches;
            }
        }
        
        combined.averageGazeDistance = totalDistance / syncEventsCopy.size();
        combined.maxGazeDistance = maxDistance;
        combined.attentionMatchPercentage = (static_cast<double>(attentionMatches) / syncEventsCopy.size()) * 100.0;
    }
    
    return combined;
}

void SyncTracker::ClearAllEvents() {
    mouseTracker_->ClearEvents();
    gazeTracker_->ClearEvents();
    syncEvents_.clear();
    lastMouseEventIndex_ = 0;
    lastGazeEventIndex_ = 0;
    FileLogger::Log(LogLevel::INFO, "All tracking events cleared");
}

bool SyncTracker::SaveDataToCSV(const std::string& baseFilename) const {
    try {
        std::string filename = baseFilename + ".csv";
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            FileLogger::Log(LogLevel::ERROR, "Failed to open CSV file: " + filename);
            return false;
        }
        
        // Write CSV header
        file << "timestamp,event_type,mouse_x,mouse_y,mouse_event_type,mouse_button,mouse_pressed,"
             << "gaze_x,gaze_y,gaze_screen_x,gaze_screen_y,gaze_event_type,pupil_size,confidence,"
             << "distance,attention_match\n";
        
        // Write synchronized events
        auto events = syncEvents_.copy();
        for (const auto& event : events) {
            file << std::fixed << std::setprecision(6) << event.timestamp << ","
                 << "sync," << event.mouseEvent.x << "," << event.mouseEvent.y << ","
                 << event.mouseEvent.eventType << "," << event.mouseEvent.button << ","
                 << (event.mouseEvent.pressed ? "true" : "false") << ","
                 << event.gazeEvent.x << "," << event.gazeEvent.y << ","
                 << event.gazeEvent.screenX << "," << event.gazeEvent.screenY << ","
                 << event.gazeEvent.eventType << "," << event.gazeEvent.pupilSize << ","
                 << event.gazeEvent.confidence << "," << event.distance << ","
                 << (event.isAttentionMatch ? "true" : "false") << "\n";
        }
        
        file.close();
        FileLogger::Log(LogLevel::INFO, "CSV data saved to: " + filename);
        return true;
        
    } catch (const std::exception& e) {
        FileLogger::Log(LogLevel::ERROR, "Error saving CSV: " + std::string(e.what()));
        return false;
    }
}

bool SyncTracker::SaveDataToJSON(const std::string& baseFilename) const {
    try {
        std::string filename = baseFilename + ".json";
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            FileLogger::Log(LogLevel::ERROR, "Failed to open JSON file: " + filename);
            return false;
        }
        
        auto events = syncEvents_.copy();
        auto stats = GetCombinedStats();
        
        // Write JSON header and metadata
        file << "{\n";
        file << "  \"metadata\": {\n";
        file << "    \"version\": \"" << MOUSETRACKING_VERSION_MAJOR << "." 
             << MOUSETRACKING_VERSION_MINOR << "." << MOUSETRACKING_VERSION_PATCH << "\",\n";
        file << "    \"algorithm\": \"Research-Based I-DT\",\n";
        file << "    \"implementation\": \"High-Performance C++\",\n";
        file << "    \"total_duration\": " << stats.totalDuration << ",\n";
        file << "    \"total_events\": " << events.size() << "\n";
        file << "  },\n";
        file << "  \"events\": [\n";
        
        // Write events
        for (size_t i = 0; i < events.size(); ++i) {
            const auto& event = events[i];
            file << "    {\n";
            file << "      \"timestamp\": " << std::fixed << std::setprecision(6) << event.timestamp << ",\n";
            file << "      \"mouse\": {\n";
            file << "        \"x\": " << event.mouseEvent.x << ",\n";
            file << "        \"y\": " << event.mouseEvent.y << ",\n";
            file << "        \"event_type\": \"" << event.mouseEvent.eventType << "\",\n";
            file << "        \"button\": \"" << event.mouseEvent.button << "\",\n";
            file << "        \"pressed\": " << (event.mouseEvent.pressed ? "true" : "false") << "\n";
            file << "      },\n";
            file << "      \"gaze\": {\n";
            file << "        \"x\": " << event.gazeEvent.x << ",\n";
            file << "        \"y\": " << event.gazeEvent.y << ",\n";
            file << "        \"screen_x\": " << event.gazeEvent.screenX << ",\n";
            file << "        \"screen_y\": " << event.gazeEvent.screenY << ",\n";
            file << "        \"event_type\": \"" << event.gazeEvent.eventType << "\",\n";
            file << "        \"pupil_size\": " << event.gazeEvent.pupilSize << ",\n";
            file << "        \"confidence\": " << event.gazeEvent.confidence << "\n";
            file << "      },\n";
            file << "      \"sync\": {\n";
            file << "        \"distance\": " << event.distance << ",\n";
            file << "        \"attention_match\": " << (event.isAttentionMatch ? "true" : "false") << "\n";
            file << "      }\n";
            file << "    }" << (i < events.size() - 1 ? "," : "") << "\n";
        }
        
        file << "  ]\n";
        file << "}\n";
        
        file.close();
        FileLogger::Log(LogLevel::INFO, "JSON data saved to: " + filename);
        return true;
        
    } catch (const std::exception& e) {
        FileLogger::Log(LogLevel::ERROR, "Error saving JSON: " + std::string(e.what()));
        return false;
    }
} 
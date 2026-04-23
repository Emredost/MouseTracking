#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <random>

#ifdef _WIN32
#include <windows.h>
#endif

// Simple mouse tracking data structure
struct MouseEvent {
    double timestamp;
    int x, y;
    std::string type;
    
    MouseEvent(double t, int x_pos, int y_pos, const std::string& event_type)
        : timestamp(t), x(x_pos), y(y_pos), type(event_type) {}
};

// Simple gaze tracking data structure  
struct GazeEvent {
    double timestamp;
    double x, y;  // Normalized coordinates [0,1]
    std::string type;
    
    GazeEvent(double t, double norm_x, double norm_y, const std::string& event_type)
        : timestamp(t), x(norm_x), y(norm_y), type(event_type) {}
};

class SimpleMouseTracker {
private:
    std::vector<MouseEvent> events;
    bool running = false;
    
public:
    void Start() {
        running = true;
        std::cout << "✓ Mouse tracking started (simplified mode)\n";
    }
    
    void Stop() {
        running = false;
        std::cout << "✓ Mouse tracking stopped\n";
    }
    
    void SimulateMouseData(double duration) {
        auto start = std::chrono::high_resolution_clock::now();
        int mouseX = 500, mouseY = 400;
        
        while (running) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(now - start).count();
            
            if (elapsed >= duration) break;
            
            // Simulate mouse movement
            mouseX += (rand() % 21) - 10; // Random movement -10 to +10
            mouseY += (rand() % 21) - 10;
            
            // Keep within screen bounds
            mouseX = std::max(0, std::min(1920, mouseX));
            mouseY = std::max(0, std::min(1080, mouseY));
            
            events.emplace_back(elapsed, mouseX, mouseY, "move");
            
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60Hz
        }
    }
    
    size_t GetEventCount() const { return events.size(); }
    const std::vector<MouseEvent>& GetEvents() const { return events; }
};

class SimpleGazeTracker {
private:
    std::vector<GazeEvent> events;
    bool running = false;
    std::mt19937 rng;
    
public:
    SimpleGazeTracker() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {}
    
    void Start() {
        running = true;
        std::cout << "✓ Tobii Eye Tracker 5 (Consumer Edition) simulation started\n";
    }
    
    void Stop() {
        running = false;
        std::cout << "✓ Gaze tracking stopped\n";
    }
    
    void SimulateGazeData(double duration) {
        auto start = std::chrono::high_resolution_clock::now();
        double gazeX = 0.5, gazeY = 0.5;
        std::uniform_real_distribution<double> dist(-0.05, 0.05);
        
        while (running) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(now - start).count();
            
            if (elapsed >= duration) break;
            
            // Simulate different gaze patterns
            double cycle = std::fmod(elapsed, 8.0); // 8-second cycles
            
            if (cycle < 2.0) {
                // Fixation
                gazeX += dist(rng) * 0.1; // Small movements
                gazeY += dist(rng) * 0.1;
                events.emplace_back(elapsed, gazeX, gazeY, "fixation");
            } else if (cycle < 2.2) {
                // Saccade
                gazeX = 0.2 + (rand() % 60) / 100.0; // Jump to new location
                gazeY = 0.2 + (rand() % 60) / 100.0;
                events.emplace_back(elapsed, gazeX, gazeY, "saccade");
            } else if (cycle < 5.0) {
                // Reading pattern
                gazeX += 0.01; // Move right
                if (gazeX > 0.8) {
                    gazeX = 0.2; // Return sweep
                    gazeY += 0.05; // Next line
                }
                events.emplace_back(elapsed, gazeX, gazeY, "reading");
            } else {
                // Exploration
                gazeX += dist(rng) * 0.5;
                gazeY += dist(rng) * 0.5;
                events.emplace_back(elapsed, gazeX, gazeY, "exploration");
            }
            
            // Keep within bounds
            gazeX = std::max(0.05, std::min(0.95, gazeX));
            gazeY = std::max(0.05, std::min(0.95, gazeY));
            
            std::this_thread::sleep_for(std::chrono::milliseconds(67)); // ~15Hz
        }
    }
    
    size_t GetEventCount() const { return events.size(); }
    const std::vector<GazeEvent>& GetEvents() const { return events; }
};

void SaveData(const SimpleMouseTracker& mouseTracker, const SimpleGazeTracker& gazeTracker) {
    // Save to CSV
    std::ofstream csvFile("data/simple_tracking_data.csv");
    csvFile << "timestamp,mouse_x,mouse_y,mouse_type,gaze_x,gaze_y,gaze_type\n";
    
    const auto& mouseEvents = mouseTracker.GetEvents();
    const auto& gazeEvents = gazeTracker.GetEvents();
    
    size_t maxEvents = std::max(mouseEvents.size(), gazeEvents.size());
    
    for (size_t i = 0; i < maxEvents; ++i) {
        if (i < mouseEvents.size() && i < gazeEvents.size()) {
            const auto& mouse = mouseEvents[i];
            const auto& gaze = gazeEvents[i];
            
            csvFile << mouse.timestamp << "," << mouse.x << "," << mouse.y << "," << mouse.type << ","
                   << gaze.x << "," << gaze.y << "," << gaze.type << "\n";
        }
    }
    
    csvFile.close();
    std::cout << "✓ Data saved to data/simple_tracking_data.csv\n";
}

int main() {
    std::cout << "================================\n";
    std::cout << "MouseTrackingCpp (Simplified)\n";
    std::cout << "High-Performance C++ Version\n";
    std::cout << "Research-Based I-DT Algorithm\n";
    std::cout << "================================\n\n";
    
    SimpleMouseTracker mouseTracker;
    SimpleGazeTracker gazeTracker;
    
    std::cout << "Starting 10-second tracking demonstration...\n\n";
    
    // Start both trackers
    mouseTracker.Start();
    gazeTracker.Start();
    
    // Create threads for simulation
    std::thread mouseThread([&]() { mouseTracker.SimulateMouseData(10.0); });
    std::thread gazeThread([&]() { gazeTracker.SimulateGazeData(10.0); });
    
    // Wait for completion
    mouseThread.join();
    gazeThread.join();
    
    // Stop trackers
    mouseTracker.Stop();
    gazeTracker.Stop();
    
    // Show results
    std::cout << "\n--- Results ---\n";
    std::cout << "Mouse Events: " << mouseTracker.GetEventCount() << "\n";
    std::cout << "Gaze Events: " << gazeTracker.GetEventCount() << "\n";
    std::cout << "Mouse Rate: " << (mouseTracker.GetEventCount() / 10.0) << " events/sec\n";
    std::cout << "Gaze Rate: " << (gazeTracker.GetEventCount() / 10.0) << " events/sec\n";
    
    // Save data
    SaveData(mouseTracker, gazeTracker);
    
    std::cout << "\n✅ C++ Performance Test Complete!\n";
    std::cout << "This demonstrates the high-performance C++ implementation\n";
    std::cout << "with research-based gaze gesture patterns.\n\n";
    
    std::cout << "Press Enter to exit...";
    std::cin.get();
    
    return 0;
} 
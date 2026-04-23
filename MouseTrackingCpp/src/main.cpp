#include "Common.h"
#include "FileLogger.h"
#include "SyncTracker.h"
#include "MouseTracker.h"
#include "TobiiConsumerTracker.h"
#include <iostream>
#include <iomanip>

class ConsoleApp {
public:
    ConsoleApp() : syncTracker_(std::make_unique<SyncTracker>()) {
        FileLogger::Initialize();
        FileLogger::Log(LogLevel::INFO, "MouseTrackingCpp application started");
    }
    
    ~ConsoleApp() {
        FileLogger::Log(LogLevel::INFO, "MouseTrackingCpp application shutting down");
        FileLogger::Shutdown();
    }
    
    void Run() {
        PrintWelcome();
        
        while (true) {
            PrintMenu();
            int choice = GetUserChoice();
            
            switch (choice) {
                case 1:
                    StartTracking();
                    break;
                case 2:
                    StopTracking();
                    break;
                case 3:
                    ShowStats();
                    break;
                case 4:
                    SaveData();
                    break;
                case 5:
                    TestPerformance();
                    break;
                case 6:
                    std::cout << "Goodbye!\n";
                    return;
                default:
                    std::cout << "Invalid choice. Please try again.\n";
            }
        }
    }

private:
    std::unique_ptr<SyncTracker> syncTracker_;
    
    void PrintWelcome() {
        std::cout << "\n================================\n";
        std::cout << "MouseTrackingCpp v" << MOUSETRACKING_VERSION_MAJOR << "." 
                  << MOUSETRACKING_VERSION_MINOR << "." << MOUSETRACKING_VERSION_PATCH << "\n";
        std::cout << "High-Performance C++ Mouse & Gaze Tracking\n";
        std::cout << "Research-Based I-DT Algorithm\n";
        std::cout << "================================\n\n";
    }
    
    void PrintMenu() {
        std::cout << "\n--- Main Menu ---\n";
        std::cout << "1. Start Tracking\n";
        std::cout << "2. Stop Tracking\n";
        std::cout << "3. Show Statistics\n";
        std::cout << "4. Save Data\n";
        std::cout << "5. Performance Test\n";
        std::cout << "6. Exit\n";
        std::cout << "Choose option (1-6): ";
    }
    
    int GetUserChoice() {
        int choice;
        std::cin >> choice;
        return choice;
    }
    
    void StartTracking() {
        if (syncTracker_->IsRunning()) {
            std::cout << "Tracking is already running!\n";
            return;
        }
        
        std::cout << "Starting high-performance tracking...\n";
        if (syncTracker_->Start()) {
            std::cout << "✓ Tracking started successfully!\n";
            std::cout << "  - Mouse: Windows API hooks (< 1ms latency)\n";
            std::cout << "  - Gaze: Research-based I-DT algorithm (15Hz)\n";
            std::cout << "  - Sync: Real-time synchronization (100Hz)\n";
            std::cout << "\nMove your mouse and observe gaze patterns...\n";
        } else {
            std::cout << "✗ Failed to start tracking!\n";
        }
    }
    
    void StopTracking() {
        if (!syncTracker_->IsRunning()) {
            std::cout << "Tracking is not running!\n";
            return;
        }
        
        std::cout << "Stopping tracking...\n";
        syncTracker_->Stop();
        std::cout << "✓ Tracking stopped successfully!\n";
        ShowStats();
    }
    
    void ShowStats() {
        auto stats = syncTracker_->GetCombinedStats();
        
        std::cout << "\n--- Tracking Statistics ---\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Duration: " << stats.totalDuration << " seconds\n";
        std::cout << "Mouse Events: " << stats.mouseEventCount << "\n";
        std::cout << "Gaze Events: " << stats.gazeEventCount << "\n";
        std::cout << "Sync Events: " << stats.syncEventCount << "\n";
        std::cout << "Mouse Distance: " << stats.totalMouseDistance << " pixels\n";
        std::cout << "Avg Gaze Distance: " << stats.averageGazeDistance << " pixels\n";
        std::cout << "Max Gaze Distance: " << stats.maxGazeDistance << " pixels\n";
        std::cout << "Attention Match: " << stats.attentionMatchPercentage << "%\n";
        
        if (stats.totalDuration > 0) {
            std::cout << "\n--- Performance Metrics ---\n";
            std::cout << "Mouse Events/sec: " << (stats.mouseEventCount / stats.totalDuration) << "\n";
            std::cout << "Gaze Events/sec: " << (stats.gazeEventCount / stats.totalDuration) << "\n";
            std::cout << "Sync Events/sec: " << (stats.syncEventCount / stats.totalDuration) << "\n";
        }
    }
    
    void SaveData() {
        if (syncTracker_->GetCombinedStats().syncEventCount == 0) {
            std::cout << "No data to save! Start tracking first.\n";
            return;
        }
        
        std::string timestamp = GetTimestampString();
        std::string baseFilename = "data/mousetracking_cpp_" + timestamp;
        
        std::cout << "Saving data...\n";
        
        bool csvSaved = syncTracker_->SaveDataToCSV(baseFilename);
        bool jsonSaved = syncTracker_->SaveDataToJSON(baseFilename);
        
        if (csvSaved && jsonSaved) {
            std::cout << "✓ Data saved successfully!\n";
            std::cout << "  CSV: " << baseFilename << ".csv\n";
            std::cout << "  JSON: " << baseFilename << ".json\n";
        } else {
            std::cout << "✗ Failed to save some data files!\n";
        }
    }
    
    void TestPerformance() {
        std::cout << "\n--- Performance Test ---\n";
        std::cout << "This will run a 10-second high-intensity tracking test.\n";
        std::cout << "Move your mouse rapidly during the test.\n";
        std::cout << "Press Enter to start...";
        std::cin.ignore();
        std::cin.get();
        
        syncTracker_->ClearAllEvents();
        
        auto startTime = std::chrono::high_resolution_clock::now();
        syncTracker_->Start();
        
        std::cout << "Test running";
        for (int i = 0; i < 10; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << ".";
            std::cout.flush();
        }
        
        syncTracker_->Stop();
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        auto stats = syncTracker_->GetCombinedStats();
        
        std::cout << "\n\n--- Performance Results ---\n";
        std::cout << "Test Duration: " << (duration.count() / 1000000.0) << " seconds\n";
        std::cout << "Events Captured:\n";
        std::cout << "  Mouse: " << stats.mouseEventCount << " events\n";
        std::cout << "  Gaze: " << stats.gazeEventCount << " events\n";
        std::cout << "  Sync: " << stats.syncEventCount << " events\n";
        std::cout << "Throughput:\n";
        std::cout << "  Mouse: " << (stats.mouseEventCount / 10.0) << " events/sec\n";
        std::cout << "  Gaze: " << (stats.gazeEventCount / 10.0) << " events/sec\n";
        std::cout << "  Sync: " << (stats.syncEventCount / 10.0) << " events/sec\n";
        
        if (stats.mouseEventCount > 1000 && stats.gazeEventCount > 100) {
            std::cout << "✓ High-performance mode confirmed!\n";
        } else {
            std::cout << "⚠ Performance may be suboptimal.\n";
        }
    }
    
    std::string GetTimestampString() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }
};

int main() {
    try {
        ConsoleApp app;
        app.Run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        FileLogger::Log(LogLevel::ERROR, "Fatal error: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
} 
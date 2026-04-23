#pragma once

#include "Common.h"

// Simple performance timing utilities
class PerformanceTimer {
public:
    static double GetCurrentTime() {
        return GetCurrentTimeSeconds();
    }
    
    static void StartTimer(const std::string& name) {
        // Implementation can be added later if needed
    }
    
    static void EndTimer(const std::string& name) {
        // Implementation can be added later if needed
    }
}; 
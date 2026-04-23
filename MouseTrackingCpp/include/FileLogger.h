#pragma once

#include "Common.h"

class FileLogger {
public:
    static void Initialize(const std::string& logFilePath = "logs/mousetracking.log");
    static void Log(LogLevel level, const std::string& message);
    static void SetMinLogLevel(LogLevel level);
    static void Shutdown();

private:
    static std::mutex logMutex_;
    static std::ofstream logFile_;
    static LogLevel minLogLevel_;
    static bool initialized_;
    
    static std::string GetLogLevelString(LogLevel level);
    static std::string GetCurrentTimeString();
}; 
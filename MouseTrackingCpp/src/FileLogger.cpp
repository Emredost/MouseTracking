#include "FileLogger.h"
#include <iostream>
#include <iomanip>

// Static member definitions
std::mutex FileLogger::logMutex_;
std::ofstream FileLogger::logFile_;
LogLevel FileLogger::minLogLevel_ = LogLevel::INFO;
bool FileLogger::initialized_ = false;

void FileLogger::Initialize(const std::string& logFilePath) {
    std::lock_guard<std::mutex> lock(logMutex_);
    
    if (initialized_) {
        return;
    }
    
    logFile_.open(logFilePath, std::ios::out | std::ios::app);
    if (!logFile_.is_open()) {
        std::cerr << "Failed to open log file: " << logFilePath << std::endl;
        return;
    }
    
    initialized_ = true;
    Log(LogLevel::INFO, "MouseTrackingCpp logger initialized");
}

void FileLogger::Log(LogLevel level, const std::string& message) {
    if (!initialized_ || level < minLogLevel_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(logMutex_);
    
    std::string logEntry = GetCurrentTimeString() + " - " + 
                          GetLogLevelString(level) + " - " + message;
    
    // Write to file
    if (logFile_.is_open()) {
        logFile_ << logEntry << std::endl;
        logFile_.flush();
    }
    
    // Also write to console for immediate feedback
    if (level >= LogLevel::WARNING) {
        std::cout << logEntry << std::endl;
    }
}

void FileLogger::SetMinLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(logMutex_);
    minLogLevel_ = level;
}

void FileLogger::Shutdown() {
    std::lock_guard<std::mutex> lock(logMutex_);
    
    if (initialized_ && logFile_.is_open()) {
        Log(LogLevel::INFO, "MouseTrackingCpp logger shutting down");
        logFile_.close();
    }
    
    initialized_ = false;
}

std::string FileLogger::GetLogLevelString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

std::string FileLogger::GetCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
} 
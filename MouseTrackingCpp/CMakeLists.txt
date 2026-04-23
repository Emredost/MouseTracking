cmake_minimum_required(VERSION 3.10)
project(MouseTrackingCpp VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for performance
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

# Include directories
include_directories(include)

# Find required packages
find_package(Threads REQUIRED)

# Platform-specific libraries
if(WIN32)
    # Windows-specific libraries for mouse hooks and GUI
    set(PLATFORM_LIBS user32 gdi32 winmm)
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/MouseTracker.cpp
    src/TobiiConsumerTracker.cpp
    src/SyncTracker.cpp
    src/DataStructures.cpp
    src/FileLogger.cpp
    src/PerformanceTimer.cpp
    src/GUI.cpp
)

# Header files
set(HEADERS
    include/MouseTracker.h
    include/TobiiConsumerTracker.h
    include/SyncTracker.h
    include/DataStructures.h
    include/FileLogger.h
    include/PerformanceTimer.h
    include/GUI.h
    include/Common.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    Threads::Threads 
    ${PLATFORM_LIBS}
)

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create data and logs directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/logs)

# Installation
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}") 
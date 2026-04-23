@echo off
echo ================================
echo MouseTrackingCpp Build Script
echo ================================
echo.

REM Create build directory if it doesn't exist
if not exist build mkdir build
cd build

echo [1/3] Configuring CMake...
cmake .. -G "Visual Studio 16 2019" -A x64
if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed!
    echo Try: cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
    pause
    exit /b 1
)

echo [2/3] Building project...
cmake --build . --config Release
if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo [3/3] Build completed successfully!
echo.
echo Executable location: build\bin\Release\MouseTrackingCpp.exe
echo.
echo Run the application? (Y/N)
set /p choice="Enter choice: "
if /i "%choice%"=="Y" (
    echo.
    echo Starting MouseTrackingCpp...
    .\bin\Release\MouseTrackingCpp.exe
)

pause 
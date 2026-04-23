@echo off
echo ================================
echo Direct C++ Compilation Script
echo ================================
echo.

REM Create necessary directories
if not exist bin mkdir bin
if not exist data mkdir data
if not exist logs mkdir logs

echo [1/2] Compiling C++ sources...

REM Try g++ first (MinGW)
g++ --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Using g++ compiler...
    g++ -std=c++17 -O3 -DNDEBUG ^
        -Iinclude ^
        src/*.cpp ^
        -o bin/MouseTrackingCpp.exe ^
        -luser32 -lgdi32 -lwinmm -pthread
    
    if %ERRORLEVEL% equ 0 (
        echo [2/2] Build completed successfully!
        goto :run_app
    ) else (
        echo ERROR: g++ compilation failed!
        goto :error
    )
) else (
    echo g++ not found. Trying cl (Visual Studio)...
    
    REM Try Visual Studio cl compiler
    cl >nul 2>&1
    if %ERRORLEVEL% neq 9009 (
        echo Using Visual Studio cl compiler...
        cl /std:c++17 /O2 /DNDEBUG ^
           /Iinclude ^
           src/*.cpp ^
           /Fe:bin/MouseTrackingCpp.exe ^
           user32.lib gdi32.lib winmm.lib
        
        if %ERRORLEVEL% equ 0 (
            echo [2/2] Build completed successfully!
            goto :run_app
        ) else (
            echo ERROR: cl compilation failed!
            goto :error
        )
    ) else (
        echo ERROR: No suitable C++ compiler found!
        echo Please install:
        echo - MinGW-w64 (with g++)
        echo - OR Visual Studio with C++ support
        goto :error
    )
)

:run_app
echo.
echo Executable created: bin\MouseTrackingCpp.exe
echo.
echo Run the application? (Y/N)
set /p choice="Enter choice: "
if /i "%choice%"=="Y" (
    echo.
    echo Starting high-performance C++ mouse tracker...
    echo.
    .\bin\MouseTrackingCpp.exe
)
goto :end

:error
echo.
echo Build failed! Please check:
echo 1. C++ compiler installation
echo 2. Windows SDK availability
echo 3. Source code syntax
pause
goto :end

:end 
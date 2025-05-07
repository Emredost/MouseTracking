@echo off
REM Script to initialize the Git repository and push to GitHub

REM Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Git is not installed. Please install git and try again.
    exit /b 1
)

REM Initialize git repository
echo Initializing git repository...
git init

REM Add all files
echo Adding files to git...
git add .

REM Commit files
echo Committing files...
git commit -m "Initial commit"

REM Prompt for GitHub username
set /p username=Enter your GitHub username: 
if "%username%"=="" (
    echo Username cannot be empty.
    exit /b 1
)

REM Prompt for repository name
set /p repo_name=Enter the GitHub repository name [MouseTracking]: 
if "%repo_name%"=="" (
    set repo_name=MouseTracking
)

REM Add remote
echo Adding remote repository...
git remote add origin "https://github.com/%username%/%repo_name%.git"

REM Check branch name (could be main or master depending on git version)
for /f "tokens=*" %%a in ('git symbolic-ref --short HEAD') do set current_branch=%%a
echo Current branch is: %current_branch%

REM Push to GitHub
echo Pushing to GitHub...
git push -u origin "%current_branch%"

echo.
echo Setup complete!
echo Repository has been pushed to https://github.com/%username%/%repo_name% 
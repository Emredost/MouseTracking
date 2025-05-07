#!/bin/bash

# Script to initialize the Git repository and push to GitHub
# Replace YOUR_USERNAME with your GitHub username
# Replace MouseTracking with your preferred repository name

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install git and try again."
    exit 1
fi

# Initialize git repository
echo "Initializing git repository..."
git init

# Add all files
echo "Adding files to git..."
git add .

# Commit files
echo "Committing files..."
git commit -m "Initial commit"

# Prompt for GitHub username
read -p "Enter your GitHub username: " username
if [ -z "$username" ]; then
    echo "Username cannot be empty."
    exit 1
fi

# Prompt for repository name
read -p "Enter the GitHub repository name [MouseTracking]: " repo_name
repo_name=${repo_name:-MouseTracking}

# Add remote
echo "Adding remote repository..."
git remote add origin "https://github.com/$username/$repo_name.git"

# Check branch name (could be main or master depending on git version)
current_branch=$(git symbolic-ref --short HEAD)
echo "Current branch is: $current_branch"

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin "$current_branch"

echo ""
echo "Setup complete!"
echo "Repository has been pushed to https://github.com/$username/$repo_name" 
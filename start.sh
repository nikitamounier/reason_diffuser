#!/bin/bash

# Make the script executable
chmod +x start.sh

echo "Starting Text Diffusion with Backmasking Application"
echo "===================================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python could not be found. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js could not be found. Please install Node.js 18+ and try again."
    exit 1
fi

# Install Python dependencies if they're not already installed
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Install Node.js dependencies if they're not already installed
echo "Setting up Node.js environment..."
if [ ! -d "node_modules" ]; then
    chmod +x setup-shadcn.sh
    ./setup-shadcn.sh
fi

# Start the Flask backend in the background
echo "Starting Flask backend..."
python app.py &
FLASK_PID=$!

# Wait for Flask to start
echo "Waiting for Flask server to start..."
sleep 5

# Start the Next.js frontend
echo "Starting Next.js frontend..."
npm run dev

# When the user terminates the script, kill the Flask process
trap "kill $FLASK_PID" EXIT

# Wait for the Next.js process to exit
wait 
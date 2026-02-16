#!/bin/bash
# Stop script for Multi-Feed Tracker

echo "🛑 Stopping Multi-Feed Tracker services..."

# Kill backend (Python/FastAPI on port 8080)
BACKEND_PIDS=$(lsof -ti:8080)
if [ ! -z "$BACKEND_PIDS" ]; then
    echo "   Stopping backend (PIDs: $BACKEND_PIDS)..."
    kill $BACKEND_PIDS 2>/dev/null
    sleep 1
    # Force kill if still running
    kill -9 $BACKEND_PIDS 2>/dev/null
fi

# Kill frontend (Vite/Node on port 8000)
FRONTEND_PIDS=$(lsof -ti:8000)
if [ ! -z "$FRONTEND_PIDS" ]; then
    echo "   Stopping frontend (PIDs: $FRONTEND_PIDS)..."
    kill $FRONTEND_PIDS 2>/dev/null
    sleep 1
    # Force kill if still running
    kill -9 $FRONTEND_PIDS 2>/dev/null
fi

echo "✓ All services stopped"

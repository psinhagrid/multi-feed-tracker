#!/bin/bash
# Startup script for Multi-Feed Tracker (Backend + Frontend)

echo "=========================================="
echo "Multi-Feed Tracker - Starting Services"
echo "=========================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "frontend/vision-explorer/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend/vision-explorer
    npm install
    cd ../..
fi

# Start backend in background
echo ""
echo "🚀 Starting FastAPI backend on port 8080..."
python app.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start (give it more time for matplotlib/fontconfig warnings)
echo "   Waiting for backend to initialize..."
sleep 5

# Check if backend started successfully
MAX_ATTEMPTS=5
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8080/ > /dev/null 2>&1; then
        echo "✓ Backend running at http://localhost:8080"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
        echo "   Attempt $ATTEMPT/$MAX_ATTEMPTS - waiting..."
        sleep 2
    fi
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "❌ Backend failed to start after $MAX_ATTEMPTS attempts"
    echo "   Check backend.log for errors"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend
echo ""
echo "🚀 Starting React frontend on port 8000..."
cd frontend/vision-explorer
npm run dev &
FRONTEND_PID=$!
cd ../..

echo ""
echo "=========================================="
echo "✓ Services Started Successfully"
echo "=========================================="
echo ""
echo "🌐 Backend:  http://localhost:8080"
echo "🖥️  Frontend: http://localhost:8000"
echo "📚 API Docs: http://localhost:8080/docs"
echo ""
echo "📋 Logs: backend.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""
echo "=========================================="
echo "Backend Logs (live):"
echo "=========================================="

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    kill $TAIL_PID 2>/dev/null
    # Give processes time to shutdown gracefully
    sleep 2
    # Force kill if still running
    kill -9 $BACKEND_PID 2>/dev/null
    kill -9 $FRONTEND_PID 2>/dev/null
    kill -9 $TAIL_PID 2>/dev/null
    echo "✓ Services stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

# Tail the backend log to show live output
tail -f backend.log &
TAIL_PID=$!

# Wait for processes
wait

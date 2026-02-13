#!/bin/bash
# Startup script for VisionAI Studio (Backend + Frontend)

echo "=========================================="
echo "VisionAI Studio - Starting Services"
echo "=========================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Start backend in background
echo ""
echo "Starting FastAPI backend on port 8000..."
python app.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Check if backend started successfully
if ! curl -s http://localhost:8000/ > /dev/null; then
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✓ Backend running at http://localhost:8000"

# Start frontend
echo ""
echo "Starting React frontend on port 5173..."
cd frontend/vision-explorer
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "✓ Services Started Successfully"
echo "=========================================="
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✓ Services stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

# Wait for processes
wait

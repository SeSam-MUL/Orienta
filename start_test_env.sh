#!/bin/bash
# Start test environment: Backend (port 8000) + Frontend (port 5173)
# Usage: source start_test_env.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve Python interpreter (env var > active conda env > PATH default)
if [ -n "$KIKUCHIPY_PYTHON" ]; then
    PYTHON="$KIKUCHIPY_PYTHON"
elif [ -n "$CONDA_PREFIX" ]; then
    if [ -f "$CONDA_PREFIX/python.exe" ]; then
        PYTHON="$CONDA_PREFIX/python.exe"
    elif [ -f "$CONDA_PREFIX/bin/python" ]; then
        PYTHON="$CONDA_PREFIX/bin/python"
    else
        PYTHON="python"
    fi
else
    PYTHON="python"
fi
echo "Using Python: $PYTHON"

echo "=== Stopping existing processes ==="
taskkill //F //IM python.exe 2>/dev/null
taskkill //F //IM node.exe 2>/dev/null
sleep 2

echo "=== Starting Backend (port 8000) ==="
cd "$PROJECT_DIR"
$PYTHON -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend health
for i in $(seq 1 15); do
  if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "Backend ready!"
    break
  fi
  sleep 1
done

echo "=== Starting Frontend (port 5173) ==="
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend
for i in $(seq 1 10); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:5173/ 2>/dev/null | grep -q 200; then
    echo "Frontend ready!"
    break
  fi
  sleep 1
done

echo "=== Loading test data ==="
curl -s -X POST "http://localhost:8000/api/ebsd/load" \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$PROJECT_DIR/Test_data/EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina\"}" > /dev/null

curl -s -X POST "http://localhost:8000/api/h5/open" \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$PROJECT_DIR/Test_data/EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina\"}" > /dev/null

echo ""
echo "=== Test environment ready ==="
echo "Backend:  http://localhost:8000 (PID $BACKEND_PID)"
echo "Frontend: http://localhost:5173 (PID $FRONTEND_PID)"
echo "EBSD data loaded: 90x120 grid, 8 EDS elements"
echo ""
echo "To stop: taskkill //F //IM python.exe && taskkill //F //IM node.exe"

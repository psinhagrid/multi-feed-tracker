#!/usr/bin/env bash
# Setup script for Re-ID (fast-reid) support
# Run from project root: ./setup_reid.sh

set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
FAST_REID_DIR="${PROJECT_ROOT}/fast-reid"

echo "=== Re-ID (fast-reid) Setup ==="

# 1. Clone fast-reid if not present
if [ ! -d "$FAST_REID_DIR" ]; then
  echo "Cloning fast-reid..."
  git clone https://github.com/JDAI-CV/fast-reid.git "$FAST_REID_DIR"
else
  echo "fast-reid already cloned at $FAST_REID_DIR"
fi

# 2. Install fast-reid dependencies (no setup.py in fast-reid; use pip for deps)
echo "Installing fast-reid dependencies..."
pip install yacs easydict prettytable termcolor gdown scikit-learn faiss-cpu 2>/dev/null || true
# Fix Python 3.10+ compatibility (Mapping moved to collections.abc)
python3 -c "
import os
for f in ['fastreid/evaluation/testing.py', 'fastreid/data/build.py']:
    path = os.path.join('$FAST_REID_DIR', f)
    if os.path.isfile(path):
        with open(path) as fp: c = fp.read()
        if 'from collections import Mapping' in c:
            c = c.replace('from collections import Mapping, OrderedDict',
                'from collections import OrderedDict\nfrom collections.abc import Mapping')
            c = c.replace('from collections import Mapping', 'from collections.abc import Mapping')
            with open(path, 'w') as fp: fp.write(c)
            print('Fixed Python 3.12 compat in', path)
" 2>/dev/null || true
cd "$PROJECT_ROOT"

# 3. Install faiss-cpu
echo "Installing faiss-cpu..."
pip install faiss-cpu opencv-python 2>/dev/null || true

# 4. Download pretrained model if not present
# Use market_sbs_R50.pth with sbs_R50.yml (Strong Baseline - they match)
MODEL_PTH="${FAST_REID_DIR}/model.pth"
REID_CONFIG="${FAST_REID_DIR}/configs/Market1501/sbs_R50.yml"

if [ ! -f "$MODEL_PTH" ]; then
  echo "Downloading pretrained model (market_sbs_R50.pth, ~294MB)..."
  curl -L -o "$MODEL_PTH" \
    "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R50.pth"
  echo "Model saved to $MODEL_PTH"
else
  echo "Model already exists at $MODEL_PTH"
fi

# 5. Verify config exists
if [ ! -f "$REID_CONFIG" ]; then
  echo "Warning: Config $REID_CONFIG not found. Trying bagtricks_R50.yml..."
  REID_CONFIG="${FAST_REID_DIR}/configs/Market1501/bagtricks_R50.yml"
fi

# 6. Update .env
ENV_FILE="${PROJECT_ROOT}/.env"

if ! grep -q "FAST_REID_PATH" "$ENV_FILE" 2>/dev/null; then
  echo "" >> "$ENV_FILE"
  echo "# Re-ID (fast-reid) - added by setup_reid.sh" >> "$ENV_FILE"
  echo "FAST_REID_PATH=${FAST_REID_DIR}" >> "$ENV_FILE"
  echo "REID_WEIGHTS_PATH=${MODEL_PTH}" >> "$ENV_FILE"
  echo "REID_CONFIG_PATH=${REID_CONFIG}" >> "$ENV_FILE"
  echo "REID_DEVICE=cpu" >> "$ENV_FILE"
  echo "Added Re-ID config to .env"
else
  # Remove old Re-ID lines and re-add
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' '/FAST_REID_PATH\|REID_WEIGHTS_PATH\|REID_CONFIG_PATH\|REID_DEVICE/d' "$ENV_FILE" 2>/dev/null || true
  else
    sed -i '/FAST_REID_PATH\|REID_WEIGHTS_PATH\|REID_CONFIG_PATH\|REID_DEVICE/d' "$ENV_FILE" 2>/dev/null || true
  fi
  echo "" >> "$ENV_FILE"
  echo "# Re-ID (fast-reid)" >> "$ENV_FILE"
  echo "FAST_REID_PATH=${FAST_REID_DIR}" >> "$ENV_FILE"
  echo "REID_WEIGHTS_PATH=${MODEL_PTH}" >> "$ENV_FILE"
  echo "REID_CONFIG_PATH=${REID_CONFIG}" >> "$ENV_FILE"
  echo "REID_DEVICE=cpu" >> "$ENV_FILE"
  echo "Updated Re-ID paths in .env"
fi

# Try cuda if available
if command -v nvidia-smi &>/dev/null; then
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/REID_DEVICE=cpu/REID_DEVICE=cuda/' "$ENV_FILE" 2>/dev/null || true
  else
    sed -i 's/REID_DEVICE=cpu/REID_DEVICE=cuda/' "$ENV_FILE" 2>/dev/null || true
  fi
  echo "Set REID_DEVICE=cuda (GPU detected)"
fi

echo ""
echo "=== Setup complete ==="
echo "Re-ID paths configured in .env:"
echo "  FAST_REID_PATH=$FAST_REID_DIR"
echo "  REID_WEIGHTS_PATH=$MODEL_PTH"
echo "  REID_CONFIG_PATH=${FAST_REID_DIR}/configs/Market1501/bagtricks_R50.yml"
echo ""
echo "Note: bagtricks_R50.yml expects a model trained with that config."
echo "market_bot_R50.pth uses 'bot' (BagOfTricks) - if you get config errors,"
echo "try: REID_CONFIG_PATH=\${FAST_REID_PATH}/configs/Market1501/bot_R50.yml"
echo ""
echo "Restart the backend for Re-ID to become available."

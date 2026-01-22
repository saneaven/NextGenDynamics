#!/bin/bash
# =============================================================================
# Launch TensorBoard for SpiderBotAIProject
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables if .env exists
if [ -f "$PROJECT_ROOT/docker/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/docker/.env" | xargs)
fi

PORT="${TENSORBOARD_PORT:-6006}"
LOG_DIR="${LOG_DIR:-/workspace/spiderbot/logs/skrl}"

echo "=============================================="
echo "Starting TensorBoard"
echo "=============================================="
echo "  Port: $PORT"
echo "  Log directory: $LOG_DIR"
echo ""
echo "Access TensorBoard at: http://localhost:$PORT"
echo "=============================================="

cd "$PROJECT_ROOT"

docker compose -f docker/docker-compose.yml run --rm \
    -p "$PORT:$PORT" \
    spiderbot \
    "tensorboard --logdir=$LOG_DIR --host=0.0.0.0 --port=$PORT"

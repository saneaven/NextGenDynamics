#!/bin/bash
# =============================================================================
# Run SpiderBotAIProject Training in Docker
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables if .env exists
if [ -f "$PROJECT_ROOT/docker/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/docker/.env" | xargs)
fi

# Default values (can be overridden by environment or command line)
NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
SEED="${SEED:-42}"
CHECKPOINT="${CHECKPOINT:-}"
EXTRA_ARGS="${@}"

echo "=============================================="
echo "SpiderBotAIProject Training"
echo "=============================================="
echo "  Environments: $NUM_ENVS"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Seed: $SEED"
if [ -n "$CHECKPOINT" ]; then
    echo "  Checkpoint: $CHECKPOINT"
fi
echo "=============================================="

# Build checkpoint argument if provided
CHECKPOINT_ARG=""
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
fi

cd "$PROJECT_ROOT"

docker compose -f docker/docker-compose.yml run --rm \
    -e NUM_ENVS="$NUM_ENVS" \
    spiderbot \
    "python scripts/skrlcustom/train.py \
        --task=SpiderBotAIProject-v0 \
        --headless \
        --num_envs=$NUM_ENVS \
        --max_iterations=$MAX_ITERATIONS \
        --seed=$SEED \
        $CHECKPOINT_ARG \
        $EXTRA_ARGS"

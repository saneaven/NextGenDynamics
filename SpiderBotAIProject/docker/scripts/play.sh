#!/bin/bash
# =============================================================================
# Run SpiderBotAIProject Evaluation in Docker
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables if .env exists
if [ -f "$PROJECT_ROOT/docker/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/docker/.env" | xargs)
fi

# Parse arguments
CHECKPOINT="${1:-}"
VIDEO="${VIDEO:-false}"
NUM_ENVS="${NUM_ENVS:-1}"
DEBUG_VIS="${DEBUG_VIS:-false}"
EXTRA_ARGS="${@:2}"

# Validate checkpoint argument
if [ -z "$CHECKPOINT" ]; then
    echo "Error: Checkpoint path required"
    echo ""
    echo "Usage: ./play.sh <checkpoint_path> [extra_args...]"
    echo ""
    echo "Examples:"
    echo "  ./play.sh /path/to/checkpoint.pt"
    echo "  VIDEO=true ./play.sh /path/to/checkpoint.pt"
    echo "  DEBUG_VIS=true ./play.sh /path/to/checkpoint.pt"
    echo ""
    echo "Environment variables:"
    echo "  VIDEO=true       Record video"
    echo "  DEBUG_VIS=true   Enable debug visualization"
    echo "  NUM_ENVS=N       Number of environments (default: 1)"
    exit 1
fi

echo "=============================================="
echo "SpiderBotAIProject Evaluation"
echo "=============================================="
echo "  Checkpoint: $CHECKPOINT"
echo "  Environments: $NUM_ENVS"
echo "  Video: $VIDEO"
echo "  Debug Vis: $DEBUG_VIS"
echo "=============================================="

# Build optional arguments
VIDEO_ARG=""
if [ "$VIDEO" = "true" ]; then
    VIDEO_ARG="--video"
fi

DEBUG_VIS_ARG=""
if [ "$DEBUG_VIS" = "true" ]; then
    DEBUG_VIS_ARG="--debug_vis"
fi

cd "$PROJECT_ROOT"

docker compose -f docker/docker-compose.yml run --rm \
    spiderbot \
    "python scripts/skrlcustom/play.py \
        --task=SpiderBotAIProject-v0 \
        --headless \
        --checkpoint=$CHECKPOINT \
        --num_envs=$NUM_ENVS \
        $VIDEO_ARG \
        $DEBUG_VIS_ARG \
        $EXTRA_ARGS"

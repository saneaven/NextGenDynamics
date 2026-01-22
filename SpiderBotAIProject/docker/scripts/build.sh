#!/bin/bash
# =============================================================================
# Build SpiderBotAIProject Docker Image
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables if .env exists
if [ -f "$PROJECT_ROOT/docker/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/docker/.env" | xargs)
fi

# Default values
ISAAC_LAB_VERSION="${ISAAC_LAB_VERSION:-2.3.1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=============================================="
echo "Building SpiderBotAIProject Docker Image"
echo "=============================================="
echo "  Isaac Lab Version: $ISAAC_LAB_VERSION"
echo "  Image Tag: spiderbot-ai:$IMAGE_TAG"
echo "  Project Root: $PROJECT_ROOT"
echo "=============================================="

cd "$PROJECT_ROOT"

docker build \
    --build-arg ISAAC_LAB_VERSION="$ISAAC_LAB_VERSION" \
    -t "spiderbot-ai:$IMAGE_TAG" \
    -f docker/Dockerfile \
    .

echo ""
echo "=============================================="
echo "Build complete: spiderbot-ai:$IMAGE_TAG"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  ./docker/scripts/train.sh      # Start training"
echo "  ./docker/scripts/shell.sh      # Open interactive shell"
echo "  ./docker/scripts/tensorboard.sh # Launch TensorBoard"

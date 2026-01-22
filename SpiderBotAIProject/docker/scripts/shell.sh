#!/bin/bash
# =============================================================================
# Open Interactive Shell in SpiderBotAIProject Container
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables if .env exists
if [ -f "$PROJECT_ROOT/docker/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/docker/.env" | xargs)
fi

echo "=============================================="
echo "Opening interactive shell..."
echo "=============================================="
echo ""
echo "Useful commands inside the container:"
echo "  python scripts/skrlcustom/train.py --help"
echo "  python scripts/skrlcustom/play.py --help"
echo "  python scripts/list_envs.py"
echo "  nvidia-smi"
echo ""

cd "$PROJECT_ROOT"

docker compose -f docker/docker-compose.yml run --rm \
    spiderbot \
    "/bin/bash"

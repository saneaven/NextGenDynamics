# =============================================================================
# SpiderBotAIProject Docker Image
# Based on NVIDIA Isaac Lab official container
# =============================================================================

# Build argument for Isaac Lab version flexibility
ARG ISAAC_LAB_VERSION=2.3.1
FROM nvcr.io/nvidia/isaac-lab:${ISAAC_LAB_VERSION}

# Metadata
LABEL version="0.1.0"

# Set environment variables
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /workspace/spiderbot

# Copy project files
# Note: .dockerignore excludes logs, outputs, __pycache__, etc.
COPY . /workspace/spiderbot/

# Install the SpiderBotAIProject extension in editable mode
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -e /workspace/spiderbot/source/SpiderBotAIProject

# Install additional dependencies if needed
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    psutil \
    "skrl>=1.4.3"

# Create directories for outputs (will be mounted as volumes)
RUN mkdir -p /workspace/spiderbot/logs \
    && mkdir -p /workspace/spiderbot/outputs \
    && mkdir -p /workspace/spiderbot/checkpoints \
    && mkdir -p /workspace/spiderbot/videos

# Set default environment variables for training
ENV SPIDERBOT_LOG_DIR=/workspace/spiderbot/logs
ENV SPIDERBOT_OUTPUT_DIR=/workspace/spiderbot/outputs
ENV SPIDERBOT_CHECKPOINT_DIR=/workspace/spiderbot/checkpoints

# Default command shows help
CMD ["echo 'SpiderBotAIProject Container Ready. Use: train.py, play.py, or /bin/bash'"]

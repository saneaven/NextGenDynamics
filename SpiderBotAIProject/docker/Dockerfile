# Pre-built Isaac Lab headless container from NGC
FROM nvcr.io/nvidia/isaac-lab:2.2.0

# Install project runtime deps into Isaac Lab python environment.
# NOTE: We intentionally keep project code out of the image and mount it at runtime.
RUN bash -lc "set -e; \
  ISAACLAB_DIR=/workspace/IsaacLab; \
  if [ ! -d \"${ISAACLAB_DIR}\" ]; then ISAACLAB_DIR=/workspace/isaaclab; fi; \
  cd \"${ISAACLAB_DIR}\"; \
  ./isaaclab.sh -p -m pip install --no-cache-dir 'skrl>=1.4.3' 'opensimplex==0.4.5.1'"

# Drop-from-root entrypoint to avoid root-owned artifacts on bind mounts.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

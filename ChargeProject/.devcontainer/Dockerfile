# Use the official Isaac Lab image as the base
FROM nvcr.io/nvidia/isaac-lab:2.2.0

# Set the working directory
WORKDIR /workspace/isaaclab

# Copy the project files into the image for the build step
COPY . /workspace/isaaclab/source/ChargeProject

# Install the custom project package in editable mode.
# This allows your code changes to be reflected without rebuilding.
RUN ./isaaclab.sh -p -m pip install -e ./source/ChargeProject/source/ChargeProject
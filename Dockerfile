# Use Ubuntu 20.04 - force x86_64 architecture
FROM --platform=linux/amd64 ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3 and system openvdb
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-numpy \
    python3-openvdb \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set working directory
WORKDIR /app

# Copy the conversion script
COPY utils/npz_to_vdb.py /app/npz_to_vdb.py

# Create directories for input/output (will be mounted as volumes)
RUN mkdir -p /app/input /app/output

# Default command: convert all files in input directory
CMD ["python", "npz_to_vdb.py", "/app/input", "/app/output"]

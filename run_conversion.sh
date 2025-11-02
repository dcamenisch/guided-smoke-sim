#!/bin/bash

# Simple helper script to run NPZ to VDB conversion using Docker

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}NPZ to VDB Converter (Docker)${NC}"
echo "================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Build the Docker image if it doesn't exist
if ! docker image inspect smoke-sim-vdb-converter > /dev/null 2>&1; then
    echo -e "${BLUE}Building Docker image...${NC}"
    docker-compose build
fi

# Create output directory if it doesn't exist
mkdir -p vdb_output

# Check if custom arguments are provided
if [ $# -eq 0 ]; then
    # Default: convert all files in output/ directory
    echo -e "${GREEN}Converting all NPZ files in output/ directory...${NC}"
    docker-compose run --rm vdb-converter
else
    # Custom command passed as arguments
    echo -e "${GREEN}Running custom conversion command...${NC}"
    docker-compose run --rm vdb-converter python npz_to_vdb.py "$@"
fi

echo -e "${GREEN}Conversion complete! VDB files are in vdb_output/${NC}"

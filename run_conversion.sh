#!/bin/bash

# Simple helper script to run NPZ to VDB conversion using Docker

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
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

# Check if custom arguments are provided
if [ $# -eq 0 ]; then
    # Default: show usage help
    echo -e "${YELLOW}Usage examples:${NC}"
    echo ""
    echo "Convert a specific experiment (auto-generates output dir):"
    echo -e "  ${GREEN}./run_conversion.sh results/experiment-name${NC}"
    echo -e "  ${BLUE}→ Creates: results/experiment-name-vdb/${NC}"
    echo ""
    echo "Convert with custom output directory:"
    echo -e "  ${GREEN}./run_conversion.sh results/experiment-name results/custom-output${NC}"
    echo ""
    echo "Convert specific field (e.g., vorticity):"
    echo -e "  ${GREEN}./run_conversion.sh results/experiment-name --field vorticity${NC}"
    echo ""
    echo -e "${YELLOW}Note: Paths are relative to the project root${NC}"
    echo ""
    docker-compose run --rm vdb-converter
else
    # Custom command passed as arguments
    # Convert local paths to Docker container paths
    ARGS=()
    for arg in "$@"; do
        # Check if argument looks like a local path (starts with results/, output/, or is a relative path without -)
        if [[ "$arg" == results/* ]] || [[ "$arg" == output/* ]] || [[ "$arg" == ./* ]]; then
            # Convert to container path
            ARGS+=("/app/$arg")
        else
            # Keep as-is (flags, options, etc.)
            ARGS+=("$arg")
        fi
    done
    
    echo -e "${GREEN}Running conversion...${NC}"
    echo -e "${BLUE}Command: python npz_to_vdb.py ${ARGS[@]}${NC}"
    docker-compose run --rm vdb-converter python npz_to_vdb.py "${ARGS[@]}"
    echo ""
    echo -e "${GREEN}✓ Conversion complete!${NC}"
fi

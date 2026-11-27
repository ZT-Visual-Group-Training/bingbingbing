#!/usr/bin/env bash
# One-time environment setup for Ubuntu 22.04.
set -euo pipefail

echo "==> installing build tools and OpenCV"
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev

echo
echo "==> versions"
cmake --version | head -n1
g++ --version | head -n1
pkg-config --modversion opencv4

echo
echo "done. next: ./scripts/build.sh"

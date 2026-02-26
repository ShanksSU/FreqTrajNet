#!/bin/bash

# Stop script on error (unless handled)
set -e

# Define Paths (Current Directory)
INSTALL_ROOT=$(pwd)
KALDI_DIR="$INSTALL_ROOT/kaldi"
TEST_DIR="$INSTALL_ROOT/Test"

echo "=========================================="
echo "Starting installation in: $INSTALL_ROOT"
echo "=========================================="

# Install System Dependencies
echo "[Step 1] Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q \
  git make automake autoconf libtool \
  g++ gcc gfortran \
  zlib1g-dev unzip \
  sox \
  wget curl \
  python3 python3-pip \
  perl

# Clone Kaldi
echo "[Step 2] Checking Kaldi source..."
if [ -d "$KALDI_DIR" ]; then
    echo "Directory 'kaldi' already exists. Skipping clone."
else
    git clone https://github.com/kaldi-asr/kaldi.git
fi

# Compile OpenFst (Kaldi Tools)
echo "[Step 3] Compiling Kaldi tools (OpenFst)..."
cd "$KALDI_DIR/tools"

mkdir -p python
touch python/.use_default_python

# Check dependencies (allow warnings)
set +e
./extras/check_dependencies.sh
set -e

echo "Compiling tools with $(nproc) cores..."
make -j$(nproc)

# Compile SCTK
echo "[Step 4] Compiling SCTK (sclite)..."

# Logic to handle sctk directory or symlink safely
if [ -L "sctk" ]; then
    echo "Symlink 'sctk' found. Entering..."
    cd sctk
elif ls -d sctk-* > /dev/null 2>&1; then
    # Pick the first directory found to avoid 'too many arguments' error
    SCTK_DIR=$(ls -d sctk-* | head -n 1)
    echo "Entering directory: $SCTK_DIR"
    cd "$SCTK_DIR"
else
    echo "Error: sctk directory not found in tools."
    exit 1
fi

echo "Configuring SCTK..."
make config
echo "Compiling SCTK..."
make all -j$(nproc)
make install
echo "SCTK compilation complete."

# Verification
cd "$INSTALL_ROOT"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "=========================================="
echo "[Step 5] Verifying Installation"
echo "=========================================="

# Setup Paths for testing
OPENFST_BIN="$KALDI_DIR/tools/openfst/bin"
# Dynamically find sclite binary
SCLITE_BIN=$(find "$KALDI_DIR/tools" -name sclite -type f | head -n 1)

# CRITICAL: Set Library Path for OpenFst
export LD_LIBRARY_PATH="$KALDI_DIR/tools/openfst/lib:$LD_LIBRARY_PATH"

# Test 1: OpenFst
echo "Test 1: Checking OpenFst..."
if [ -f "$OPENFST_BIN/fstinfo" ]; then
    # Run with || true because fstinfo --help returns exit code 1 on some versions
    "$OPENFST_BIN/fstinfo" --help > fst_test.log 2>&1 || true
    
    # Check if the log contains usage info
    if grep -q "Usage" fst_test.log || grep -q "PROGRAM FLAGS" fst_test.log; then
        echo "SUCCESS: OpenFst is working."
    else
        echo "FAILURE: OpenFst execution failed. Check log:"
        cat fst_test.log
    fi
    rm -f fst_test.log
else
    echo "FAILURE: fstinfo binary not found."
fi

# Test 2: SCTK
echo "Test 2: Checking SCTK..."
if [ -n "$SCLITE_BIN" ] && [ -x "$SCLITE_BIN" ]; then
    "$SCLITE_BIN" -h > sclite_test.log 2>&1 || true
    
    if grep -q "Usage" sclite_test.log || grep -q "sclite" sclite_test.log; then
        echo "SUCCESS: sclite is working."
    else
        echo "FAILURE: sclite execution failed."
    fi
    rm -f sclite_test.log
else
    echo "FAILURE: sclite binary not found."
fi

# Finalize
echo "=========================================="
echo "Installation & Verification Finished!"
echo ""
echo "To use these tools, run this command now:"
echo "------------------------------------------"
echo "export KALDI_ROOT=$KALDI_DIR"
echo "export PATH=\$PATH:$OPENFST_BIN:$(dirname "$SCLITE_BIN")"
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$KALDI_DIR/tools/openfst/lib"
echo "------------------------------------------"
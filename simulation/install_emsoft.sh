#!/usr/bin/env bash
# =============================================================================
# EMsoft + EMSphInx Full Installation Script for WSL / Linux / macOS
# =============================================================================
# This script installs:
#   0. Pre-flight checks (architecture, RAM, disk, network)
#   1. 16GB swap file (if not present)
#   2. Build dependencies (clang-18, cmake, ninja, etc.)
#   3. POCL (OpenCL backend - CUDA optional, CPU fallback)
#   4. CMake 3.27.9 (downgrade required for EMsoft)
#   5. EMsoft Superbuild (SDK)
#   6. EMsoft (with OpenMP)
#   7. EMSphInx
#   8. Config files + PATH setup
#   9. Verification
#
# Features:
#   - Resume support: re-run the script to continue from the last completed step
#   - Platform detection: WSL2, native Linux, macOS (CPU-only)
#   - Safe parallelism: adjusts make -j based on available RAM
#
# Requirements:
#   - Ubuntu/Debian (WSL or native) or macOS
#   - ~20 GB free disk space, >= 4 GB RAM recommended
#   - Internet connection
#   - Duration: approximately 30-90 minutes depending on hardware
# =============================================================================

set -e

# --- Resume marker system ---
# Each completed phase writes a marker file. Re-running the script skips
# completed phases, enabling safe resume after failures or timeouts.
MARKER_DIR="$HOME/.emsoft_install_progress"
mkdir -p "$MARKER_DIR"

phase_done() {
  [ -f "$MARKER_DIR/phase_$1_done" ]
}

mark_phase() {
  touch "$MARKER_DIR/phase_$1_done"
  echo ">> Phase $1 completed."
}

echo "=========================================="
echo "  EMsoft + EMSphInx Installation"
echo "=========================================="
echo ""
echo "Start time: $(date)"
echo ""

# --- 0. PRE-FLIGHT CHECKS ---
echo "[0/9] Pre-flight system checks..."

# Architecture check
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
  echo ">> Architecture: $ARCH (x86_64) — OK"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  # Apple Silicon (M1/M2/M3/M4) and Linux ARM64 are supported
  # EMsoft ReadMe: "On ARM processors, both the Superbuild and EMsoft 5.X have been successfully built."
  echo ">> Architecture: $ARCH (ARM64) — supported"
  if [ "$(uname -s)" = "Darwin" ]; then
    echo ">> Apple Silicon detected. EMsoft + EMSphInx build natively on M-series."
  fi
else
  echo "=========================================="
  echo "  WARNING: Untested architecture: $ARCH"
  echo "=========================================="
  echo "EMsoft is tested on x86_64 and arm64."
  echo "This architecture may work but is not officially supported."
  echo "Continuing anyway..."
fi

# Platform detection
IS_WSL=false
IS_MACOS=false
if grep -qi microsoft /proc/version 2>/dev/null; then
  IS_WSL=true
  echo ">> Platform: WSL2 (Windows Subsystem for Linux)"
elif [ "$(uname -s)" = "Darwin" ]; then
  IS_MACOS=true
  echo ">> Platform: macOS (CPU-only mode — no NVIDIA GPU support)"
else
  echo ">> Platform: Native Linux"
fi

# RAM check — adjust parallelism to avoid OOM kills
TOTAL_RAM_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}' || echo 0)
if [ "$TOTAL_RAM_MB" -gt 0 ]; then
  echo ">> RAM: ${TOTAL_RAM_MB} MB"
  if [ "$TOTAL_RAM_MB" -lt 2048 ]; then
    echo "=========================================="
    echo "  ERROR: Insufficient RAM (${TOTAL_RAM_MB} MB)"
    echo "=========================================="
    echo "EMsoft compilation requires at least 4 GB RAM."
    echo "With swap, 2 GB physical RAM may work but is very slow."
    exit 1
  elif [ "$TOTAL_RAM_MB" -lt 4096 ]; then
    echo ">> WARNING: Low RAM. Build parallelism will be reduced."
  fi
fi

# Calculate safe make parallelism based on RAM
# Rule: ~1.5 GB per compiler process, minimum 1, maximum nproc
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
if [ "$TOTAL_RAM_MB" -gt 0 ]; then
  MAX_BY_RAM=$((TOTAL_RAM_MB / 1500))
  [ "$MAX_BY_RAM" -lt 1 ] && MAX_BY_RAM=1
  [ "$MAX_BY_RAM" -gt "$NPROC" ] && MAX_BY_RAM="$NPROC"
  MAKE_JOBS="$MAX_BY_RAM"
else
  MAKE_JOBS="$NPROC"
fi
echo ">> CPU cores: $NPROC, build parallelism: make -j${MAKE_JOBS}"

# Disk space check (in home directory)
AVAIL_GB=$(df -BG "$HOME" 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}' || echo 0)
if [ "$AVAIL_GB" -gt 0 ]; then
  echo ">> Disk space: ${AVAIL_GB} GB available in $HOME"
  if [ "$AVAIL_GB" -lt 10 ]; then
    echo "=========================================="
    echo "  ERROR: Insufficient disk space (${AVAIL_GB} GB)"
    echo "=========================================="
    echo "EMsoft installation needs at least 15-20 GB free space."
    exit 1
  elif [ "$AVAIL_GB" -lt 20 ]; then
    echo ">> WARNING: Low disk space. Installation may fail if < 15 GB remain."
  fi
fi

# Network connectivity check
echo -n ">> Network: "
if wget -q --spider https://github.com --timeout=10 2>/dev/null; then
  echo "OK (github.com reachable)"
elif curl -s --head https://github.com --connect-timeout 10 >/dev/null 2>&1; then
  echo "OK (github.com reachable via curl)"
else
  echo "=========================================="
  echo "  WARNING: Cannot reach github.com"
  echo "=========================================="
  echo "The installation requires downloading source code from GitHub."
  echo "If you are behind a proxy or firewall, configure git/wget proxy settings."
  echo "Continuing anyway — downloads may fail."
fi

echo ""

# --- 1. SWAP ---
if phase_done 1; then
  echo "[1/9] Swap — already done, skipping."
else
  echo "[1/9] Setting up swap..."
  if [ "$IS_MACOS" = true ]; then
    echo ">> macOS manages swap automatically. Skipping."
  elif ! grep -q ' /swapfile ' /etc/fstab 2>/dev/null && [ ! -f /swapfile ]; then
    echo ">> Creating 16G swap file ..."
    sudo fallocate -l 16G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    sudo swapon -a || true
  else
    echo ">> Swap already exists."
  fi
  swapon --show 2>/dev/null || true
  free -h 2>/dev/null || true
  mark_phase 1
fi

# --- 2. Build Dependencies ---
if phase_done 2; then
  echo "[2/9] Build dependencies — already done, skipping."
else
  echo ""
  echo "[2/9] Installing build dependencies..."
  if [ "$IS_MACOS" = true ]; then
    echo ">> macOS: using Homebrew..."
    brew install llvm cmake ninja pkg-config hdf5 gcc open-mpi autoconf automake libtool wget || true
  else
    sudo apt update
    sudo apt install -y lsb-release wget software-properties-common
    wget -q https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    sudo ./llvm.sh 18
    sudo apt install -y \
      build-essential git cmake clang-18 llvm-18 llvm-18-dev \
      libclang-18-dev libclang-cpp18-dev libclang-rt-18-dev libclang1-18 \
      libhwloc-dev ocl-icd-opencl-dev libopencl-clang-dev \
      ninja-build \
      opencl-headers opencl-c-headers opencl-clhpp-headers clinfo
    rm -f llvm.sh
  fi
  mark_phase 2
fi

# --- NVIDIA detection (always runs, needed by Phase 3) ---
HAS_NVIDIA=false
if [ "$IS_MACOS" = true ]; then
  echo ">> macOS: No NVIDIA GPU support. CPU-only mode."
elif nvidia-smi &>/dev/null; then
  HAS_NVIDIA=true
  echo ">> NVIDIA GPU detected via WSL2 passthrough."
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null || true
  echo ""

  # Check that the WSL2 GPU libraries are available
  if [ -d /usr/lib/wsl/lib ]; then
    echo ">> WSL2 GPU libraries found at /usr/lib/wsl/lib/"
    ls /usr/lib/wsl/lib/libcuda* /usr/lib/wsl/lib/libnvidia* 2>/dev/null | head -5 || true
  else
    echo ">> WARNING: /usr/lib/wsl/lib/ not found. Is this WSL2?"
    echo "   GPU support requires WSL2 + a recent Windows NVIDIA driver (>= 470)."
  fi

  # Install CUDA toolkit from NVIDIA's repo (compiler + libs only, NO driver)
  echo ">> Installing CUDA toolkit from NVIDIA repository..."
  # Add NVIDIA's CUDA keyring if not present
  if ! dpkg -l cuda-keyring 2>/dev/null | grep -q "^ii"; then
    DISTRO_ID=$(. /etc/os-release && echo "${ID}${VERSION_ID}" | tr -d '.')
    # Map to supported distro (e.g. ubuntu2204)
    case "$DISTRO_ID" in
      ubuntu2004|ubuntu2204|ubuntu2404) ;;
      *) DISTRO_ID="ubuntu2204" ;;  # fallback
    esac
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO_ID}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb 2>/dev/null && \
      sudo dpkg -i /tmp/cuda-keyring.deb && \
      rm -f /tmp/cuda-keyring.deb && \
      sudo apt update || echo ">> WARNING: Could not add NVIDIA CUDA repo. Trying system packages..."
  fi

  # Install CUDA toolkit (compiler, headers, libraries — NO driver)
  sudo apt install -y cuda-toolkit 2>/dev/null || \
    sudo apt install -y nvidia-cuda-toolkit nvidia-cuda-dev 2>/dev/null || \
    echo ">> WARNING: CUDA toolkit install failed. PoCL will use CPU-only mode."
else
  echo ">> No NVIDIA GPU detected (nvidia-smi not available). CPU-only mode."
fi

# --- 3. POCL (OpenCL) ---
if phase_done 3; then
  echo "[3/9] POCL — already done, skipping."
else
echo ""
echo "[3/9] Building POCL (OpenCL backend)..."
cd ~
if [ ! -d pocl ]; then
  git clone https://github.com/pocl/pocl.git
fi
cd pocl
mkdir -p build
cd build

# Try CUDA first, fall back to CPU-only if it fails
POCL_BUILD_OK=false

if [ "$HAS_NVIDIA" = true ]; then
  echo ">> Attempting POCL build with CUDA + CPU support..."

  # Find CUDA toolkit root (varies by installation method)
  CUDA_ROOT=""
  for candidate in /usr/local/cuda /usr/local/cuda-* /usr; do
    if [ -f "$candidate/bin/nvcc" ] || [ -f "$candidate/include/cuda.h" ]; then
      CUDA_ROOT="$candidate"
      break
    fi
  done

  if [ -z "$CUDA_ROOT" ]; then
    echo ">> CUDA toolkit not found (no nvcc). Skipping CUDA PoCL build."
  else
    echo ">> Using CUDA toolkit at: $CUDA_ROOT"

    # WSL2: ensure libcuda.so is findable (provided by Windows driver)
    if [ -d /usr/lib/wsl/lib ] && ! echo "$LD_LIBRARY_PATH" | grep -q "/usr/lib/wsl/lib"; then
      export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
      echo ">> Added /usr/lib/wsl/lib to LD_LIBRARY_PATH for WSL2 GPU libs"
    fi

    set +e  # Temporarily allow failures
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DENABLE_HOST_CPU_DEVICES=ON \
      -DENABLE_CUDA=ON \
      -DWITH_CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
      -DENABLE_LLVM=ON \
      -DSTATIC_LLVM=OFF \
      -DCMAKE_BUILD_TYPE=Release 2>&1
    if [ $? -eq 0 ]; then
      make -j${MAKE_JOBS} 2>&1
      if [ $? -eq 0 ]; then
        POCL_BUILD_OK=true
        echo ">> POCL with CUDA built successfully."
      fi
    fi
    set -e  # Re-enable strict mode

    if [ "$POCL_BUILD_OK" = false ]; then
      echo ">> CUDA build failed. Cleaning up and retrying CPU-only..."
      cd ~/pocl
      rm -rf build
      mkdir -p build
      cd build
    fi
  fi
fi

if [ "$POCL_BUILD_OK" = false ]; then
  echo ">> Building POCL with CPU-only support..."
  cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DENABLE_HOST_CPU_DEVICES=ON \
    -DENABLE_CUDA=OFF \
    -DENABLE_LLVM=ON \
    -DSTATIC_LLVM=OFF \
    -DCMAKE_BUILD_TYPE=Release
  make -j${MAKE_JOBS}
  echo ">> POCL CPU-only built successfully."
fi

sudo make install

# --- Post-install: platform-specific library registration ---
if [ "$IS_MACOS" = true ]; then
  # macOS: update dylib cache, register PoCL via Homebrew-compatible paths
  echo ">> macOS: updating library cache..."
  # PoCL installs to /usr/local/lib/ — macOS finds it via DYLD_LIBRARY_PATH or install_name
  # No ldconfig on macOS, no /etc/OpenCL/vendors needed — PoCL registers itself via pkg-config
  # But we still create the ICD dir if the ICD loader expects it
  sudo mkdir -p /etc/OpenCL/vendors 2>/dev/null || true
  if [ -f /usr/local/lib/libpocl.dylib ]; then
    echo "/usr/local/lib/libpocl.dylib" | sudo tee /etc/OpenCL/vendors/pocl.icd 2>/dev/null || true
    echo ">> PoCL registered (macOS .dylib)"
  elif [ -f /usr/local/lib/libpocl.so ]; then
    echo "libpocl.so" | sudo tee /etc/OpenCL/vendors/pocl.icd 2>/dev/null || true
  fi
  echo ""
  echo ">> macOS also provides Apple OpenCL.framework (CPU compute)."
  echo ">> EMsoft will use Apple OpenCL or PoCL, whichever cmake finds."
else
  # Linux / WSL: ldconfig + ICD registration
  sudo ldconfig
  sudo mkdir -p /etc/OpenCL/vendors
  echo "libpocl.so" | sudo tee /etc/OpenCL/vendors/pocl.icd

  # Register NVIDIA OpenCL ICD if driver is present
  # In WSL2 the library is at /usr/lib/wsl/lib/, on native Linux at /usr/lib/x86_64-linux-gnu/
  NVIDIA_OCL_LIB=""
  if [ -f /usr/lib/wsl/lib/libnvidia-opencl.so.1 ]; then
    NVIDIA_OCL_LIB="/usr/lib/wsl/lib/libnvidia-opencl.so.1"
    echo ">> Found NVIDIA OpenCL at WSL2 path: $NVIDIA_OCL_LIB"
  elif [ -f /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1 ]; then
    NVIDIA_OCL_LIB="libnvidia-opencl.so.1"
    echo ">> Found NVIDIA OpenCL at native Linux path."
  fi

  if [ -n "$NVIDIA_OCL_LIB" ]; then
    echo "$NVIDIA_OCL_LIB" | sudo tee /etc/OpenCL/vendors/nvidia.icd
    echo ">> NVIDIA OpenCL ICD registered."
    # WSL2: also ensure /usr/lib/wsl/lib is in the linker path
    if [ -d /usr/lib/wsl/lib ]; then
      if [ ! -f /etc/ld.so.conf.d/wsl-gpu.conf ]; then
        echo "/usr/lib/wsl/lib" | sudo tee /etc/ld.so.conf.d/wsl-gpu.conf
        sudo ldconfig
        echo ">> Added /usr/lib/wsl/lib to linker search path."
      fi
    fi
  fi
fi

echo ""
echo "POCL installed. Testing OpenCL..."
clinfo -l || echo "WARNING: clinfo failed, OpenCL may not be fully working yet"
mark_phase 3
fi

# --- 4. CMake Downgrade ---
if phase_done 4; then
  echo "[4/9] CMake — already done, skipping."
else
echo ""
echo "[4/9] Installing CMake 3.27.9..."
if [ "$IS_MACOS" = true ]; then
  echo ">> macOS: using Homebrew cmake. Skipping downgrade."
else
  sudo apt -y remove cmake || true
  cd /tmp
  wget -q https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.tar.gz
  sudo tar -C /opt -xzf cmake-3.27.9-linux-x86_64.tar.gz
  sudo ln -sfn /opt/cmake-3.27.9-linux-x86_64/bin/* /usr/local/bin/
  rm -f cmake-3.27.9-linux-x86_64.tar.gz
  hash -r
fi
echo "CMake version: $(cmake --version | head -1)"
mark_phase 4
fi

# --- 5. EMsoft Superbuild (SDK) ---
if phase_done 5; then
  echo "[5/9] EMsoft SDK — already done, skipping."
else
echo ""
echo "[5/9] Building EMsoft Superbuild (SDK)..."
if [ "$IS_MACOS" != true ]; then
  sudo apt update
  sudo apt -y install build-essential git gfortran g++ ninja-build pkg-config \
                      ocl-icd-opencl-dev opencl-headers autoconf automake libtool \
                      libhdf5-dev hdf5-tools
fi

mkdir -p $HOME/emsoft/{SDK,src,builds}
cd $HOME/emsoft/src
if [ ! -d EMsoftSuperbuild ]; then
  git clone https://github.com/EMsoft-org/EMsoftSuperbuild.git
fi

mkdir -p $HOME/emsoft/builds/EMsoftSuperbuild
cd $HOME/emsoft/builds/EMsoftSuperbuild

SUPERBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DEMsoft_SDK=$HOME/emsoft/SDK"
if [ "$IS_MACOS" = true ]; then
  # macOS: help cmake find Homebrew gfortran and Apple's OpenCL.framework
  BREW_PREFIX=$(brew --prefix 2>/dev/null || echo "/usr/local")
  GCC_VER=$(ls "${BREW_PREFIX}/bin/gfortran-"* 2>/dev/null | head -1 | grep -o '[0-9]*$')
  if [ -n "$GCC_VER" ]; then
    export FC="${BREW_PREFIX}/bin/gfortran-${GCC_VER}"
    export CC="${BREW_PREFIX}/bin/gcc-${GCC_VER}"
    export CXX="${BREW_PREFIX}/bin/g++-${GCC_VER}"
    echo ">> macOS: using Homebrew GCC ${GCC_VER} (FC=$FC)"
  fi
  # Apple's OpenCL is at /System/Library/Frameworks/OpenCL.framework
  # cmake should find it automatically via find_package(OpenCL)
  SUPERBUILD_CMAKE_ARGS="$SUPERBUILD_CMAKE_ARGS -DCMAKE_PREFIX_PATH=${BREW_PREFIX}"
fi

cmake $SUPERBUILD_CMAKE_ARGS $HOME/emsoft/src/EMsoftSuperbuild
make -j"${MAKE_JOBS}"
mark_phase 5
fi

# --- 6. EMsoft ---
if phase_done 6; then
  echo "[6/9] EMsoft — already done, skipping."
else
echo ""
echo "[6/9] Building EMsoft (with OpenMP)..."
if [ "$IS_MACOS" = true ]; then
  # macOS: OpenBLAS + LAPACK via Homebrew (already installed via Phase 2)
  BREW_PREFIX=$(brew --prefix 2>/dev/null || echo "/usr/local")
  echo ">> macOS: using Homebrew at $BREW_PREFIX"
else
  sudo apt -y install libopenblas-dev liblapack-dev
fi
cd ~/emsoft/src
[ -d EMsoft ] || git clone https://github.com/EMsoft-org/EMsoft.git
mkdir -p ~/emsoft/builds/EMsoft-Release
cd ~/emsoft/builds/EMsoft-Release

EMSOFT_CMAKE_ARGS="-G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DEMsoft_SDK=$HOME/emsoft/SDK \
  -DUSE_OpenMP=ON \
  -DEMsoft_ENABLE_TESTING=OFF \
  -DCLFortran_DIR=$HOME/emsoft/SDK/CLFortran-0.0.1-Release/lib/cmake/CLFortran"
if [ "$IS_MACOS" = true ]; then
  EMSOFT_CMAKE_ARGS="$EMSOFT_CMAKE_ARGS -DCMAKE_PREFIX_PATH=${BREW_PREFIX}"
fi

cmake $EMSOFT_CMAKE_ARGS ~/emsoft/src/EMsoft
ninja -j"${MAKE_JOBS}"
mark_phase 6
fi

# --- 7. EMSphInx ---
if phase_done 7; then
  echo "[7/9] EMSphInx — already done, skipping."
else
echo ""
echo "[7/9] Building EMSphInx..."
if [ "$IS_MACOS" != true ]; then
  sudo apt -y install \
    libgtk-3-dev libsm-dev \
    libx11-dev libxext-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libgl1-mesa-dev libglu1-mesa-dev
fi
cd ~/emsoft/src
if [ ! -d EMSphInx ]; then
  git clone https://github.com/EMsoft-org/EMSphInx.git
fi

mkdir -p ~/emsoft/builds/EMSphInx-Release
cd ~/emsoft/builds/EMSphInx-Release

SPHINX_GUI=ON
SPHINX_EXTRA_ARGS=""
if [ "$IS_MACOS" = true ]; then
  # macOS: skip GUI build (needs GTK/X11 which aren't standard on macOS)
  SPHINX_GUI=OFF
  echo ">> macOS: building EMSphInx without GUI (IndexEBSD CLI only)"
  # Apple Silicon: FFTW SIMD must be disabled (EMSphInx ReadMe, 02/2023)
  if [ "$ARCH" = "arm64" ]; then
    SPHINX_EXTRA_ARGS="-DEMSPHINX_FFTW_SIMD=OFF"
    echo ">> Apple Silicon: FFTW SIMD disabled (required for ARM)"
  fi
fi

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DEMSPHINX_BUILD_GUIS=$SPHINX_GUI \
  $SPHINX_EXTRA_ARGS \
  ~/emsoft/src/EMSphInx

ninja fftwd
ninja hdf5
ninja -j"${MAKE_JOBS}"
mark_phase 7
fi

# --- Define path variables needed by Phase 8 and 9 (always, even on resume) ---
EMSOFT_BUILD="$HOME/emsoft/builds/EMsoft-Release"
EMSOFT_BIN="$EMSOFT_BUILD/Bin"
EMSOFT_DATA="$HOME/EMsoftData"
EMSOFT_XTAL="$HOME/EMsoftXtal"
EMSOFT_TMP="$HOME/.config/EMsoft/tmp"

# Find the actual EMSphInx binary location
EMSPHINX_BIN=""
EMSPHINX_BUILD="$HOME/emsoft/builds/EMSphInx-Release"
if [ -x "$EMSPHINX_BUILD/IndexEBSD" ]; then
  EMSPHINX_BIN="$EMSPHINX_BUILD"
elif [ -x "$EMSPHINX_BUILD/bin/IndexEBSD" ]; then
  EMSPHINX_BIN="$EMSPHINX_BUILD/bin"
elif [ -x "$EMSPHINX_BUILD/Bin/IndexEBSD" ]; then
  EMSPHINX_BIN="$EMSPHINX_BUILD/Bin"
else
  # Search for it
  FOUND_PATH=$(find "$EMSPHINX_BUILD" -name "IndexEBSD" -type f -executable 2>/dev/null | head -1)
  if [ -n "$FOUND_PATH" ]; then
    EMSPHINX_BIN=$(dirname "$FOUND_PATH")
  else
    echo "WARNING: IndexEBSD binary not found in $EMSPHINX_BUILD"
    echo "         EMSphInx may not have built correctly."
    EMSPHINX_BIN="$EMSPHINX_BUILD"
  fi
fi
echo "EMSphInx binary directory: $EMSPHINX_BIN"

# --- 8. Config + PATH ---
if phase_done 8; then
  echo "[8/9] Config + PATH — already done, skipping."
else
echo ""
echo "[8/9] Creating EMsoft config and PATH entries..."
mkdir -p $HOME/EMsoftData
mkdir -p $HOME/EMsoftXtal
mkdir -p $HOME/.config/EMsoft/tmp

# Create EMsoft config
REALNAME="$(getent passwd "$USER" | cut -d: -f5 | cut -d, -f1)"
[ -z "$REALNAME" ] && REALNAME="$USER"
EMAIL="$(git config --global user.email 2>/dev/null || true)"
LOCATION="$(hostname)"

cat > "$HOME/.config/EMsoft/EMsoftConfig.json" <<EOF
{
  "EMsoftpathname":       "${EMSOFT_BIN}/",
  "EMXtalFolderpathname": "${EMSOFT_XTAL}/",
  "EMdatapathname":       "${EMSOFT_DATA}/",
  "EMtmppathname":        "${EMSOFT_TMP}/",
  "EMsoftLibraryLocation":"${EMSOFT_BUILD}/",
  "Release":              "Yes",
  "Develop":              "No",
  "UserName":             "${REALNAME}",
  "UserEmail":            "${EMAIL}",
  "UserLocation":         "${LOCATION}"
}
EOF

echo "Wrote: $HOME/.config/EMsoft/EMsoftConfig.json"
cat "$HOME/.config/EMsoft/EMsoftConfig.json"

# Add to PATH in .profile (works for both interactive and non-interactive login shells)
# Note: .bashrc has a non-interactive guard that prevents PATH from being set
# when called via 'bash -lc' (e.g. from Python subprocess), so .profile is the
# correct location for PATH additions.
if ! grep -q "EMsoft/EMSphInx PATH" "$HOME/.profile"; then
  cat >> "$HOME/.profile" <<EOF

# --- EMsoft/EMSphInx PATH ---
export PATH="\$PATH:$EMSOFT_BIN:$EMSPHINX_BIN"
EOF
  echo "Added PATH entries to ~/.profile"
fi
# Also add to .bashrc for interactive-only shells (terminals)
if [ -f "$HOME/.bashrc" ] && ! grep -q "EMsoft/EMSphInx PATH" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<EOF

# --- EMsoft/EMSphInx PATH ---
export PATH="\$PATH:$EMSOFT_BIN:$EMSPHINX_BIN"
EOF
  echo "Added PATH entries to ~/.bashrc"
fi
# macOS: also add to .zprofile (default shell is zsh since Catalina)
if [ "$IS_MACOS" = true ] && ! grep -q "EMsoft/EMSphInx PATH" "$HOME/.zprofile" 2>/dev/null; then
  cat >> "$HOME/.zprofile" <<EOF

# --- EMsoft/EMSphInx PATH ---
export PATH="\$PATH:$EMSOFT_BIN:$EMSPHINX_BIN"
EOF
  echo "Added PATH entries to ~/.zprofile (macOS zsh)"
fi

# Source bashrc to make PATH available immediately
export PATH="$PATH:$EMSOFT_BIN:$EMSPHINX_BIN"
mark_phase 8
fi

# --- 9. Verification ---
echo ""
echo "[9/9] Verifying installation..."
echo "=========================================="
echo "  VERIFICATION"
echo "=========================================="
echo ""

PASS=0
FAIL=0

echo -n "EMMCOpenCL:       "
if command -v EMMCOpenCL &>/dev/null; then
  echo "OK ($(which EMMCOpenCL))"
  PASS=$((PASS+1))
else
  echo "FAIL - not found in PATH"
  FAIL=$((FAIL+1))
fi

echo -n "EMEBSDmasterSHT:  "
if command -v EMEBSDmasterSHT &>/dev/null; then
  echo "OK ($(which EMEBSDmasterSHT))"
  PASS=$((PASS+1))
else
  echo "FAIL - not found in PATH"
  FAIL=$((FAIL+1))
fi

echo -n "IndexEBSD:        "
if command -v IndexEBSD &>/dev/null; then
  echo "OK ($(which IndexEBSD))"
  PASS=$((PASS+1))
else
  echo "FAIL - not found in PATH"
  FAIL=$((FAIL+1))
  echo "  Searching manually..."
  find ~/emsoft/builds/EMSphInx-Release/ -name "IndexEBSD" -type f 2>/dev/null || echo "  Not found anywhere in EMSphInx build directory."
fi

echo -n "OpenCL ICD files: "
ICD_OK=true
# Determine correct PoCL library name for this platform
if [ "$IS_MACOS" = true ] && [ -f /usr/local/lib/libpocl.dylib ]; then
  POCL_LIB="/usr/local/lib/libpocl.dylib"
else
  POCL_LIB="libpocl.so"
fi
# Verify pocl.icd contains a valid library path (not garbage like "1234")
if [ -f /etc/OpenCL/vendors/pocl.icd ]; then
  POCL_CONTENT=$(cat /etc/OpenCL/vendors/pocl.icd)
  if [[ "$POCL_CONTENT" != *"libpocl"* ]]; then
    echo -n "REPAIRING pocl.icd (was: '$POCL_CONTENT')... "
    echo "$POCL_LIB" | sudo tee /etc/OpenCL/vendors/pocl.icd > /dev/null
  fi
elif [ "$IS_MACOS" != true ]; then
  # Only auto-create on Linux (macOS may not need /etc/OpenCL/vendors)
  echo -n "CREATING pocl.icd... "
  sudo mkdir -p /etc/OpenCL/vendors
  echo "$POCL_LIB" | sudo tee /etc/OpenCL/vendors/pocl.icd > /dev/null
fi
# Verify nvidia.icd if NVIDIA driver present (WSL2 or native)
NV_LIB_PATH=""
if [ -f /usr/lib/wsl/lib/libnvidia-opencl.so.1 ]; then
  NV_LIB_PATH="/usr/lib/wsl/lib/libnvidia-opencl.so.1"
elif [ -f /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1 ]; then
  NV_LIB_PATH="libnvidia-opencl.so.1"
fi
if [ -n "$NV_LIB_PATH" ]; then
  if [ -f /etc/OpenCL/vendors/nvidia.icd ]; then
    NV_CONTENT=$(cat /etc/OpenCL/vendors/nvidia.icd)
    if [[ "$NV_CONTENT" != *"libnvidia-opencl"* ]]; then
      echo -n "REPAIRING nvidia.icd (was: '$NV_CONTENT')... "
      echo "$NV_LIB_PATH" | sudo tee /etc/OpenCL/vendors/nvidia.icd > /dev/null
    fi
  else
    echo -n "CREATING nvidia.icd... "
    echo "$NV_LIB_PATH" | sudo tee /etc/OpenCL/vendors/nvidia.icd > /dev/null
  fi
fi
echo "OK"

echo -n "OpenCL:           "
if clinfo -l &>/dev/null; then
  echo "OK"
  clinfo -l
  PASS=$((PASS+1))
else
  echo "WARNING - clinfo not working (GPU simulation may be slow)"
fi

echo -n "EMsoft Config:    "
if [ -f "$HOME/.config/EMsoft/EMsoftConfig.json" ]; then
  echo "OK"
  PASS=$((PASS+1))
else
  echo "FAIL"
  FAIL=$((FAIL+1))
fi

echo ""
echo "=========================================="
if [ $FAIL -eq 0 ]; then
  echo "  ALL CHECKS PASSED ($PASS/$PASS)"
else
  echo "  $PASS passed, $FAIL FAILED"
fi
echo "=========================================="
echo ""
echo "Installation paths:"
echo "  EMsoft Bin:   $EMSOFT_BIN"
echo "  EMSphInx Bin: $EMSPHINX_BIN"
echo "  EMsoft Data:  $EMSOFT_DATA"
echo "  EMsoft Xtal:  $EMSOFT_XTAL"
echo "  Config:       $HOME/.config/EMsoft/EMsoftConfig.json"
echo ""
echo "For your emsphinx_config.ini:"
echo "  EMMCOpenCL_Executable_WSL = $EMSOFT_BIN/EMMCOpenCL"
echo "  EMEBSDmasterSHT_Executable_WSL = $EMSOFT_BIN/EMEBSDmasterSHT"
echo ""
echo "Restart your WSL terminal or run: source ~/.bashrc"
echo "End time: $(date)"

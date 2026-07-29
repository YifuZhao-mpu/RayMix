#!/bin/bash
# Build torchsparse for the pointcept env (torch 2.7.1+cu128, sm_120 Blackwell).
set -x
PY=/root/miniconda3/envs/pointcept/bin/python
PIP=/root/miniconda3/envs/pointcept/bin/pip
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=48
SCRATCH=/tmp/claude-0/-root-Fresh-ARIS8-paper-q2/86f7feb7-1221-44ae-b75a-b4647f538900/scratchpad

# 1) sparsehash headers (header-only) — apt first, git fallback
if [ ! -d /usr/include/sparsehash ] && [ ! -d "$SCRATCH/sparsehash/src/sparsehash" ]; then
  apt-get install -y libsparsehash-dev || {
    cd "$SCRATCH" && rm -rf sparsehash
    git clone --depth 1 https://github.com/sparsehash/sparsehash.git
    cd sparsehash && ./configure >/dev/null && make -j8 >/dev/null 2>&1 || true
  }
fi
# header include path fallback (sparsehash generates config headers into src/)
if [ ! -d /usr/include/sparsehash ]; then
  export CPATH="$SCRATCH/sparsehash/src:${CPATH:-}"
fi
$PY -c "import glob;print('sparsehash at /usr/include:', bool(glob.glob('/usr/include/sparsehash')))"

# 2) build torchsparse v2.1.0
cd "$SCRATCH" && rm -rf torchsparse
git clone --depth 1 --branch v2.1.0 https://github.com/mit-han-lab/torchsparse.git
cd torchsparse
$PIP install . --no-build-isolation -v 2>&1 | tail -40

# 3) verify import (CPU-level; GPU test done separately on GPU 4/5 when free)
$PY -c "import torchsparse; print('TORCHSPARSE_OK', torchsparse.__version__)"

# ---------- Base: CUDA 11.7, cuDNN8, Ubuntu 22.04 ----------
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    WORKROOT=/workspace \
    PIP_ROOT_USER_ACTION=ignore

# ---------- OS deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl wget ca-certificates \
    build-essential cmake ninja-build \
    python3 python3-dev python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Alias python3 -> python; lock pip/setuptools to avoid PEP517 surprises
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade "pip<25" "setuptools<81" "wheel<0.45"

# Keep NumPy/SciPy-compatible pins for older OpenMMLab
RUN pip install --no-cache-dir "numpy==1.23.5" pybind11 packaging

# ---------- PyTorch 1.13.1 + CUDA 11.7 ----------
RUN pip install --no-cache-dir \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# ---------- OpenMMLab base (prebuilt mmcv wheel for cu117/torch1.13) ----------
RUN pip install --no-cache-dir \
     mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html && \
    pip install --no-cache-dir mmdet==2.28.2 mmsegmentation==0.30.0 transformers==4.31.0

# OpenCV (full) to satisfy mmcv-full and stay NumPy 1.x compatible
RUN pip install --no-cache-dir --force-reinstall opencv-python==4.10.0.84 --no-deps

# ---------- spconv must match cu117 before mmdetection3d ----------
RUN pip install --no-cache-dir spconv-cu117==2.3.6

# ---------- Clone and set up repos ----------
WORKDIR ${WORKROOT}

# mmdetection3d: build with env CUDA 11.7, force CUDA build, avoid PEP517 isolation
ENV CUDA_HOME=/usr/local/cuda-11.7 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    FORCE_CUDA=1 \
    PIP_NO_BUILD_ISOLATION=1

RUN git clone --depth 1 --branch v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git && \
    cd mmdetection3d && \
    pip install -v . --no-build-isolation --no-deps

# OpenLane-V2: install runtime deps only (avoid old ortools pins)
RUN git clone --depth 1 https://github.com/OpenDriveLab/OpenLane-V2.git && \
    python -m pip install --no-cache-dir \
      scipy==1.8.0 \
      "scikit-learn==1.3.2" \
      similaritymeasures \
      ninja \
      shapely==1.8.5.post1 \
      "ortools>=9.10"

# Make both repos importable
ENV PYTHONPATH=/workspace/mmdetection3d:/workspace/OpenLane-V2

# ---------- Copy your OmniDrive code ----------
WORKDIR ${WORKROOT}/OmniDrive
COPY . ${WORKROOT}/OmniDrive

# ---------- Normalize project requirements, then install ----------
RUN if [ -f requirements.txt ]; then \
      sed -i 's/^shapely==1\.8\.5\.post$/shapely==1.8.5.post1/' requirements.txt || true; \
      sed -i 's/^numpy==1\.23\.4$/numpy==1.23.5/' requirements.txt || true; \
      sed -i 's/^setuptools==.*/# setuptools pinned by project; using image default/' requirements.txt || true; \
      sed -i 's/^opencv-python-headless.*/opencv-python==4.10.0.84/' requirements.txt || true; \
      sed -i 's/^opencv-python.*/opencv-python==4.10.0.84/' requirements.txt || true; \
      sed -i 's/^accelerate.*/# accelerate requires torch>=2; skipping for torch 1.13.1/' requirements.txt || true; \
    fi
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN pip uninstall -y opencv-python-headless || true
RUN pip uninstall -y accelerate openmim opendatalab openxlab || true

# ---------- Add mmdet3d runtime deps to satisfy rc6 metadata ----------
RUN pip install --no-cache-dir \
      nuscenes-devkit==1.1.11 \
      plyfile==0.7.4 \
      scikit-image==0.21.0 \
      tensorboard==2.12.0 \
      trimesh==3.23.5 \
      numba==0.56.4 llvmlite==0.39.0 \
      lyft-dataset-sdk==0.0.8

# ---------- Lock core stack (guard against accidental upgrades) ----------
RUN pip install --no-cache-dir --force-reinstall \
      torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
      --extra-index-url https://download.pytorch.org/whl/cu117 --no-deps && \
    pip install --no-cache-dir --force-reinstall \
      numpy==1.23.5 scipy==1.8.0 shapely==1.8.5.post1 opencv-python==4.10.0.84 --no-deps && \
    (pip check || true)

# Create expected folders
RUN mkdir -p ${WORKROOT}/OmniDrive/ckpts ${WORKROOT}/OmniDrive/data/nuscenes ${WORKROOT}/OmniDrive/work_dirs

# ---------- Entrypoint ----------
WORKDIR ${WORKROOT}/OmniDrive
CMD ["/bin/bash"]

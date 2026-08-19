# Policy server — Jetson AGX Thor (JetPack 7.2, L4T R39.2, aarch64, sm_110).
#
# BUILD THIS ON THE THOR, or on another aarch64 host with the same JetPack line.
# Isaac-GR00T's Thor installer refuses any other architecture on purpose: a
# plain `uv sync` pulls the dGPU torch build (sm_80/90/100/120) and every kernel
# launch then dies with "no kernel image available" on Thor's sm_110. The Thor
# install path is the only one that produces sm_110 kernels, and it can only run
# where those wheels are valid.
#
#   docker build -f docker/Dockerfile.thor -t <registry>/manipurl-thor:<tag> .
#
# The base image is the NGC CUDA image, not an l4t-* tag: JetPack 7 uses unified
# Arm CUDA, so this is what the organizer's README names for the Thor. Pulling
# it needs an NGC login even though it is public -- see INSTRUCTIONS.md.
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV DEBIAN_FRONTEND=noninteractive

# The package lists are deliberately left in place: install_deps.sh decides
# whether to add its own CUDA apt repo by asking apt-cache whether it can see
# libnvpl-lapack0, and an empty /var/lib/apt/lists makes that check fail for
# the wrong reason.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        build-essential yasm cmake libtool git git-lfs pkg-config curl ca-certificates \
        libass-dev libfreetype6-dev libvorbis-dev \
        autoconf automake texinfo ffmpeg

# --- normalise the CUDA apt repo before the installer touches it ------------
# The base image registers the CUDA repo WITHOUT a Signed-By option.
# install_deps.sh installs NVIDIA's cuda-keyring whenever it cannot find
# libnvpl-lapack0, and that package drops a SECOND entry for the same URI that
# does set Signed-By. apt refuses to read any sources at all when two entries
# disagree, so the installer dies on its first apt call:
#
#   E: Conflicting values set for option Signed-By regarding source
#      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/
#
# So we do it once, properly: drop the unsigned entry, install the keyring, and
# assert the package is visible. The installer's own check then finds it and
# skips its version of all this.
RUN curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
    && rm -f /etc/apt/sources.list.d/*cuda* \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm -f /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-cache show libnvpl-lapack0 > /dev/null

# --- Isaac-GR00T with the Thor (sm_110) dependency set ----------------------
# Pinned by commit: this is the branch carrying the BCT recipe the deployed
# checkpoint was trained with. The torchcodec wheel under
# scripts/deployment/thor/wheels/ is a git-lfs object, so lfs must be pulled.
ARG GR00T_REPO=https://github.com/RooibosT/Isaac-GR00T.git
ARG GR00T_REF=bb9be1f9a17cc61c5a70cc85b68aeafb6ef1a50c
ENV DOCKER_CONTAINER=1
ENV UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv

RUN git lfs install --system \
    && git clone --filter=blob:none "${GR00T_REPO}" /opt/Isaac-GR00T \
    && cd /opt/Isaac-GR00T \
    && git checkout "${GR00T_REF}" \
    && git lfs pull \
    && bash scripts/deployment/thor/install_deps.sh

# Same environment scripts/activate_thor.sh sets up on bare metal.
ENV VIRTUAL_ENV=/opt/gr00t-venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV CUDA_PATH=/usr/local/cuda-13.0
ENV CPATH="/usr/local/cuda-13.0/include"
ENV C_INCLUDE_PATH="/usr/local/cuda-13.0/include"
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda-13.0/include"
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudss/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

# --- boundary + transport deps ----------------------------------------------
COPY docker/requirements-thor.txt /tmp/requirements-thor.txt
RUN pip install --no-cache-dir -r /tmp/requirements-thor.txt \
    && python -c "import cv2, numpy, zmq, msgpack, websockets, torch; print('imports ok:', cv2.__version__, numpy.__version__, torch.__version__)"

# --- the submission ----------------------------------------------------------
WORKDIR /submission
COPY boundary/ /submission/boundary/
COPY components/ /submission/components/
COPY mocks/ /submission/mocks/
COPY assets/ /submission/assets/
COPY scripts/ /submission/scripts/
COPY docs/boundary.sha256 /submission/docs/boundary.sha256
COPY conformance.py requirements.txt /submission/
COPY docker/entrypoint_thor.sh /usr/local/bin/entrypoint_thor.sh
RUN chmod +x /usr/local/bin/entrypoint_thor.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/submission
# Weights are NOT baked in; mount them read-only at /weights. See INSTRUCTIONS.md.
ENV PEVAL_CHECKPOINT=/weights/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40
ENV PEVAL_LANE=decoupled
ENV PEVAL_THOR_PORT=8765

EXPOSE 8765
ENTRYPOINT ["/usr/local/bin/entrypoint_thor.sh"]
CMD ["python", "components/server.py", "--lane", "decoupled", "--port", "8765"]

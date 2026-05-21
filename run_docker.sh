#!/usr/bin/env bash
set -e  # exit on error

if [ -z "$1" ]
  then
    echo "Please specify docker image name as the first argument"
    echo "Usage $0 DOCKER_IMAGE_NAME"
    exit 1
fi

DOCKER_IMAGE=${1}
shift # Consume argument 1
RUN_DOCKER_INTERACTIVE=${RUN_DOCKER_INTERACTIVE:-1}
ROOT_DIR=$(cd "$(dirname "$0")"/../../; pwd)
#CACHE_DIR=${CACHE_DIR:-$HOME/.cache/$DOCKER_IMAGE}
if [[ -z "${DOCKER_USER}" ]]; then
  DOCKER_USER="$(id -u):$(id -g)"
fi

DEBUG_FLAGS="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
DOCKER_MAP="-v /home/ymx/git_project/Multi-LLM-Agent:/home/Multi-llm-agent -w /home -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v \
  $ROOT_DIR:/source -v /home/ymx/.cache:/root/.cache -v /data0/ymx/.cache/:/root/ymx/.cache"

DOCKER_FLAGS="--rm ${DOCKER_MAP}  -e HOME=/home --network=host --user root --ipc=host --security-opt seccomp=unconfined ${DEBUG_FLAGS}"
if [[ ${DOCKER_IMAGE} == *"rocm"* ]]; then
    DOCKER_FLAGS="${DOCKER_FLAGS} --device=/dev/kfd --device=/dev/dri --group-add video"
elif [[ ${DOCKER_IMAGE} == *"cuda"* ]]; then
    DOCKER_FLAGS="${DOCKER_FLAGS} --gpus all"
elif [[ ${DOCKER_IMAGE} == *"multi"* ]]; then
    DOCKER_FLAGS="${DOCKER_FLAGS} --runtime=nvidia \
    -e CUDA_MPS_PIPE_DIRECTORY=/home \
    -e CUDA_MPS_LOG_DIRECTORY=/home \
    -e NVIDIA_VISIBLE_DEVICES=1 \
    -e TMPDIR=/home/my_tmp \
    "
fi

if [ "${RUN_DOCKER_INTERACTIVE}" -eq 1 ]; then
    DOCKER_CMD="docker run -it ${DOCKER_FLAGS} ${DOCKER_IMAGE}"
else
    DOCKER_CMD="docker run -i ${DOCKER_FLAGS} ${DOCKER_IMAGE}"
fi

${DOCKER_CMD} "$@"

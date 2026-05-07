#!/bin/bash
VERSION="v01"
PROJECT="syn_ecg-eandd"
mode=${1:-none}
HOSTNAME_C=$PROJECT"-"`hostname`
CONTAINER_NAME=$PROJECT"-"$VERSION
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRNAME="$(basename "$SCRIPT_DIR")"

HOST_ORIGINAL_DATA_PATH="$SCRIPT_DIR/raw_data"
HOST_PROCESSED_DATA_PATH="$SCRIPT_DIR/outputs"
HOST_SYSTEM_DATA_PATH="$SCRIPT_DIR/system_data"

CONTAINER_REPO_DIR="/home/$USER/$DIRNAME"
CONTAINER_ORIGINAL_DATA_PATH="$CONTAINER_REPO_DIR/raw_data"
CONTAINER_PROCESSED_DATA_PATH="$CONTAINER_REPO_DIR/outputs"
CONTAINER_SYSTEM_DATA_PATH="$CONTAINER_REPO_DIR/system_data"

mkdir -p \
  "$HOST_ORIGINAL_DATA_PATH" \
  "$HOST_PROCESSED_DATA_PATH" \
  "$HOST_SYSTEM_DATA_PATH"

if [ $mode = "buildx" ]; then
    docker buildx build --load \
        --build-arg USERNAME="$USER" \
        --build-arg UID="$UID" \
        --build-arg GID="$(id -g "$USER")" \
        --build-arg ORIGINAL_DATA_PATH="$CONTAINER_ORIGINAL_DATA_PATH" \
        --build-arg PROCESSED_DATA_PATH="$CONTAINER_PROCESSED_DATA_PATH" \
        --build-arg SYSTEM_DATA_PATH="$CONTAINER_SYSTEM_DATA_PATH" \
        --progress=plain \
        -t "$PROJECT:$VERSION" \
        .
    docker run \
        --gpus all \
        -v "$HOST_ORIGINAL_DATA_PATH:$CONTAINER_ORIGINAL_DATA_PATH" \
        -v "$HOST_PROCESSED_DATA_PATH:$CONTAINER_PROCESSED_DATA_PATH" \
        -v "$HOST_SYSTEM_DATA_PATH:$CONTAINER_SYSTEM_DATA_PATH" \
        -v "$SCRIPT_DIR:$CONTAINER_REPO_DIR" \
        -it -d --shm-size=180g \
        --hostname=$HOSTNAME_C \
        --name $CONTAINER_NAME $PROJECT:$VERSION  /bin/bash
elif [ $mode = "build" ]; then
    docker buildx build \
        --build-arg USERNAME=$USER \
        --build-arg UID=$UID \
        --build-arg GID=$(id -g $USER) \
        --build-arg ORIGINAL_DATA_PATH=$CONTAINER_ORIGINAL_DATA_PATH \
        --build-arg PROCESSED_DATA_PATH=$CONTAINER_PROCESSED_DATA_PATH \
        --build-arg SYSTEM_DATA_PATH=$CONTAINER_SYSTEM_DATA_PATH \
        --progress=plain \
        -t $PROJECT:$VERSION . < Dockerfile
    docker run \
        --gpus all \
        -v "$HOST_ORIGINAL_DATA_PATH:$CONTAINER_ORIGINAL_DATA_PATH" \
        -v "$HOST_PROCESSED_DATA_PATH:$CONTAINER_PROCESSED_DATA_PATH" \
        -v "$HOST_SYSTEM_DATA_PATH:$CONTAINER_SYSTEM_DATA_PATH" \
        -v "$SCRIPT_DIR:$CONTAINER_REPO_DIR" \
        -it -d --shm-size=180g \
        --hostname=$HOSTNAME_C \
        --name $CONTAINER_NAME $PROJECT:$VERSION  /bin/bash        
elif [ $mode = "start" ]; then
    docker run \
        --gpus all \
        -v "$HOST_ORIGINAL_DATA_PATH:$CONTAINER_ORIGINAL_DATA_PATH" \
        -v "$HOST_PROCESSED_DATA_PATH:$CONTAINER_PROCESSED_DATA_PATH" \
        -v "$HOST_SYSTEM_DATA_PATH:$CONTAINER_SYSTEM_DATA_PATH" \
        -v "$SCRIPT_DIR:$CONTAINER_REPO_DIR" \
        -it -d --shm-size=180g \
        --hostname=$HOSTNAME_C \
        --name $CONTAINER_NAME $PROJECT:$VERSION  /bin/bash
elif [ $mode = "restart" ]; then
    docker start $CONTAINER_NAME 
fi

docker exec -it $CONTAINER_NAME /bin/bash

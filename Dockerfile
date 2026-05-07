# Base image.
FROM nvcr.io/nvidia/cuda:11.6.1-runtime-ubuntu20.04

# Set timezone and update packages in a single RUN command
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        make \
        lzma \
        liblzma-dev \
        gcc \
        zlib1g-dev \
        bzip2 \
        libbz2-dev \
        libreadline8 \
        libreadline-dev \
        sqlite3 \
        libsqlite3-dev \
        openssl \
        libssl-dev \
        build-essential \
        git \
        curl \
        wget \
        vim \
        sudo \
        libffi-dev \
        libgl1-mesa-dev \
        libglib2.0-0 \
        lsb-release \
        gnupg \
        python3.10 \
        python3-pip \
        byobu && \
    rm /bin/sh && \
    ln -s /bin/bash /bin/sh

# Env variables.
ARG USERNAME
ENV USERNAME=$USERNAME
ARG UID
ENV UID=$UID
ARG GID
ENV GID=$GID
ARG ORIGINAL_DATA_PATH
ENV ORIGINAL_DATA_PATH=$ORIGINAL_DATA_PATH
ARG PROCESSED_DATA_PATH
ENV PROCESSED_DATA_PATH=$PROCESSED_DATA_PATH
ARG SYSTEM_DATA_PATH
ENV SYSTEM_DATA_PATH=$SYSTEM_DATA_PATH

# Prepare user and directory.
RUN groupadd -g $GID $USERNAME && \
    useradd -u $UID -g $GID -m -s /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd

RUN mkdir -p $ORIGINAL_DATA_PATH && \
    chmod 777 $ORIGINAL_DATA_PATH && \
    mkdir -p $PROCESSED_DATA_PATH && \
    chmod 777 $PROCESSED_DATA_PATH

WORKDIR /home/$USERNAME
USER $USERNAME
ENV HOME /home/$USERNAME

# Pyenv.
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH


RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    source ~/.bashrc && \
    pyenv install 3.10.2
RUN pyenv global 3.10.2
RUN source ~/.bashrc


RUN pip install --upgrade pip
RUN pip install --no-cache-dir --compile torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir --compile pandas matplotlib scikit-learn pyYaml tqdm optuna timm==0.9.12 numpy==1.26.2 dtw-python==1.5.3 umap-learn==0.5.7

RUN pip install --no-cache-dir --compile soundfile
RUN pip install --no-cache-dir --compile einops --no-deps
RUN pip install --no-cache-dir --compile fastai==2.7.14 tsai==0.3.9 --no-deps
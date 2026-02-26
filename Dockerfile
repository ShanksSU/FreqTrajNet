FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    vim \
    openssh-server \
    ca-certificates \
    make \
    automake \
    autoconf \
    libtool \
    g++ \
    gcc \
    gfortran \
    zlib1g-dev \
    unzip \
    sox \
    python3 \
    python3-pip \
    perl \
    libgl1-mesa-glx \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# install Fastfetch
RUN wget https://github.com/fastfetch-cli/fastfetch/releases/download/2.57.1/fastfetch-linux-amd64.deb && \
    apt-get update && \
    apt-get install -y ./fastfetch-linux-amd64.deb && \
    rm fastfetch-linux-amd64.deb

RUN mkdir /var/run/sshd
RUN echo 'root:123456' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22

RUN echo "set -g mouse on" > /root/.tmux.conf

ENV CONDA_DIR=/root/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    (conda config --remove channels defaults || true)

RUN conda create -n torch_222 python=3.10 --override-channels -c conda-forge -y
RUN /root/miniconda3/envs/torch_222/bin/pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
RUN conda init bash && \
    echo "conda activate torch_222" >> /root/.bashrc && \
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /root/.bashrc && \
    echo 'cd /root/workspace' >> /root/.bashrc && \
    echo 'fastfetch' >> /root/.bashrc

WORKDIR /root/workspace

CMD ["/usr/sbin/sshd", "-D"]

# C:\Users\<username>\.docker\config.json
# [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("username:password"))
# echo { "auths": { "https://index.docker.io/v1/": { "auth": "<Insert_Base64_String_Here>" } }, "credsStore": "" } > C:\Users\<username>\.docker\config.json

# docker build -t <image_name> .
# docker run -d --gpus all -p 8025:22 --name <container_name> <image_name>

# docker run -d --gpus all -p 8025:22 -v "C:\Users\<username>\Downloads\Dataset:/root/workspace/Dataset" --name <container_name> <image_name>

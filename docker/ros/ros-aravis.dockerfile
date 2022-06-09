ARG DISTRO=humble-desktop

FROM osrf/ros:${DISTRO} as ros-base
ARG DEBIAN_FRONTEND=noninteractive

ENV \
    TZ=Pacific/Auckland \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update --quiet \
    # Ensure Timezone is set
    && ln -f -s /usr/share/zoneinfo/${TZ} /etc/localtime \
    && apt-get install --yes --quiet --no-install-recommends \
        software-properties-common \
        wget \
        curl \
        gnupg \
        lsb-release \
        build-essential \
        git \
        cmake \ 
        ninja-build \ 
        meson \
        aravis-tools-cli \
        libaravis-0.8-0 \
        libaravis-dev \
        gir1.2-aravis-0.8 \
        libcairo2-dev \
        libgirepository1.0-dev \
        python3-dev \
        python3-cairo-dev \
        gir1.2-gtk-3.0 \
        sudo \
        gdb \
        gdbserver \
    && wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3 /tmp/get-pip.py \
    && python3 -m pip install -U pip setuptools wheel PyGObject opencv-python numpy

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && echo "\nsource /opt/ros/humble/setup.bash\n" >> /home/$USERNAME/.bashrc

# Use tini as entrypoint
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]
SHELL [ "/bin/bash", "-l", "-c" ]
WORKDIR /src
USER $USERNAME

ARG DISTRO=jammy

FROM ubuntu:${DISTRO} as ros-base
ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV \
    TZ=Pacific/Auckland \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Use tini as entrypoint
RUN apt-get update --quiet \
    && apt-get install --quiet --yes tini \
    && apt-get clean autoclean
ENTRYPOINT ["/usr/bin/tini", "--"]

RUN apt-get update --quiet \
    # Ensure Timezone is set
    && ln -f -s /usr/share/zoneinfo/${TZ} /etc/localtime \
    && apt-get install --yes --quiet --no-install-recommends \
        software-properties-common \
        sudo \
        wget \
        curl \
        gnupg \
        lsb-release

# Create the user with sudo support
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update --quiet \
    && apt-get upgrade --quiet --yes \
    && apt-get install --yes --quiet --no-install-recommends \
        build-essential \
        ros-humble-ros-base \
        python3-rosdep \
        python3-colcon-common-extensions

RUN wget -q -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3 /tmp/get-pip.py \
    # pyreadline3 for better line autocompletion
    && python3 -m pip install -U pip setuptools wheel pyreadline3

RUN rosdep init \
    && su ${USERNAME} bash -l -c "rosdep update"
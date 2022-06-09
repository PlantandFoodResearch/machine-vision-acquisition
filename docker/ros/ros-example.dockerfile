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
        curl \
        gnupg \
        lsb-release \
        build-essential \
        git \
        cmake \ 
        ninja-build \ 
        meson
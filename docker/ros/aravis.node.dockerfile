ARG DISTRO=noetic-ros-base-buster

FROM ros:${DISTRO} as ros-base
ARG DEBIAN_FRONTEND=noninteractive

ENV \
    TZ=Pacific/Auckland \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
        libaravis-dev \
        ros-noetic-camera-aravis \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_NAMESPACE=cam1
CMD ["rosrun", "camera_aravis", "cam_aravis"]
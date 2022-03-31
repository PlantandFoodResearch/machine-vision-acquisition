# bionic==18.04
ARG DISTRO=bionic
ARG GCC_MAJOR=11
ARG CMAKE_VERSION=3.21.4
ARG CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz
# storage.powerplant.pfr.co.nz has a self-signed certificate!
ARG SPINNAKER_URL=https://storage.powerplant.pfr.co.nz/workspace/software_cache/flir/spinnaker-2.5.0.80-Ubuntu18.04-amd64-pkg.tar.gz
ARG ARENA_URL=https://storage.powerplant.pfr.co.nz/workspace/software_cache/lucid/ArenaSDK_v0.1.57_Linux_x64.tar.gz
ARG CHRONOPTICS_URL=https://storage.powerplant.pfr.co.nz/workspace/software_cache/chronoptic/tof-linux-x86_64.tar.gz
ARG TEKNIC_URL=https://storage.powerplant.pfr.co.nz/workspace/software_cache/teknic/sFoundation.tar
ARG MV_GENTL_URL=http://static.matrix-vision.com/mvIMPACT_Acquire/2.46.2/mvGenTL_Acquire-ARM64_gnu-2.46.2.tgz
ARG MV_GENTL_INSTALL_URL=http://static.matrix-vision.com/mvIMPACT_Acquire/2.46.2/install_mvGenTL_Acquire_ARM.sh

FROM ubuntu:${DISTRO} as dev-container-base
# [Option] Install zsh
ARG INSTALL_ZSH="true"
# [Option] Upgrade OS packages to their latest versions
ARG UPGRADE_PACKAGES="true"

# Install needed packages and setup non-root user. Use a separate RUN statement to add your
# own dependencies. A user of "automatic" attempts to reuse an user ID if one already exists.
ARG USERNAME=automatic
ARG USER_UID=1000
ARG USER_GID=$USER_UID
COPY library-scripts/*.sh /tmp/library-scripts/
RUN apt-get update \
    && /bin/bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    # Clean up
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

CMD [ "sleep", "infinity" ]


#  FROM ubuntu:${DISTRO} as cmake-gcc  # If you don't want the dev container components
FROM dev-container-base as cmake-gcc
ARG DISTRO
ARG GCC_MAJOR
ARG CMAKE_URL
ARG CMAKE_VERSION
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
ARG DEBIAN_FRONTEND=noninteractive

LABEL Description="Ubuntu ${DISTRO} - Gcc${GCC_MAJOR} + CMake ${CMAKE_VERSION}"

ENV \
    TZ=Pacific/Auckland \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# install GCC
RUN apt-get update --quiet \
    # Ensure Timezone is set
    && ln -s /usr/share/zoneinfo/${TZ} /etc/localtime \
    && apt-get upgrade --yes --quiet \
    && apt-get install --yes --quiet --no-install-recommends \
        wget \
        gnupg \
        apt-transport-https \
        ca-certificates \
        tzdata \
        software-properties-common \
        lsb-release \
    # Add modern GCC repository
    && wget -qO - "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x60c317803a41ba51845e371a1e9377a2ba9ef27f" | apt-key add - \
    && echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/gcc.list \
    # Add modern cmake repository
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    && apt-get update --quiet \
    && apt-get install --yes --quiet --no-install-recommends \
        cmake \
        git \
        ninja-build \
        libstdc++-${GCC_MAJOR}-dev \
        gcc-${GCC_MAJOR} \
        g++-${GCC_MAJOR} \
        gdb \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-${GCC_MAJOR} 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-${GCC_MAJOR} 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_MAJOR} 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_MAJOR} 100 \
    && c++ --version \
    && apt-get --yes autoremove \
    && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*


FROM cmake-gcc as cmake-gcc-spinnaker-arena-opencv
# Install all dependencies
RUN apt-get update --quiet \
    && apt-get install --yes --quiet --no-install-recommends \
        # Spinnaker depenencies
        libavcodec57 \
        libavformat57 \
        libswscale4 \
        libswresample2 \
        libavutil55 \
        libraw1394-11 \
        libusb-1.0-0 \
        # Arena SDK dependencies
        libncurses5-dev \
        # Open CV
        libopencv-dev \
        python3-opencv \
    && apt-get --yes autoremove \
    && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Install cached spinnaker SDK
# ARG SPINNAKER_URL
# RUN mkdir -p /tmp/opt/spinnaker \
#     && wget --no-check-certificate -qO - ${SPINNAKER_URL} | tar --strip-components=1 -xz -C /tmp/opt/spinnaker \
#     && cd /tmp/opt/spinnaker \
#     && dpkg -i lib*.deb || true \
#     && echo "set libspinnaker/accepted-flir-eula true" | debconf-communicate \
#     && dpkg -i lib*.deb \
#     && cd / \ 
#     && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Download and unpack Arena SDK
# ARG ARENA_URL
# RUN mkdir -p /opt/src \
#     && mkdir -p /opt/src/ArenaSDK \
#     && wget --no-check-certificate -qO - ${ARENA_URL} | tar --strip-components=1 -xz -C /opt/src/ArenaSDK \
#     # Install library load paths
#     && cd /opt/src/ArenaSDK \
#     && chmod +x Arena_SDK_Linux_x64.conf \
#     && ./Arena_SDK_Linux_x64.conf \
#     && cd /
# ENV ARENA_ROOT="/opt/src/ArenaSDK"

# Download and chronoptics SDK
# ARG CHRONOPTICS_URL
# RUN mkdir -p /opt/src \
#     && mkdir -p /opt/src/chronoptics \
#     && wget --no-check-certificate -qO - ${CHRONOPTICS_URL} | tar -xz -C /opt/src/chronoptics \
#     # Install library load paths
#     && touch /etc/ld.so.conf.d/Chronoptics.conf \
#     && echo /opt/src/chronoptics/lib >> /etc/ld.so.conf.d/Chronoptics.conf
# ENV CHRONOPTICS_ROOT="/opt/src/chronoptics"

# Download and Teknic SDK
# we have to build the library
# ARG TEKNIC_URL
# RUN mkdir -p /opt/src \
#     && mkdir -p /opt/src/teknic \
#     && wget --no-check-certificate -qO - ${TEKNIC_URL} | tar -x -C /opt/src/teknic \
#     && cd /opt/src/teknic/sFoundation \
#     && make \
#     && ln -s /opt/src/teknic/sFoundation/libsFoundation20.so /opt/src/teknic/sFoundation/libsFoundation20.so.1 \
#     && rm -rf build \
#     # Install library load paths
#     && touch /etc/ld.so.conf.d/teknic.conf \
#     && echo /opt/src/teknic/sFoundation/ >> /etc/ld.so.conf.d/teknic.conf \
#     && cd /
# ENV TEKNIC_ROOT="/opt/src/teknic"

# Download and unpack mvGenTL
ARG MV_GENTL_URL
ARG MV_GENTL_INSTALL_URL
RUN mkdir -p /opt/src \
    && mkdir -p /opt/src/mvGenTL \
    && cd /opt/src/mvGenTL \
    && wget --no-check-certificate -q ${MV_GENTL_URL} \
    && wget --no-check-certificate -qO install.sh ${MV_GENTL_INSTALL_URL} \
    # Install library load paths
    && chmod +x install.sh \
    && ./install.sh --gev_support --u3v_support --unattended --minimal \
    && cd /
ENV MV_GENTL_ROOT="/opt/src/mvGenTL"
# bionic==18.04
ARG DISTRO=bionic
ARG GCC_MAJOR=11
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

LABEL Description="Ubuntu ${DISTRO} - Gcc${GCC_MAJOR} + CMake ${CMAKE_VERSION} + Python3.8"

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
    # Modern python
    && add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt-get update --quiet \
    && apt-get install --yes --quiet --no-install-recommends \
        # Some tools still require gcc 7 builders (includes aarch64-linux-gnu-gcc)
        build-essential \
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
    # Python 3.8
    && apt-get install --yes python3.8 python3.8-distutils python3.8-dev \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.8 get-pip.py \
    && apt-get --yes autoremove \
    && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*


FROM cmake-gcc as mvIMPACT-dev
# Get dependencies
RUN apt-get update --quiet \
    && apt-get install --yes \
        # Can probably remove these if not wanting any GUI. Not sure if the GUI works yet
        qt5-default \
        libwxgtk3.0-gtk3-0v5 \
        libwxgtk3.0-gtk3-dev \
        libwxgtk-media3.0-gtk3-0v5 \
        libwxgtk-media3.0-gtk3-dev \
        libwxgtk-webview3.0-gtk3 \
        libwxgtk-webview3.0-gtk3-dev \
    && apt-get clean autoclean

# Download and unpack mvIMPACT
ARG MV_GENTL_URL
ARG MV_GENTL_INSTALL_URL
RUN mkdir -p /opt/src/mvIMPACT \
    && cd /opt/src/mvIMPACT \
    && wget --no-check-certificate -q ${MV_GENTL_URL} \
    && wget --no-check-certificate -qO install.sh ${MV_GENTL_INSTALL_URL} \
    # Install library load paths
    && chmod +x install.sh \
    # Their script is not perfect and has a fair few errors...
    && ./install.sh --gev_support --u3v_support --unattended\
    # Build and install the python bindings
    && cd /opt/mvIMPACT_Acquire/LanguageBindings/Python/ \
    && MVIMPACT_ACQUIRE_DIR=/opt/mvIMPACT_Acquire python3.8 setup.py bdist_wheel \
    && MVIMPACT_ACQUIRE_DIR=/opt/mvIMPACT_Acquire python3.8 -m pip install Output/mvIMPACT*.whl \
    && cd /
ENV MVIMPACT_ACQUIRE_DIR=/opt/mvIMPACT_Acquire

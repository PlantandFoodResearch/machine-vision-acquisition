# bionic==18.04
ARG DISTRO=bionic
ARG GCC_MAJOR=11
ARG ARAVIS_URL=https://github.com/AravisProject/aravis/releases/download/0.8.21/aravis-0.8.21.tar.xz

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


FROM cmake-gcc as cmake-gcc-aravis-dev

# Download and unpack mvGenTL
ARG ARAVIS_URL
RUN mkdir -p /opt/src \
    && apt-get update --quiet \
    && apt-get install --yes \
        libxml2-dev \
        libglib2.0-dev \
        libusb-1.0-0-dev \
        gobject-introspection \
        libgtk-3-dev \
        gtk-doc-tools \
        xsltproc \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        libgstreamer-plugins-bad1.0-dev \
        gstreamer1.0-plugins-ugly \
        libgirepository1.0-dev \
        ninja-build \
    && apt-get clean autoclean \
    # apt's meson is too old. The others enable gi doc generation
    && pip3 install meson jinja2 markdown pygments toml typogrify

RUN mkdir -p /opt/src/aravis \
    && wget -qO - ${ARAVIS_URL} | tar --strip-components=1 -xJ -C /opt/src/aravis \
    && cd /opt/src/aravis \
    # Install library load paths
    && meson build \
    && cd build \
    && ninja \
    && ninja install \
    && cd /
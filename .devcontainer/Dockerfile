# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.224.3/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version (use hirsuite or bionic on local arm64/Apple Silicon): hirsute, focal, bionic
ARG VARIANT="bionic"
ARG GCC_MAJOR=11
ARG CMAKE_VERSION=3.21.4
ARG CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz

FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}
ARG VARIANT
ARG GCC_MAJOR
ARG CMAKE_VERSION
ARG CMAKE_URL
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
ARG DEBIAN_FRONTEND=noninteractive
# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    wget gnupg apt-transport-https ca-certificates tzdata \
    && wget -qO - "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x60c317803a41ba51845e371a1e9377a2ba9ef27f" | apt-key add - \
    && echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu ${VARIANT} main" > /etc/apt/sources.list.d/gcc.list \
    && apt-get update --quiet \
    && apt-get install --yes --quiet --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/* \
# install cmake
    && wget -qO - ${CMAKE_URL} | tar --strip-components=1 -xz -C /usr/local

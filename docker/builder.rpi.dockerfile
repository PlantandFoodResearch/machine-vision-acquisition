# bullseye==11
ARG DISTRO=bullseye
ARG ARAVIS_URL=https://github.com/AravisProject/aravis/releases/download/0.8.21/aravis-0.8.21.tar.xz

#  FROM ubuntu:${DISTRO} as cmake-gcc  # If you don't want the dev container components
FROM debian:${DISTRO}-slim as cmake-gcc
ARG DISTRO
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
ARG DEBIAN_FRONTEND=noninteractive

ENV \
    TZ=Pacific/Auckland \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update --quiet \
    # Ensure Timezone is set
    && ln -f -s /usr/share/zoneinfo/${TZ} /etc/localtime \
    && apt-get upgrade --yes --quiet \
    && apt-get install --yes --quiet --no-install-recommends \
        wget \
        gnupg \
        apt-transport-https \
        ca-certificates \
        tzdata \
        software-properties-common \
        lsb-release \
        # Some tools still require build-essential
        build-essential \
        cmake \
        git \
        ninja-build \
        gdb \
        python3 \
        python3-distutils \
        python3-dev \
    && c++ --version \
    && python3 --version \
    # Python 3.8
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && apt-get --yes autoremove \
    && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Use tini as entrypoint
RUN apt-get update --quiet \
    && apt-get install --quiet --yes tini \
    && apt-get clean autoclean
ENTRYPOINT ["/usr/bin/tini", "--"]


FROM cmake-gcc as base-dev
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


FROM base-dev as aravis-dev
# Download and unpack mvGenTL
ARG ARAVIS_URL=https://github.com/AravisProject/aravis/releases/download/0.8.21/aravis-0.8.21.tar.xz
RUN mkdir -p /opt/src \
    && apt-get update --quiet \
    && apt-get install --yes \
        libxml2-dev \
        gettext \
        libglib2.0-dev \
        libusb-1.0-0-dev \
        gobject-introspection \
        libgtk-3-dev \
        gtk-doc-tools \
        xsltproc \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        # libgstreamer-plugins-good1.0-dev \  Missing on debian. Will have to investigate.
        libgstreamer-plugins-bad1.0-dev \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-gtk3 \
        dbus-x11 \
        libgirepository1.0-dev \
        # Probably already installed, but better safe than sorry
        ninja-build \
        meson \
    && apt-get clean autoclean \
    # Needed for gi-docgen
    && python3 -m pip install jinja2 markdown markupsafe pygments toml typogrify

RUN mkdir -p /opt/src/aravis \
    && wget -qO - ${ARAVIS_URL} | tar --strip-components=1 -xJ -C /opt/src/aravis \
    && cd /opt/src/aravis \
    # Install library load paths
    && meson build \
    && cd build \
    && ninja \
    && ninja install \
    && ldconfig \
    && cd /

RUN python3 -m pip install -U pip setuptools wheel \
    # needs --ignore-installed because python3.6 interferes
    && python3 -m pip install --ignore-installed -U PyGObject pycairo PyGObject-stubs \
    && apt-get update --quiet \
    # Installing from pip attempts to build many things from source and this struggles on the Raspberry Pi (cv2, numpy, cmake,...)
    && apt-get install --yes python3-opencv python3-numpy \
    && mkdir -p /usr/lib/girepository-1.0/ \
    && ln -s /usr/local/lib/$(arch)-linux-gnu/girepository-1.0/Aravis-0.8.typelib /usr/lib/girepository-1.0/ \
    # Generate python stubs
    && wget -qO - https://raw.githubusercontent.com/pygobject/pygobject-stubs/master/tools/generate.py | \
    python3 - Aravis 0.8 >> /usr/local/lib/python3.9/dist-packages/gi-stubs/repository/Aravis.pyi

WORKDIR /src
USER vscode
CMD [ "bash", "-l"]


# mvIMPACT
FROM base-dev as mvIMPACT-dev
ARG MV_GENTL_ARCH=x86_64_ABI2
ARG MV_GENTL_URL=http://static.matrix-vision.com/mvIMPACT_Acquire/2.46.2/mvGenTL_Acquire-${MV_GENTL_ARCH}-2.46.2.tgz
# Todo: this needs to be better at being platform agnostic
ARG MV_GENTL_INSTALL_URL=http://static.matrix-vision.com/mvIMPACT_Acquire/2.46.2/install_mvGenTL_Acquire.sh

# Download and unpack mvIMPACT
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

WORKDIR /src
USER vscode
CMD [ "bash", "-l"]

# Future multi arch support notes:
# arch: armv7l
# dpkg --print-architecture: armhf
# uname -m: armv7l
# bullseye==11
ARG DISTRO=bullseye
ARG ARAVIS_URL=https://github.com/AravisProject/aravis/releases/download/0.8.22/aravis-0.8.22.tar.xz
ARG CHRONOPTICS_URL=https://storage.powerplant.pfr.co.nz/workspace/software_cache/chronoptic/3.0.1-January2022/tof-linux-x86_64.tar.gz


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


# Borrowed from VSCode's dev contianer setup
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

# Add piwheels to prevent/reduce compiling on-device (e.g. otherwise installing opencv can take days!)
RUN echo "[global]\n" \
         "extra-index-url=https://www.piwheels.org/simple\n"\ > /etc/pip.conf \
    # Enable poetry
    && python3 -m pip install -U poetry \
    # Disable virtual env creation (as we are inside docker)
    && poetry config virtualenvs.create false

CMD [ "sleep", "infinity" ]


FROM base-dev as aravis-dev
# Download, build, install Aravis
ARG ARAVIS_URL
RUN mkdir -p /opt/src \
    && apt-get update --quiet \
    && apt-get --no-install-recommends install --yes \
        libxml2-dev \
        gettext \
        libglib2.0-dev \
        libusb-1.0-0-dev \
        gobject-introspection \
        libgtk-3-dev \
        gtk-doc-tools \
        xsltproc \
        # Gstreamer install from https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c#install-gstreamer-on-ubuntu-or-debian
        # Removed "gstreamer1.0-qt5 gstreamer1.0-pulseaudio", not needed / supported
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
        # This appears to possibly help arv-viewer
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

# Ensure Aravis Python bindings and type hints are working well
RUN python3 -m pip install -U pip setuptools wheel \
    && python3 -m pip install -U PyGObject pycairo PyGObject-stubs \
    # Force binary install of opencv since building it can take days!
    # Refer: https://www.piwheels.org/project/opencv-python/ & https://www.piwheels.org/project/numpy/
    && apt-get install --yes libva-drm2 libpangoft2-1.0-0 libxvidcore4 libxkbcommon0 libchromaprint1 libpgm-5.3-0 libopus0 libwayland-cursor0 libpango-1.0-0 libbluray2 libsnappy1v5 libxrandr2 libthai0 libzvbi0 libnorm1 libpixman-1-0 libzmq5 libx265-192 libgraphite2-3 libxdamage1 libwayland-client0 libgtk-3-0 libsrt1.4-gnutls libxcursor1 libx264-160 libspeex1 libswscale5 libdav1d4 libmp3lame0 libgsm1 libatspi2.0-0 libxcb-render0 libavformat58 libvdpau1 libgme0 libcodec2-0.9 libwebpmux3 libshine3 libvorbis0a libsoxr0 libdrm2 libva-x11-2 libcairo-gobject2 libavutil56 libxfixes3 libvorbisfile3 librabbitmq4 libxrender1 libsodium23 libharfbuzz0b libtwolame0 libswresample3 libavcodec58 libxcomposite1 libwavpack1 libogg0 libepoxy0 libvorbisenc2 libxi6 libatlas3-base libgfortran5 libvpx6 libcairo2 libudfread0 libatk1.0-0 libgdk-pixbuf-2.0-0 libdatrie1 libmpg123-0 libxinerama1 libopenjp2-7 libaom0 libva2 libopenmpt0 libpangocairo-1.0-0 libwayland-egl1 libatk-bridge2.0-0 libtheora0 ocl-icd-libopencl1 libxcb-shm0 librsvg2-2 libssh-gcrypt-4 libgfortran5 libatlas3-base \
    && apt-get clean autoclean \
    && python3 -m pip install -U --only-binary=:all: numpy opencv-python \
    && mkdir -p /usr/lib/girepository-1.0/ \
    && ln -s $(find /usr/local/lib/ -type f -name "Aravis-0.8.typelib") /usr/lib/girepository-1.0/ \
    # Generate python stubs
    && wget -qO - https://raw.githubusercontent.com/pygobject/pygobject-stubs/master/tools/generate.py | \
    python3 - Aravis 0.8 >> $(python3 -c 'import site; print(site.getsitepackages()[0])')/gi-stubs/repository/Aravis.pyi

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


# Uses the Aravis image as a starting point, so that must be build first!
FROM aravis-dev as multi-sdk-dev
LABEL Description="Debian Aravis + Chronoptics SDK development container"

# Download and chronoptics SDK
ARG CHRONOPTICS_URL
ARG TARGETPLATFORM
ENV CHRONOPTICS_ROOT="/opt/src/chronoptics"
USER root
RUN export ARCHITECTURE="$(arch)" \
    && echo "ARCHITECTURE: $ARCHITECTURE\nTARGETPLATFORM: $TARGETPLATFORM" \
    && if [ $ARCHITECTURE != "x86_64" ]; then \
        echo "$ARCHITECTURE not supported" && exit 127; \
    fi \
    && mkdir -p ${CHRONOPTICS_ROOT} \
    && wget --no-check-certificate -qO - ${CHRONOPTICS_URL} | tar -xz -C ${CHRONOPTICS_ROOT} \
    # Install library load paths
    && touch /etc/ld.so.conf.d/Chronoptics.conf \
    && echo ${CHRONOPTICS_ROOT}/lib >> /etc/ld.so.conf.d/Chronoptics.conf \
    && ldconfig
ENV PYTHONPATH="${PYTHONPATH}:${CHRONOPTICS_ROOT}/lib/python"
ENV PATH="${PATH}:${CHRONOPTICS_ROOT}/bin"

WORKDIR /src
USER vscode
CMD [ "bash", "-l"]


# Future multi arch support notes:
# https://docs.docker.com/engine/reference/builder/#automatic-platform-args-in-the-global-scope
# arch: armv7l
# dpkg --print-architecture: armhf
# uname -m: armv7l
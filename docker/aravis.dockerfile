FROM debian:11-slim

RUN apt-get update \
    && apt-get install --yes \
        # General tools
        git \
        wget \
        ca-certificates \
        cmake \
        ninja-build \
        aravis-tools \
        build-essential \
        # Aravis: https://packages.debian.org/source/sid/aravis
        aravis-tools-cli \
        libaravis-0.8-0 \
        libaravis-dev \
        gir1.2-aravis-0.8 \
        # gstreamer for aravis
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        # Python GObject: https://pygobject.readthedocs.io/en/latest/getting_started.html
        libgirepository1.0-dev \
        libcairo2-dev \
        pkg-config \
        python3-dev \
        gir1.2-gtk-3.0 \
    && wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3 /tmp/get-pip.py \
    && python3 -m pip install -U pip setuptools wheel pycairo PyGObject opencv-python numpy \
    && apt-get --yes autoremove \
    && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

WORKDIR /src

# Use tini as entrypoint
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y \
    dbus-x11 \
    xfce4 \
    xfce4-clipman-plugin \
    xfce4-cpugraph-plugin \
    xfce4-netload-plugin \
    xfce4-screenshooter \
    xfce4-taskmanager \
    xfce4-terminal \
    xfce4-xkb-plugin 

RUN apt-get install -y \
    sudo \
    wget \
    xorgxrdp \
    xrdp && \
    apt remove -y light-locker xscreensaver && \
    apt autoremove -y && \
    rm -rf /var/cache/apt /var/lib/apt/lists

COPY ./xrdp_files/ubuntu-run.sh /usr/bin/
RUN mv /usr/bin/ubuntu-run.sh /usr/bin/run.sh && chmod +x /usr/bin/run.sh

# https://github.com/danielguerra69/ubuntu-xrdp/blob/master/Dockerfile
RUN mkdir /var/run/dbus && \
    cp /etc/X11/xrdp/xorg.conf /etc/X11 && \
    sed -i "s/console/anybody/g" /etc/X11/Xwrapper.config && \
    sed -i "s/xrdp\/xorg/xorg/g" /etc/xrdp/sesman.ini && \
    sed -i "s/port=.*/port=13389/" /etc/xrdp/xrdp.ini && \
    echo "xfce4-session" >> /etc/skel/.Xsession

# Docker config
EXPOSE 13389
ENTRYPOINT [ "/usr/bin/run.sh" ]
CMD [ "user", "user", "yes" ]
# CMD ["bash", "-c", "xrdp-sesman && xrdp -n"]

# Possible other
# apt-get install --yes
# net-tools
# python3-pyqt5 ffmpeg qttools5-dev libswscale-dev libavcodec-dev libopencv-dev
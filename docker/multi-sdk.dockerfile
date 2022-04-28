ARG CHRONOPTICS_URL=https://storage.powerplant.pfr.co.nz/workspace/software_cache/chronoptic/3.0.1-January2022/tof-linux-x86_64.tar.gz

# Uses the Aravis image as a starting point, so that must be build first!
FROM machine-vision-acquisition_aravis:latest
LABEL Description="Multiple machine vision SDKs in one image"
USER root

# Download and chronoptics SDK
ARG CHRONOPTICS_URL
RUN mkdir -p /opt/src \
    && mkdir -p /opt/src/chronoptics \
    && wget --no-check-certificate -qO - ${CHRONOPTICS_URL} | tar -xz -C /opt/src/chronoptics \
    # Install library load paths
    && touch /etc/ld.so.conf.d/Chronoptics.conf \
    && echo /opt/src/chronoptics/lib >> /etc/ld.so.conf.d/Chronoptics.conf \
    && ldconfig \
    # Make python 3.8 default
    && ln -f -s /usr/bin/python3.8 /usr/bin/python3
ENV CHRONOPTICS_ROOT="/opt/src/chronoptics"
ENV PYTHONPATH "${PYTHONPATH}:/opt/src/chronoptics/lib/python"
ENV PATH "${PATH}:/opt/src/chronoptics/bin"

USER vscode
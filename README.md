# Machine Vision Acquisition
This repository is a lose grouping of code, environments, scripts, notes, and more capturing efforts relating to acquiring images from machine vision cameras such as those from Lucid. Predominantly this will focus on GenICam as an interface layer and utilise [Aravis](https://github.com/AravisProject/aravis).

## Setup / Install
This repository is mainly a mix of Python and Docker. It uses *Poetry* and *Docker Compose*. The user should be somewhat familiar with these.

### Quick Start
**Note:** You won't have access to Aravis unless you manually install or you use the Docker workflow suggested.
```bash
poetry install
# Or if you're having poetry issues, nut your pip is uptodate:
python3 -m venv .venv # Ensure this is > python3.8
. .venv/bin/activate
pip install -e .
# If you plan to use stereo aspects:
pip install -e .[stereo]
```

## Docker Environments
Since many manufacturers of cameras offer thier individual SDKs; this repository attempts to *capture* the setup of a development environment. Some are vendor specific (e.g. FLIR's Spinnaker), some more vendor agnostic (Aravis or mvIMPACT).

If possible, the [*Aravis*](https://github.com/AravisProject/aravis) environment is a good starting point. This is an open-source vendor agnostic GenICam compliant acquisition framework.

### Docker: Quick start
Within in appropriate environment (i.e. docker host and CLI available) and the root of this folder, run:
```bash
# Optional: Enable build kit on old docker versions
export DOCKER_BUILDKIT=1
# multi-sdk depends on aravis
docker compose build aravis && docker compose build multi-sdk
# Optionally explicitly control output directory
OUTPUT_DIR=/media/powerplant-sink/ docker compose up -d aravis multi-sdk
# Otherwise:
docker compose up -d aravis multi-sdk
# Give the containers a few seconds to perform the initial startup commands. Then you can launch a shell into the container with:
docker compose exec multi-sdk bash
# List cameras (with Aravis, sudo for USB3)
docker compose exec aravis arv-tool-0.8
```

### Docker: Aravis
The Aravis development environment has two main flavors:
* `aravis` Built from source
* `aravis-deb` Debian package installed (in [docker/docker-compose.extras.yml](docker/docker-compose.extras.yml))

### Docker: Multi-SDK
This is basically the Aravis container but with some additonal other SDKs. This should be the main container used (on platforms that support it). Currently included are:
* Aravis (built from source)
* Chronoptics

### Output Volume
The containers will mount the value of the environment variable `OUTPUT_DIR` to `/output`. This will default to the path `/media/powerplant-sink/` if not defined or empty. To use this follow the example:
```bash
# Setup lsyncd or other syncrhonising on host at a specific point. E.g. "/media/powerplant-sink/"
export OUTPUT_DIR=/media/powerplant-sink/
docker compose up -d aravis
```

## GenICam Interfaces & SDKs
Whilst it is generally best to look directly at the vendor's documentation, these sections attempt to capture some tips and tricks.

### SDK: [*Aravis*](https://github.com/AravisProject/aravis)
* Vendor Agnostic
* Open Source
* Resonably well featured
* C++ & Python interface (via GObject)
#### **arv-viewer**: A quick camera viewer with limited features.
* e.g. `DISPLAY=localhost:10.0 LIBGL_ALWAYS_INDIRECT=1 arv-viewer-0.8 --auto-buffer-size` allows remote camera viewing with x11 forwarding.
* Struggles with super high datarates, but can do 1FPS 20MP or 45FPS normal cameras.
#### **Commandline tools**
##### `arv-tool`: List & change camera settings
* e.g. list cameras with:
```bash
> arv-tool-0.8 
Chronoptics-KeaC1RevB/6mm/NIFilter/RGB-2020029 (169.254.27.139)
Lucid Vision Labs-ATL314S-C-220700207 (192.168.1.102)
Lucid Vision Labs-TRI023S-C-213902307 (192.168.1.101)
Lucid Vision Labs-TRI023S-C-213902309 (192.168.1.100)
```
* e.g. reset a camera with:
```bash
# Find appropriate GenICam command
> arv-tool-0.8 --name "Lucid Vision Labs-TRI023S-C-213902307" features | grep Reset
        Command      : [WO] 'DeviceReset'
        Command      : [WO] 'DeviceFactoryReset'
        Command      : [WO] 'TimestampReset'
            EnumEntry   : 'CounterResetSource'
            EnumEntry   : 'CounterResetActivation'
              * CounterResetSource
              * CounterResetActivation
              * CounterReset
              * CounterValueAtReset
# Call GenICam command
> arv-tool-0.8 --name "Lucid Vision Labs-TRI023S-C-213902307" control DeviceFactoryReset
DeviceFactoryReset executed
```

## Tools
To use these tools, if they are python, ensure you have the dependencies installed. Either run `python3 -m poetry install` (or if not using poetry: `python3 -m pip install -e .`) in the repo root or manually do this. Use a virtual environment if not in a Docker container.

In general the tools are called `mva_*` and use `click` with sub-commands.

### **mva_capture**: Generic Camera Capture CLI interface
Mainly in [capture](./src/machine_vision_acquisition_python/capture/capture.py). View CLI help for info: `mva_capture --help`

This tool uses a JSON config file. See [capture-config.json](etc/capture-config.json) for an example. Create a copy, and modify it to your needs. Then run mva_capture:
```bash
HTTP_PORT=5000 mva_capture --config ./etc/capture-config.json
```

### **mva_process**: CLI Batch image file processor
Mainly in [process.cli](./src/machine_vision_acquisition_python/process/cli.py). View CLI help for info:
```
Usage: mva_process [OPTIONS] COMMAND [ARGS]...

  A set of tools to work on folders of images. Check the help for each sub-
  command for details.

Options:
  --help  Show this message and exit.

Commands:
  convert    Batch converts raw 12bit 'PNG' images to de-bayered 12bit...
  stats      Generate basic numerical stats from folders of images...
  undistort  Rectify images using CalibIO and OpenCV from a single camera.
  stereo     (Experimental) Process left/right image pairs to produce disparity based outputs.
```
**Note**: This is fairly functional without Aravis and so can run outside of Docker/Aravis

**Note**: `convert` can also tonemap 12b images to 8b.

#### Examples
```bash
#Stereo:
mva_process stereo --input /mnt/powerplant/input/projects/dhs/smartsensingandimaging/development/fops/2022-07-21/2 --serial-left=213500023 --serial-right=213500031 --calibio-json /mnt/powerplant/input/projects/dhs/smartsensingandimaging/development/fops/2022-04-29/calibration-images/caloutput.json --output ./tmp/stereo/2022-07-21/disp340-960-16bout/2/ --disparity-max 960
```

### Systemd service to restart DHCP and NMCLI connections
To mitigate issues with the devices not being stable, a quick helper service was created.
#### Install
First, ensure the devices in [etc/nmcli-dhcp-manager.service](etc/nmcli-dhcp-manager.service) match those on the system. Use `ip link` to check.
```bash
# In a sudo shell (i.e. sudo -E su)
mkdir -p /opt/nmcli-dhcp-manager
python3.8 -m venv /opt/nmcli-dhcp-manager/.venv
cp ./etc/nmcli-dhcp-manager.service /etc/systemd/system/
cp ./src/utils/nmcli-dhcp-manager.py /opt/nmcli-dhcp-manager/nmcli-dhcp-manager.py
/opt/nmcli-dhcp-manager/.venv/bin/python -m pip install -U pip setuptools wheel
/opt/nmcli-dhcp-manager/.venv/bin/python -m pip install -r ./src/utils/requirements.nmcli-dhcp-manager.txt
systemctl daemon-reload
systemctl enable nmcli-dhcp-manager
systemctl start nmcli-dhcp-manager
```

# Troubleshooting & FAQ

## Sudo-less USB3 cameras:
Follow the `udev` advice here: https://aravisproject.github.io/aravis/usb.html

### Can't see USB3 camera?
* Often you must use `sudo -E` to access USB devices. Try this first.
* Replugged (or reset) USB devices are not being refreshed in docker container. This should have been fixed, otherwise: `https://www.balena.io/docs/reference/base-images/base-images/#working-with-dynamically-plugged-devices` might serve as a starting point. Or `https://github.com/moby/moby/issues/35359`. Or `https://forums.docker.com/t/usb-device-not-working-not-sure-why/1143/3`

### FPNGE Bug:
You currently must manually install FPNGE:
```bash
pip install https://github.com/animetosho/python-fpnge/tarball/master
```

### HSM Stereo:
The original repo is not PIP-installable :(
```
A fork has been created with some fixes and pip installable
pip install git+https://github.com/nznobody/high-res-stereo
# download the model

wget http://www.contrib.andrew.cmu.edu/~gengshay/wordpress/wp-content/uploads/2020/01/final-768px.tar -O ./tmp/middlebury-final-768px.tar

export HSM_MODEL_PATH=$(readlink -f ./tmp/middlebury-final-768px.tar)
```
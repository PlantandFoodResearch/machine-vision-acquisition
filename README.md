# Machine Vision Acquisition
This repository is a lose grouping of code, environments, scripts, notes, and more capturing efforts relating to acquiring images from machine vision cameras such as those from Lucid. Predominantly this will focus on GenICam as an interface layer.

## Docker Environments
Since many manufacturers of cameras offer thier individual SDKs; this repository attempts to *capture* the setup of a development environment. Some are vendor specific (e.g. FLIR's Spinnaker), some vendor agnostic (Aravis or mvIMPACT).

The [*Aravis*](https://github.com/AravisProject/aravis) environment is a good starting point. This is an open-source vendor agnostic GenICam compliant acquisition framework.

### Docker: Aravis
The Aravis development environment has two main flavors:
* `aravis` Built from source
* `aravis-deb` Debian package installed

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
To use these tools, if they are python, ensure you have the dependencies installed. Either run `python3.8 -m pip install -e .` in the repo root or manually do this. Use a virtual environment if not in a Docker container.

### CLI Image Viewer and Save tool
Mainly in [cli.py](./src/machine_vision_acquisition_python/viewer/cli.py). View CLI help for info:
```
python3.8 -m machine_vision_acquisition_python.viewer --help
```
Example to run with all cameras and forward display to x11:
```bash
DISPLAY=localhost:10.0 LIBGL_ALWAYS_INDIRECT=1 python3.8 -m machine_vision_acquisition_python.viewer --all --out-dir=./tmp/data-root/manu/
# Example output if you pressed 'n' followed by 's', then 'q' to quit. "169.254.27.139" is the Chronoptics camera and can safely be ignored.
python3.8 -m machine_vision_acquisition_python.viewer --all --out-dir=./tmp/data-root/manu/
ERROR:machine_vision_acquisition_python.viewer.cli:Could not open camera
Traceback (most recent call last):
  File "/src/src/machine_vision_acquisition_python/viewer/cli.py", line 70, in __init__
    self.camera: Aravis.Camera = Aravis.Camera.new(name)
gi.repository.GLib.GError: arv-device-error-quark: Can't connect to device at address '169.254.27.139' (6)
INFO:machine_vision_acquisition_python.viewer.cli:Opened ATL314S-C-220700207
INFO:machine_vision_acquisition_python.viewer.cli:Opened TRI023S-C-213902307
INFO:machine_vision_acquisition_python.viewer.cli:Opened TRI023S-C-213902309
DEBUG:machine_vision_acquisition_python.viewer.cli:Acquiring image for ATL314S-C-220700207 took: 0.3851250330917537
DEBUG:machine_vision_acquisition_python.viewer.cli:Acquiring image for TRI023S-C-213902307 took: 0.09678720706142485
DEBUG:machine_vision_acquisition_python.viewer.cli:Acquiring image for TRI023S-C-213902309 took: 0.09508740995079279
DEBUG:machine_vision_acquisition_python.viewer.cli:Saved /src/tmp/data-root/manu/2022-04-14T164139-ATL314S-C-220700207-snapshot-1.png
DEBUG:machine_vision_acquisition_python.viewer.cli:Saved /src/tmp/data-root/manu/2022-04-14T164139-TRI023S-C-213902307-snapshot-1.png
DEBUG:machine_vision_acquisition_python.viewer.cli:Saved /src/tmp/data-root/manu/2022-04-14T164139-TRI023S-C-213902309-snapshot-1.png
```

### Chronoptics ToF Camera viewer
Mainly in [tof.py](./src/machine_vision_acquisition_python/viewer/tof.py). View CLI help for info:
```
python3.8 -m machine_vision_acquisition_python.viewer.tof --help
```
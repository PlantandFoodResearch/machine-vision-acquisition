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
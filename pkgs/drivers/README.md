# CC Hardware Drivers

This package contains drivers for interfacing with hardware.

## Driver API

## Supported Drivers

### TMF8828

> [!NOTE]
> By default, the device location for the TMF8828 is set to
> `/usr/local/dev/arduino-tmf8828`. This is an atypical location, but makes it easier
> from the code side so we can just manually symlink the device to this location. To
> do this, run the following command:
> ```bash
> sudo ln -s <actual location> /usr/local/dev/arduino-tmf8828
> ```

### PySpin Installation Instructions

Install PySpin and Spinnaker [as usual](https://www.flir.co.uk/products/spinnaker-sdk).
As of writing (2024-09-21), PySpin only supports <= 3.10. To install PySpin on newer
versions of Python, you can use the following steps:

```bash
# After installing Spinnaker, you're instructed to run the following command:
tar -xvzf spinnaker_python-<version>-cp<version>-<os>-<version>-<arch>.tar.gz
pip install spinnaker_python-<version>-cp<version>-<os>-<version>-<arch>.whl

# But this will fail for python versions > 3.10. To install on newer versions,
# replace the cp<version> with your python version. For instance, for python 3.11 on
# M2 Mac, the command would turn from
tar -xvzf spinnaker_python-4.1.0.172-cp310-cp310-macosx_13_0_arm64.tar.gz
pip instal spinnaker_python-4.1.0.172-cp310-cp310-macosx_13_0_arm64.whl
# To
tar -xvzf spinnaker_python-4.1.0.172-cp310-cp310-macosx_13_0_arm64.tar.gz
mv spinnaker_python-4.1.0.172-cp310-cp310-macosx_13_0_arm64.whl \
    spinnaker_python-4.1.0.172-cp311-cp311-macosx_13_0_arm64.whl
pip install spinnaker_python-4.1.0.172-cp311-cp311-macosx_13_0_arm64.whl

# And then go to your site packages and do
mv _PySpin.cpython-310-darwin.so _PySpin.cpython-311-darwin.so
```
